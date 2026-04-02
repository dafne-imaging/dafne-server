<?php
declare(strict_types=1);
session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';

// ---------------------------------------------------------------------------
// CSRF
// ---------------------------------------------------------------------------

if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['csrf_token'];

function csrf_check(): void
{
    if (!hash_equals($_SESSION['csrf_token'], $_POST['csrf_token'] ?? '')) {
        http_response_code(403);
        die('Invalid CSRF token.');
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function sanitize_posted_models(array $raw): array
{
    $out = [];
    foreach ($raw as $m) {
        $clean = sanitize_model_type((string) $m);
        if ($clean !== null) {
            $out[] = $clean;
        }
    }
    return array_values(array_unique($out));
}

/**
 * Parse a CSV file handle into rows of associative arrays.
 * The first row must contain headers (api_key, name, email).
 * Returns [rows[], error_string|null].
 */
function parse_csv_upload(string $tmp_path): array
{
    $fh = fopen($tmp_path, 'r');
    if ($fh === false) {
        return [[], 'Cannot open uploaded file.'];
    }

    // Detect BOM and skip it.
    $bom = fread($fh, 3);
    if ($bom !== "\xEF\xBB\xBF") {
        rewind($fh);
    }

    $header = fgetcsv($fh);
    if ($header === false || $header === null) {
        fclose($fh);
        return [[], 'The file appears to be empty.'];
    }

    // Normalise header names.
    $header = array_map(fn($h) => strtolower(trim((string) $h)), $header);
    $required = ['api_key', 'name'];
    foreach ($required as $col) {
        if (!in_array($col, $header, true)) {
            fclose($fh);
            return [[], "Missing required column \"{$col}\" in CSV header."];
        }
    }

    $rows = [];
    $line = 1; // 1-based data rows (header = 0)
    while (($row = fgetcsv($fh)) !== false) {
        $line++;
        if ($row === [null]) {
            continue; // skip blank lines
        }
        $assoc = [];
        foreach ($header as $i => $col) {
            $assoc[$col] = trim((string) ($row[$i] ?? ''));
        }
        $assoc['_line'] = $line;
        $rows[] = $assoc;
    }

    fclose($fh);

    if (empty($rows)) {
        return [[], 'The file contains a header but no data rows.'];
    }

    return [$rows, null];
}

/**
 * Insert one user and grant access permissions.
 * Returns null on success or an error message string.
 */
function import_one_user(string $api_key, string $name, string $email, array $access_models): ?string
{
    if ($api_key === '') {
        return 'API key is empty.';
    }
    if ($name === '') {
        return 'Name is empty.';
    }

    $db = get_db();

    try {
        $db->beginTransaction();

        $hash = hash_api_key($api_key);
        $stmt = $db->prepare('INSERT INTO users (name, email, api_key_hash) VALUES (?, ?, ?)');
        $stmt->execute([$name, $email, $hash]);
        $user_id = (int) $db->lastInsertId();

        $ins = $db->prepare('INSERT INTO users_accesspermissions (user_id, model_type) VALUES (?, ?)');
        foreach ($access_models as $mt) {
            $ins->execute([$user_id, $mt]);
        }

        $db->commit();
        return null;
    } catch (PDOException $e) {
        $db->rollBack();
        // Surface the most common constraint violation clearly.
        if ($e->getCode() === '23000') {
            $msg = $e->getMessage();
            if (stripos($msg, 'api_key_hash') !== false) {
                return 'Duplicate API key.';
            }
            if (stripos($msg, 'email') !== false) {
                return 'Email address already registered.';
            }
            return 'Duplicate value (constraint violation).';
        }
        return 'Database error: ' . $e->getMessage();
    }
}

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

$logged_in_key    = $_SESSION['admin_api_key'] ?? null;
$current_user     = $logged_in_key !== null ? get_user_by_key($logged_in_key) : null;
$is_authenticated = $current_user !== null && (bool) $current_user['is_admin'];

// ---------------------------------------------------------------------------
// POST handlers
// ---------------------------------------------------------------------------

$login_error = null;
$import_results = null; // array of result rows after a successful import attempt

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';

    if ($action === 'login') {
        $key  = trim($_POST['api_key'] ?? '');
        $user = get_user_by_key($key);
        if ($user !== null && (bool) $user['is_admin']) {
            $_SESSION['admin_api_key'] = $key;
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }
        $login_error = 'Invalid API key or insufficient permissions.';

    } elseif ($action === 'logout') {
        session_destroy();
        header('Location: ' . $_SERVER['PHP_SELF']);
        exit;

    } elseif ($is_authenticated && $action === 'import_csv') {
        csrf_check();

        $access_models = sanitize_posted_models((array) ($_POST['access_models'] ?? []));

        $upload = $_FILES['csv_file'] ?? null;
        if ($upload === null || $upload['error'] === UPLOAD_ERR_NO_FILE) {
            $_SESSION['flash'][] = ['type' => 'error', 'msg' => 'Please select a CSV file to upload.'];
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }
        if ($upload['error'] !== UPLOAD_ERR_OK) {
            $_SESSION['flash'][] = ['type' => 'error', 'msg' => 'File upload error (code ' . $upload['error'] . ').'];
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }

        [$rows, $parse_error] = parse_csv_upload($upload['tmp_name']);
        if ($parse_error !== null) {
            $_SESSION['flash'][] = ['type' => 'error', 'msg' => 'CSV parse error: ' . $parse_error];
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }

        $results = [];
        foreach ($rows as $row) {
            $api_key = $row['api_key']  ?? '';
            $name    = $row['name']     ?? '';
            $email   = $row['email']    ?? '';
            $line    = $row['_line'];

            $err = import_one_user($api_key, $name, $email, $access_models);
            $results[] = [
                'line'    => $line,
                'name'    => $name,
                'email'   => $email,
                'api_key' => $api_key,
                'ok'      => $err === null,
                'error'   => $err,
            ];
        }

        $ok_count   = count(array_filter($results, fn($r) => $r['ok']));
        $fail_count = count($results) - $ok_count;

        $summary = "Import complete: {$ok_count} user(s) created";
        if ($fail_count > 0) {
            $summary .= ", {$fail_count} failed";
        }
        $summary .= '.';

        $_SESSION['import_results'] = $results;
        $_SESSION['flash'][] = [
            'type' => $fail_count === 0 ? 'success' : ($ok_count === 0 ? 'error' : 'warning'),
            'msg'  => $summary,
        ];
        header('Location: ' . $_SERVER['PHP_SELF']);
        exit;
    }
}

// ---------------------------------------------------------------------------
// Collect render data
// ---------------------------------------------------------------------------

$model_types = $is_authenticated ? get_model_types() : [];

$flashes = $_SESSION['flash'] ?? [];
unset($_SESSION['flash']);

$import_results = $_SESSION['import_results'] ?? null;
unset($_SESSION['import_results']);

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Bulk User Import</title>
<link rel="stylesheet" href="css/admin.css">
<style>
  .result-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }
  .result-table th,
  .result-table td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }
  .result-table th { background: #f8fafc; font-weight: 600; color: #475569; }
  .result-table tr.ok   td:first-child { border-left: 3px solid #22c55e; }
  .result-table tr.fail td:first-child { border-left: 3px solid #ef4444; }
  .result-table .tag-ok   { color: #16a34a; font-weight: 600; }
  .result-table .tag-fail { color: #dc2626; font-weight: 600; }
  .result-table .err-msg  { color: #7f1d1d; font-size: 12px; }
  .flash.warning { background: #fef9c3; border-color: #ca8a04; color: #713f12; }
  .model-checks { display: flex; flex-wrap: wrap; gap: 10px 24px; margin-top: 10px; }
  .model-checks label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .csv-hint { font-size: 12px; color: #64748b; margin-top: 8px; line-height: 1.6; }
  .csv-hint code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>

<?php if (!$is_authenticated): ?>

<div class="login-wrap">
  <div class="login-card">
    <h2>Dafne Server</h2>
    <p>Sign in with an administrator API key to manage users.</p>
    <?php if ($login_error): ?>
      <div class="flash error"><?= h($login_error) ?></div>
    <?php endif ?>
    <form method="post">
      <input type="hidden" name="action" value="login">
      <div class="form-row">
        <label for="api_key">API Key</label>
        <input type="password" id="api_key" name="api_key" autocomplete="off" autofocus required>
      </div>
      <button type="submit" class="btn btn-primary" style="width:100%;margin-top:6px">Sign In</button>
    </form>
  </div>
</div>

<?php else: ?>

<div class="topbar">
  <div style="display:flex;align-items:center">
    <h1>Dafne Server</h1>
    <nav class="topnav">
      <a href="admin.php">Users</a>
      <a href="admin_models.php">Models</a>
      <a href="admin_upload.php">Upload</a>
      <a href="admin_log.php">Log</a>
      <a href="admin_bulk_import.php" class="active">Bulk Import</a>
    </nav>
  </div>
  <form method="post" style="margin:0">
    <input type="hidden" name="action" value="logout">
    <button type="submit" class="btn btn-outline btn-sm" style="color:#cbd5e1;border-color:#3a5a7c">
      Sign Out (<?= h($current_user['name']) ?>)
    </button>
  </form>
</div>

<div class="container container-narrow" style="max-width:740px">

  <?php foreach ($flashes as $f): ?>
    <div class="flash <?= h($f['type']) ?>"><?= h($f['msg']) ?></div>
  <?php endforeach ?>

  <!-- ================================================================ -->
  <!-- Import form                                                        -->
  <!-- ================================================================ -->
  <div class="card">
    <div class="card-header"><h2>Bulk User Import</h2></div>
    <div class="card-body">
      <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="action"     value="import_csv">
        <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">

        <div class="form-row">
          <label for="csv_file">CSV File</label>
          <input type="file" id="csv_file" name="csv_file" accept=".csv,text/csv" required>
        </div>

        <p class="csv-hint">
          The file must include a header row with at least the columns
          <code>api_key</code> and <code>name</code>.
          An <code>email</code> column is optional (leave blank or omit).
          Column order does not matter.<br>
          Example: <code>api_key,name,email</code>
        </p>

        <?php if (!empty($model_types)): ?>
        <div class="form-row" style="margin-top:18px">
          <label>Model Access</label>
          <div>
            <p style="color:#64748b;font-size:13px;margin-bottom:6px">
              Select which models all imported users will be granted access to:
            </p>
            <div class="model-checks">
              <?php foreach ($model_types as $mt): ?>
              <label>
                <input type="checkbox" name="access_models[]" value="<?= h($mt) ?>" checked>
                <?= h($mt) ?>
              </label>
              <?php endforeach ?>
            </div>
          </div>
        </div>
        <?php else: ?>
          <p class="no-models" style="margin-top:16px">
            No model types configured yet — users will be created without model access.
          </p>
        <?php endif ?>

        <div style="margin-top:20px">
          <button type="submit" class="btn btn-success">Import Users</button>
        </div>
      </form>
    </div>
  </div>

  <!-- ================================================================ -->
  <!-- Import results                                                     -->
  <!-- ================================================================ -->
  <?php if ($import_results !== null): ?>
  <div class="card">
    <div class="card-header"><h2>Import Results</h2></div>
    <div class="card-body" style="padding:0">
      <table class="result-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>Email</th>
            <th>API Key</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <?php foreach ($import_results as $r): ?>
          <tr class="<?= $r['ok'] ? 'ok' : 'fail' ?>">
            <td><?= (int) $r['line'] ?></td>
            <td><?= h($r['name']) ?></td>
            <td><?= $r['email'] !== '' ? h($r['email']) : '<em style="color:#94a3b8">—</em>' ?></td>
            <td><code><?= h($r['api_key']) ?></code></td>
            <td>
              <?php if ($r['ok']): ?>
                <span class="tag-ok">Created</span>
              <?php else: ?>
                <span class="tag-fail">Failed</span>
                <br><span class="err-msg"><?= h($r['error'] ?? '') ?></span>
              <?php endif ?>
            </td>
          </tr>
          <?php endforeach ?>
        </tbody>
      </table>
    </div>
  </div>
  <?php endif ?>

</div><!-- /container -->

<?php endif ?>
</body>
</html>