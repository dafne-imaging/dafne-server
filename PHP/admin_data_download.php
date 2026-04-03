<?php
declare(strict_types=1);
session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/logger.php';
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
// Flash messages
// ---------------------------------------------------------------------------

function flash(string $type, string $msg): void
{
    $_SESSION['flash'][] = ['type' => $type, 'msg' => $msg];
}

function pop_flashes(): array
{
    $f = $_SESSION['flash'] ?? [];
    unset($_SESSION['flash']);
    return $f;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

/**
 * Validate a user subdirectory name: must match what sanitize_username_for_filename produces.
 */
function valid_upload_user(string $s): bool
{
    return $s !== '' && preg_match('/^[a-zA-Z0-9_\- ]+$/', $s) === 1;
}

/**
 * Validate an uploaded data filename: {digits}.npz only.
 */
function valid_upload_file(string $s): bool
{
    return preg_match('/^\d+\.npz$/', $s) === 1;
}

/**
 * Resolve and verify that $path is inside $base_dir (no path traversal).
 */
function safe_upload_path(string $username, string $filename): ?string
{
    if (!valid_upload_user($username) || !valid_upload_file($filename)) {
        return null;
    }
    $base = realpath(UPLOAD_DATA_DIR);
    if ($base === false) {
        return null;
    }
    $path = realpath($base . DIRECTORY_SEPARATOR . $username . DIRECTORY_SEPARATOR . $filename);
    if ($path === false || strpos($path, $base . DIRECTORY_SEPARATOR) !== 0) {
        return null;
    }
    return $path;
}

/**
 * Collect all upload files, grouped by username subfolder.
 * Returns: [ ['user' => string, 'files' => [ ['name'=>, 'size'=>, 'mtime'=>], … ]], … ]
 */
function list_uploads(): array
{
    $base = UPLOAD_DATA_DIR;
    if (!is_dir($base)) {
        return [];
    }
    $groups = [];
    foreach (scandir($base) as $entry) {
        if ($entry === '.' || $entry === '..') {
            continue;
        }
        $dir = $base . DIRECTORY_SEPARATOR . $entry;
        if (!is_dir($dir) || !valid_upload_user($entry)) {
            continue;
        }
        $files = [];
        foreach (scandir($dir) as $fname) {
            if (!valid_upload_file($fname)) {
                continue;
            }
            $fpath   = $dir . DIRECTORY_SEPARATOR . $fname;
            $files[] = [
                'name'  => $fname,
                'size'  => filesize($fpath),
                'mtime' => filemtime($fpath),
            ];
        }
        // newest first
        usort($files, fn($a, $b) => $b['mtime'] <=> $a['mtime']);
        if (!empty($files)) {
            $groups[] = ['user' => $entry, 'files' => $files];
        }
    }
    usort($groups, fn($a, $b) => strcmp($a['user'], $b['user']));
    return $groups;
}

function format_bytes(int $bytes): string
{
    if ($bytes >= 1_073_741_824) {
        return round($bytes / 1_073_741_824, 2) . ' GB';
    }
    if ($bytes >= 1_048_576) {
        return round($bytes / 1_048_576, 2) . ' MB';
    }
    if ($bytes >= 1_024) {
        return round($bytes / 1_024, 1) . ' KB';
    }
    return $bytes . ' B';
}

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

$logged_in_key    = $_SESSION['admin_api_key'] ?? null;
$current_user     = $logged_in_key !== null ? get_user_by_key($logged_in_key) : null;
$is_authenticated = $current_user !== null && (bool) $current_user['is_admin'];

// ---------------------------------------------------------------------------
// File download (GET — must come before any HTML output)
// ---------------------------------------------------------------------------

if ($is_authenticated && isset($_GET['download'])) {
    $dl_user = (string) ($_GET['user'] ?? '');
    $dl_file = (string) ($_GET['file'] ?? '');
    $path    = safe_upload_path($dl_user, $dl_file);

    if ($path === null || !is_file($path)) {
        http_response_code(404);
        die('File not found.');
    }

    header('Content-Type: application/octet-stream');
    header('Content-Disposition: attachment; filename="' . $dl_user . '_' . $dl_file . '"');
    header('Content-Length: ' . filesize($path));
    header('Cache-Control: no-cache, no-store');
    readfile($path);
    exit;
}

// ---------------------------------------------------------------------------
// POST handlers
// ---------------------------------------------------------------------------

$login_error = null;

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

    } elseif ($is_authenticated) {
        csrf_check();

        if ($action === 'delete_upload') {
            $del_user = (string) ($_POST['user'] ?? '');
            $del_file = (string) ($_POST['file'] ?? '');
            $path     = safe_upload_path($del_user, $del_file);

            if ($path === null || !is_file($path)) {
                flash('error', 'File not found.');
            } elseif (!unlink($path)) {
                flash('error', 'Failed to delete file.');
            } else {
                // Remove user directory if now empty
                $dir = dirname($path);
                $remaining = array_diff((array) scandir($dir), ['.', '..']);
                if (empty($remaining)) {
                    @rmdir($dir);
                }
                flash('success', "Deleted {$del_user}/{$del_file}.");
                server_log("Admin deleted upload {$del_user}/{$del_file}");
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }
    }
}

// ---------------------------------------------------------------------------
// Render data
// ---------------------------------------------------------------------------

$upload_groups = $is_authenticated ? list_uploads() : [];
$flashes       = pop_flashes();

// Total stats
$total_files = 0;
$total_bytes = 0;
foreach ($upload_groups as $g) {
    $total_files += count($g['files']);
    foreach ($g['files'] as $f) {
        $total_bytes += $f['size'];
    }
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Data Uploads</title>
<link rel="stylesheet" href="css/admin.css">
<style>
.uploads-table          { width:100%; border-collapse:collapse; font-size:13px; }
.uploads-table th,
.uploads-table td       { padding:7px 10px; border-bottom:1px solid #e2e8f0; vertical-align:middle; }
.uploads-table th       { background:#f8fafc; font-weight:600; color:#475569;
                          text-transform:uppercase; font-size:11px; letter-spacing:.04em; }
.uploads-table tr:last-child td { border-bottom:none; }
.uploads-table tr:hover td { background:#f1f5f9; }
.user-group-header      { background:#f1f5f9 !important; }
.user-group-header td   { padding:8px 10px; font-weight:600; color:#1e293b;
                          font-size:13px; border-bottom:2px solid #cbd5e1; }
.file-name              { font-family:monospace; font-size:12px; color:#334155; }
.file-size              { color:#64748b; white-space:nowrap; }
.file-ts                { color:#64748b; white-space:nowrap; font-size:12px; }
.actions                { display:flex; gap:6px; justify-content:flex-end; }
.stats-bar              { display:flex; gap:20px; font-size:13px; color:#64748b;
                          margin-bottom:4px; padding:8px 12px;
                          background:#f8fafc; border:1px solid #e2e8f0;
                          border-radius:6px; }
.stats-bar strong       { color:#1e293b; }
</style>
</head>
<body>

<?php if (!$is_authenticated): ?>

<div class="login-wrap">
  <div class="login-card">
    <h2>Dafne Server</h2>
    <p>Sign in with an administrator API key to manage data uploads.</p>
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
      <a href="admin_bulk_import.php">Bulk Import</a>
      <a href="admin_data_download.php" class="active">Data</a>
    </nav>
  </div>
  <form method="post" style="margin:0">
    <input type="hidden" name="action" value="logout">
    <button type="submit" class="btn btn-outline btn-sm" style="color:#cbd5e1;border-color:#3a5a7c">
      Sign Out (<?= h($current_user['name']) ?>)
    </button>
  </form>
</div>

<div class="container container-wide">

  <?php foreach ($flashes as $f): ?>
    <div class="flash <?= h($f['type']) ?>"><?= h($f['msg']) ?></div>
  <?php endforeach ?>

  <div class="card">
    <div class="card-header">
      <h2>Data Uploads</h2>
    </div>
    <div class="card-body">

      <?php if (empty($upload_groups)): ?>
        <p style="color:#64748b">No data uploads found.</p>
      <?php else: ?>

        <div class="stats-bar">
          <span><strong><?= count($upload_groups) ?></strong> user<?= count($upload_groups) !== 1 ? 's' : '' ?></span>
          <span><strong><?= $total_files ?></strong> file<?= $total_files !== 1 ? 's' : '' ?></span>
          <span><strong><?= h(format_bytes($total_bytes)) ?></strong> total</span>
        </div>

        <table class="uploads-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Size</th>
              <th>Uploaded</th>
              <th style="text-align:right">Actions</th>
            </tr>
          </thead>
          <tbody>
            <?php foreach ($upload_groups as $group):
                $uname = $group['user'];
            ?>
            <tr class="user-group-header">
              <td colspan="4"><?= h($uname) ?> &mdash; <?= count($group['files']) ?> file<?= count($group['files']) !== 1 ? 's' : '' ?></td>
            </tr>
            <?php foreach ($group['files'] as $file): ?>
            <tr>
              <td class="file-name"><?= h($file['name']) ?></td>
              <td class="file-size"><?= h(format_bytes((int) $file['size'])) ?></td>
              <td class="file-ts"><?= h(date('Y-m-d H:i:s', (int) $file['mtime'])) ?></td>
              <td>
                <div class="actions">
                  <a href="?download=1&amp;user=<?= urlencode($uname) ?>&amp;file=<?= urlencode($file['name']) ?>"
                     class="btn btn-outline btn-sm">Download</a>
                  <button type="button" class="btn btn-danger btn-sm"
                          onclick="confirmDelete(<?= h(json_encode($uname)) ?>, <?= h(json_encode($file['name'])) ?>)">
                    Delete
                  </button>
                </div>
              </td>
            </tr>
            <?php endforeach ?>
            <?php endforeach ?>
          </tbody>
        </table>

      <?php endif ?>
    </div>
  </div>

</div><!-- /container -->

<!-- Hidden delete form -->
<form id="delete-form" method="post" style="display:none">
  <input type="hidden" name="action"     value="delete_upload">
  <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
  <input type="hidden" name="user"       id="delete-user">
  <input type="hidden" name="file"       id="delete-file">
</form>

<script>
function confirmDelete(user, file) {
    if (!confirm('Delete ' + user + '/' + file + '?\n\nThis cannot be undone.')) return;
    document.getElementById('delete-user').value = user;
    document.getElementById('delete-file').value = file;
    document.getElementById('delete-form').submit();
}
</script>

<?php endif ?>
</body>
</html>