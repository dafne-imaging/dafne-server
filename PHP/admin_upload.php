<?php
declare(strict_types=1);

/**
 * Model upload endpoint — web UI + programmatic API.
 *
 * ── Programmatic usage ───────────────────────────────────────────────────────
 *
 *   POST multipart/form-data to this URL with:
 *
 *     api_key       (string)  Admin API key (required)
 *     model_type    (string)  Model name, e.g. "Thigh" – [a-zA-Z0-9_-]+
 *     model_binary  (file)    The canonical .model file
 *     model_json    (file)    The model.json metadata file (must be valid JSON)
 *
 *   Returns JSON:
 *     200  { "message": "…", "model_type": "…", "timestamp": 123, "sha256": "…", "already_existed": false }
 *     400  { "message": "validation error" }
 *     401  { "message": "invalid access code" }
 *     500  { "message": "server error" }
 *
 * ── Python example ───────────────────────────────────────────────────────────
 *
 *   import requests
 *
 *   r = requests.post(
 *       'https://your-server/admin_upload.php',
 *       data={'api_key': 'YOUR_ADMIN_KEY', 'model_type': 'Thigh'},
 *       files={
 *           'model_binary': ('canonical.model', open('canonical.model', 'rb')),
 *           'model_json':   ('model.json',      open('model.json',      'rb')),
 *       },
 *   )
 *   r.raise_for_status()
 *   print(r.json())
 */

session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/logger.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';

// ---------------------------------------------------------------------------
// Request-mode detection
//
// API mode  : request carries an api_key POST field (Python scripts, curl, …)
// UI mode   : session-authenticated browser request
// ---------------------------------------------------------------------------

$content_type = $_SERVER['CONTENT_TYPE'] ?? '';
$is_multipart = str_contains($content_type, 'multipart/form-data');

// For JSON bodies (unlikely for file uploads, but handle gracefully)
if (!$is_multipart && $_SERVER['REQUEST_METHOD'] === 'POST') {
    $raw  = (string) file_get_contents('php://input');
    $body = json_decode($raw, true) ?? [];
} else {
    $body = $_POST;
}

$is_api_request = isset($body['api_key']) && $body['api_key'] !== '';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function json_respond(array $data, int $status = 200): never
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($data);
    exit;
}

// ---------------------------------------------------------------------------
// CSRF (UI mode only)
// ---------------------------------------------------------------------------

if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['csrf_token'];

// ---------------------------------------------------------------------------
// Flash messages (UI mode only)
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
// Core upload logic (shared by both modes)
// ---------------------------------------------------------------------------

/**
 * Validate uploaded files, create the model directory tree, and place the
 * files in the correct locations.  Throws on any error.
 *
 * @return array{model_type:string, timestamp:int, sha256:string, already_existed:bool}
 */
function perform_model_upload(
    string $model_type,
    array  $model_file,   // entry from $_FILES['model_binary']
    array  $json_file,    // entry from $_FILES['model_json']
    string $uploaded_by
): array {
    // Validate model type (redundant if caller already sanitised, but be safe)
    if (sanitize_model_type($model_type) === null) {
        throw new InvalidArgumentException('Invalid model type name.');
    }

    // File upload errors
    if ($model_file['error'] !== UPLOAD_ERR_OK) {
        throw new InvalidArgumentException('Model file upload error (code ' . $model_file['error'] . ').');
    }
    if ($json_file['error'] !== UPLOAD_ERR_OK) {
        throw new InvalidArgumentException('JSON file upload error (code ' . $json_file['error'] . ').');
    }

    // Extension checks
    $model_ext = strtolower(pathinfo($model_file['name'], PATHINFO_EXTENSION));
    $json_ext  = strtolower(pathinfo($json_file['name'],  PATHINFO_EXTENSION));
    if ($model_ext !== 'model') {
        throw new InvalidArgumentException('Model file must have a .model extension.');
    }
    if ($json_ext !== 'json') {
        throw new InvalidArgumentException('Metadata file must have a .json extension.');
    }

    // Validate JSON content before touching the filesystem
    $json_raw = file_get_contents($json_file['tmp_name']);
    if ($json_raw === false || json_decode($json_raw) === null) {
        throw new InvalidArgumentException('model.json is not valid JSON.');
    }

    // Directory setup
    $model_dir       = MODELS_DIR . "/{$model_type}";
    $uploads_dir     = $model_dir . '/uploads';
    $already_existed = is_dir($model_dir);

    foreach ([$model_dir, $uploads_dir] as $dir) {
        if (!is_dir($dir) && !mkdir($dir, 0755, true)) {
            throw new RuntimeException("Could not create directory: {$dir}");
        }
    }

    // Save model binary as a new timestamped canonical model
    $timestamp  = time();
    $model_path = "{$model_dir}/{$timestamp}.model";

    if (!move_uploaded_file($model_file['tmp_name'], $model_path)) {
        throw new RuntimeException('Failed to save model binary.');
    }

    // SHA-256 sidecar (allows info_model to serve the hash without re-hashing)
    $sha256 = (string) hash_file('sha256', $model_path);
    if (file_put_contents("{$model_path}.sha256", $sha256) === false) {
        // Non-fatal: hash will be computed on-demand if sidecar is missing
        server_log("upload_new_model: warning – could not write sha256 sidecar for {$model_path}");
    }

    // Save / overwrite model.json
    $json_path = "{$model_dir}/model.json";
    if (!move_uploaded_file($json_file['tmp_name'], $json_path)) {
        // Roll back the model binary
        @unlink($model_path);
        @unlink("{$model_path}.sha256");
        throw new RuntimeException('Failed to save model.json.');
    }

    server_log(
        "upload_new_model: {$model_type}/{$timestamp}.model saved by {$uploaded_by}" .
        ($already_existed ? ' (model type already existed)' : ' (new model type)'),
        true
    );

    return [
        'model_type'      => $model_type,
        'timestamp'       => $timestamp,
        'sha256'          => $sha256,
        'already_existed' => $already_existed,
    ];
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

$logged_in_key    = $_SESSION['admin_api_key'] ?? null;
$current_user     = $logged_in_key !== null ? get_user_by_key($logged_in_key) : null;
$is_authenticated = $current_user !== null && (bool) $current_user['is_admin'];

// ---------------------------------------------------------------------------
// POST handling
// ---------------------------------------------------------------------------

$login_error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $body['action'] ?? '';

    // ── API mode ─────────────────────────────────────────────────────────
    if ($is_api_request) {
        $api_key = trim($body['api_key']);
        $user    = get_user_by_key($api_key);

        if ($user === null || !(bool) $user['is_admin']) {
            json_respond(['message' => 'invalid access code'], 401);
        }

        $model_type = sanitize_model_type(trim($body['model_type'] ?? ''));
        if ($model_type === null || $model_type === '') {
            json_respond(['message' => 'model_type is required and must match [a-zA-Z0-9_-]+'], 400);
        }

        if (
            !isset($_FILES['model_binary']) ||
            !isset($_FILES['model_json'])
        ) {
            json_respond(['message' => 'Both model_binary and model_json file fields are required'], 400);
        }

        try {
            $result = perform_model_upload(
                $model_type,
                $_FILES['model_binary'],
                $_FILES['model_json'],
                $user['name']
            );
            json_respond(array_merge(
                ['message' => 'Model uploaded successfully'],
                $result
            ));
        } catch (InvalidArgumentException $e) {
            json_respond(['message' => $e->getMessage()], 400);
        } catch (Throwable $e) {
            json_respond(['message' => 'Server error: ' . $e->getMessage()], 500);
        }
    }

    // ── UI mode ───────────────────────────────────────────────────────────

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
        // CSRF check only in authenticated UI mode
        if (!hash_equals($csrf_token, $_POST['csrf_token'] ?? '')) {
            http_response_code(403);
            die('Invalid CSRF token.');
        }

        if ($action === 'upload_model') {
            $model_type = sanitize_model_type(trim($_POST['model_type'] ?? ''));

            if ($model_type === null || $model_type === '') {
                flash('error', 'Model name is required and must only contain letters, numbers, underscores, and hyphens.');
            } elseif (!isset($_FILES['model_binary']) || !isset($_FILES['model_json'])) {
                flash('error', 'Both files are required.');
            } else {
                try {
                    $result = perform_model_upload(
                        $model_type,
                        $_FILES['model_binary'],
                        $_FILES['model_json'],
                        $current_user['name']
                    );
                    $notice = $result['already_existed']
                        ? "New canonical model added to existing type \"{$model_type}\" and model.json updated."
                        : "Model type \"{$model_type}\" created successfully.";
                    flash('success', $notice);
                    flash('key', json_encode($result)); // pass result through flash for display
                } catch (InvalidArgumentException $e) {
                    flash('error', $e->getMessage());
                } catch (Throwable $e) {
                    flash('error', 'Upload failed: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }
    }
}

// ---------------------------------------------------------------------------
// Render data
// ---------------------------------------------------------------------------

$existing_models = $is_authenticated ? get_model_types() : [];
$flashes         = pop_flashes();

// Separate the result flash from display flashes
$upload_result = null;
$display_flashes = [];
foreach ($flashes as $f) {
    if ($f['type'] === 'key') {
        $upload_result = json_decode($f['msg'], true);
    } else {
        $display_flashes[] = $f;
    }
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Upload Model</title>
<link rel="stylesheet" href="css/admin.css">
</head>
<body>

<?php if (!$is_authenticated): ?>

<div class="login-wrap">
  <div class="login-card">
    <h2>Dafne Server</h2>
    <p>Sign in with an administrator API key.</p>
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
      <a href="admin_upload.php" class="active">Upload</a>
    </nav>
  </div>
  <form method="post" style="margin:0">
    <input type="hidden" name="action" value="logout">
    <button type="submit" class="btn btn-outline btn-sm" style="color:#cbd5e1;border-color:#3a5a7c">
      Sign Out (<?= h($current_user['name']) ?>)
    </button>
  </form>
</div>

<div class="container container-narrow">

  <?php foreach ($display_flashes as $f): ?>
    <div class="flash <?= h($f['type']) ?>"><?= h($f['msg']) ?></div>
  <?php endforeach ?>

  <?php if ($upload_result): ?>
  <div class="result-banner">
    <h3>Upload successful</h3>
    <div class="result-row">
      <span class="result-label">Model type</span>
      <span class="result-val"><?= h($upload_result['model_type']) ?></span>
    </div>
    <div class="result-row">
      <span class="result-label">Timestamp</span>
      <span class="result-val"><?= (int) $upload_result['timestamp'] ?></span>
    </div>
    <div class="result-row">
      <span class="result-label">SHA-256</span>
      <span class="result-val"><?= h($upload_result['sha256']) ?></span>
    </div>
    <?php if ($upload_result['already_existed']): ?>
    <div style="margin-top:8px;font-size:12px;color:#854d0e;background:#fef9c3;border:1px solid #fde68a;border-radius:4px;padding:5px 10px">
      This model type already existed — a new canonical model was added and model.json was overwritten.
    </div>
    <?php endif ?>
  </div>
  <?php endif ?>

  <!-- Upload form -->
  <div class="card">
    <div class="card-header"><h2>Upload New Model</h2></div>
    <div class="card-body">
      <form method="post" enctype="multipart/form-data" id="upload-form">
        <input type="hidden" name="action"     value="upload_model">
        <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">

        <div class="form-row">
          <label for="model_type">
            Model Name
            <span class="hint">Letters, numbers, underscores and hyphens only — e.g. <code>Thigh</code>, <code>Hip_Left</code></span>
          </label>
          <input type="text" id="model_type" name="model_type"
                 pattern="[a-zA-Z0-9_\-]+" required
                 placeholder="e.g. Thigh"
                 value="<?= h($_POST['model_type'] ?? '') ?>">

          <?php if (!empty($existing_models)): ?>
          <div style="margin-top:8px;font-size:12px;color:#64748b">
            Existing models (click to fill):
            <div class="model-pills">
              <?php foreach ($existing_models as $mt): ?>
              <span class="model-pill" onclick="document.getElementById('model_type').value='<?= h(addslashes($mt)) ?>'"><?= h($mt) ?></span>
              <?php endforeach ?>
            </div>
          </div>
          <?php endif ?>
        </div>

        <div class="form-row">
          <label>Model Binary
            <span class="hint">The canonical <code>.model</code> file — will be stored as <code>{timestamp}.model</code></span>
          </label>
          <label class="file-input-wrap" id="model-wrap">
            <input type="file" name="model_binary" id="model_binary" accept=".model" required>
            <span class="file-input-icon">📦</span>
            <span class="file-input-text">
              <strong id="model-name">Choose .model file…</strong>
              Click to browse
            </span>
          </label>
        </div>

        <div class="form-row">
          <label>Model Metadata
            <span class="hint">The <code>model.json</code> file — will be saved as <code>model.json</code> (overwrites if exists)</span>
          </label>
          <label class="file-input-wrap" id="json-wrap">
            <input type="file" name="model_json" id="model_json" accept=".json,application/json" required>
            <span class="file-input-icon">📄</span>
            <span class="file-input-text">
              <strong id="json-name">Choose model.json…</strong>
              Click to browse
            </span>
          </label>
        </div>

        <button type="submit" class="btn btn-primary btn-lg">Upload Model</button>
      </form>
    </div>
  </div>

  <!-- API reference -->
  <div class="card">
    <div class="card-header"><h2>Programmatic API</h2></div>
    <div class="card-body" style="padding:0">
<pre class="api-ref"><span class="comment"># Python example — upload a model without using the browser UI</span>
<span class="comment"># Requires: pip install requests</span>

import requests

r = requests.post(
    <span class="val">'https://your-server/admin_upload.php'</span>,
    data={
        <span class="key">'api_key'</span>:    <span class="val">'YOUR_ADMIN_API_KEY'</span>,
        <span class="key">'model_type'</span>: <span class="val">'Thigh'</span>,
    },
    files={
        <span class="key">'model_binary'</span>: (<span class="val">'canonical.model'</span>, open(<span class="val">'canonical.model'</span>, <span class="val">'rb'</span>)),
        <span class="key">'model_json'</span>:   (<span class="val">'model.json'</span>,      open(<span class="val">'model.json'</span>,      <span class="val">'rb'</span>)),
    },
)
r.raise_for_status()
print(r.json())
<span class="comment"># {"message": "Model uploaded successfully", "model_type": "Thigh",</span>
<span class="comment">#  "timestamp": 1700000000, "sha256": "abc…", "already_existed": false}</span></pre>
    </div>
  </div>

</div><!-- /container -->

<script>
// Update file input labels when a file is chosen
function bindFileInput(inputId, wrapperId, nameId) {
    const input = document.getElementById(inputId);
    const wrap  = document.getElementById(wrapperId);
    const label = document.getElementById(nameId);
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            label.textContent = input.files[0].name;
            wrap.classList.add('file-selected');
        } else {
            label.textContent = input.getAttribute('placeholder') || 'Choose file…';
            wrap.classList.remove('file-selected');
        }
    });
}
bindFileInput('model_binary', 'model-wrap', 'model-name');
bindFileInput('model_json',   'json-wrap',  'json-name');
</script>

<?php endif ?>
</body>
</html>