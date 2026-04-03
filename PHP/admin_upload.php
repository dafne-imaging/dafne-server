<?php
declare(strict_types=1);

/**
 * Model upload — web UI wrapper.
 *
 * The actual upload logic lives in api/upload_new_model.php.
 * For programmatic access, POST directly to /upload_new_model instead:
 *
 * ── Python example ───────────────────────────────────────────────────────────
 *
 *   import requests
 *
 *   r = requests.post(
 *       'https://your-server/upload_new_model',
 *       data={'api_key': 'YOUR_ADMIN_KEY', 'model_type': 'Thigh'},
 *       files={
 *           'model_binary': ('canonical.model', open('canonical.model', 'rb')),
 *           'model_json':   ('model.json',      open('model.json',      'rb')),
 *       },
 *   )
 *   r.raise_for_status()
 *   print(r.json())
 *   # {"message": "Model uploaded successfully", "model_type": "Thigh",
 *   #  "timestamp": 1700000000, "sha256": "abc…", "already_existed": false}
 */

session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/logger.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';
require_once __DIR__ . '/api/upload_new_model.php';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

// ---------------------------------------------------------------------------
// CSRF
// ---------------------------------------------------------------------------

if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['csrf_token'];

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
        // CSRF check
        if (!hash_equals($csrf_token, $_POST['csrf_token'] ?? '')) {
            http_response_code(403);
            die('Invalid CSRF token.');
        }

        if ($action === 'upload_model') {
            $model_type = sanitize_model_type(trim($_POST['model_type'] ?? ''));

            if ($model_type === null || $model_type === '') {
                flash('error', 'Model name is required and must only contain letters, numbers, and hyphens.');
            } elseif (!isset($_FILES['model_binary']) || !isset($_FILES['model_json'])) {
                flash('error', 'Both files are required.');
            } else {
                // Inject the session key so handle_upload_new_model can auth the user.
                $body   = ['api_key' => $logged_in_key, 'model_type' => $model_type];
                $result = handle_upload_new_model($body);

                $status = $result['__status'] ?? 200;
                unset($result['__status']);

                if ($status === 200) {
                    $notice = $result['already_existed']
                        ? "New canonical model added to existing type \"{$model_type}\" and model.json updated."
                        : "Model type \"{$model_type}\" created successfully.";
                    flash('success', $notice);
                    flash('key', json_encode($result));
                } else {
                    flash('error', $result['message'] ?? 'Upload failed.');
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
$upload_result   = null;
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
      <a href="admin_log.php">Log</a>
      <a href="admin_bulk_import.php">Bulk Import</a>
      <a href="admin_data_download.php">Data</a>
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

        <div id="upload-status"></div>
        <button type="submit" class="btn btn-primary btn-lg" id="upload-btn">Upload Model</button>
        <div id="progress-wrap" style="display:none;margin-top:14px">
          <div style="background:#e2e8f0;border-radius:4px;overflow:hidden;height:8px">
            <div id="progress-bar" style="background:#1a3a5c;height:100%;width:0;transition:width .25s ease"></div>
          </div>
          <p id="progress-label" style="margin-top:6px;font-size:12px;color:#64748b"></p>
        </div>
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
    <span class="val">'https://your-server/upload_new_model'</span>,
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
(function () {
'use strict';

// ---------------------------------------------------------------------------
// Constants (api_key is safe to expose here — the page is admin-only)
// ---------------------------------------------------------------------------
const CHUNK_SIZE = 50 * 1024 * 1024;  // 50 MiB
const API_KEY    = <?= json_encode($logged_in_key) ?>;
const UPLOAD_URL = <?= json_encode(rtrim(dirname($_SERVER['SCRIPT_NAME']), '/') . '/upload_new_model') ?>;

// ---------------------------------------------------------------------------
// File input label bindings
// ---------------------------------------------------------------------------
function bindFileInput(inputId, wrapperId, nameId) {
    const input = document.getElementById(inputId);
    const wrap  = document.getElementById(wrapperId);
    const label = document.getElementById(nameId);
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            label.textContent = input.files[0].name;
            wrap.classList.add('file-selected');
        } else {
            label.textContent = 'Choose file…';
            wrap.classList.remove('file-selected');
        }
    });
}
bindFileInput('model_binary', 'model-wrap', 'model-name');
bindFileInput('model_json',   'json-wrap',  'json-name');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function setProgress(fraction, label) {
    document.getElementById('progress-bar').style.width   = Math.round(fraction * 100) + '%';
    document.getElementById('progress-label').textContent = label;
}

function showStatus(html) {
    document.getElementById('upload-status').innerHTML = html;
}

function showResult(data) {
    document.getElementById('progress-wrap').style.display = 'none';
    const warning = data.already_existed
        ? `<div style="margin-top:8px;font-size:12px;color:#854d0e;background:#fef9c3;border:1px solid #fde68a;border-radius:4px;padding:5px 10px">
               This model type already existed — a new canonical model was added and model.json was overwritten.
           </div>`
        : '';
    showStatus(`
        <div class="result-banner" style="margin-bottom:14px">
            <h3>Upload successful</h3>
            <div class="result-row">
                <span class="result-label">Model type</span>
                <span class="result-val">${escHtml(data.model_type)}</span>
            </div>
            <div class="result-row">
                <span class="result-label">Timestamp</span>
                <span class="result-val">${escHtml(String(data.timestamp))}</span>
            </div>
            <div class="result-row">
                <span class="result-label">SHA-256</span>
                <span class="result-val">${escHtml(data.sha256)}</span>
            </div>
            ${warning}
        </div>`);

    const btn = document.getElementById('upload-btn');
    btn.disabled = false;

    // Reset the form
    document.getElementById('upload-form').reset();
    document.getElementById('model-name').textContent = 'Choose .model file…';
    document.getElementById('json-name').textContent  = 'Choose model.json…';
    document.getElementById('model-wrap').classList.remove('file-selected');
    document.getElementById('json-wrap').classList.remove('file-selected');
}

// ---------------------------------------------------------------------------
// Chunked upload
// ---------------------------------------------------------------------------
document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const modelType = document.getElementById('model_type').value.trim();
    const modelFile = document.getElementById('model_binary').files[0];
    const jsonFile  = document.getElementById('model_json').files[0];

    if (!modelType || !modelFile || !jsonFile) return;

    const btn         = document.getElementById('upload-btn');
    const progressWrap = document.getElementById('progress-wrap');

    btn.disabled = true;
    showStatus('');
    progressWrap.style.display = 'block';

    const totalChunks = Math.ceil(modelFile.size / CHUNK_SIZE) || 1;
    let lastData = null;

    try {
        for (let i = 0; i < totalChunks; i++) {
            const chunk = modelFile.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
            const fd    = new FormData();
            fd.append('api_key',      API_KEY);
            fd.append('model_type',   modelType);
            fd.append('chunk_index',  i);
            fd.append('total_chunks', totalChunks);
            fd.append('filename',     modelFile.name);
            fd.append('model_binary', chunk, modelFile.name);

            // model_json is only required on the last chunk
            if (i === totalChunks - 1) {
                fd.append('model_json', jsonFile, jsonFile.name);
            }

            setProgress(
                (i + 1) / totalChunks,
                totalChunks > 1
                    ? `Uploading part ${i + 1} of ${totalChunks}…`
                    : 'Uploading…'
            );

            const resp = await fetch(UPLOAD_URL, { method: 'POST', body: fd });
            const data = await resp.json().catch(() => ({}));

            if (!resp.ok) {
                throw new Error(data.message || `Server error (HTTP ${resp.status})`);
            }
            lastData = data;
        }
        showResult(lastData);
    } catch (err) {
        progressWrap.style.display = 'none';
        showStatus(`<div class="flash error">${escHtml(err.message)}</div>`);
        btn.disabled = false;
    }
});

})();
</script>

<?php endif ?>
</body>
</html>