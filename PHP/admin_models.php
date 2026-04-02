<?php
declare(strict_types=1);
session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';

// ---------------------------------------------------------------------------
// CSRF
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
// DB helpers
// ---------------------------------------------------------------------------

/**
 * Return all users with has_access and has_merge flags for a given model type.
 */
function db_get_model_user_perms(string $model_type): array
{
    $stmt = get_db()->prepare(
        'SELECT u.id, u.name, u.email,
                (SELECT COUNT(*) FROM users_accesspermissions ap
                  WHERE ap.user_id = u.id AND ap.model_type = ?) AS has_access,
                (SELECT COUNT(*) FROM users_mergepermissions  mp
                  WHERE mp.user_id = u.id AND mp.model_type = ?) AS has_merge
         FROM users u
         ORDER BY u.name'
    );
    $stmt->execute([$model_type, $model_type]);
    return $stmt->fetchAll();
}

/**
 * Replace all access and merge permissions for a given model type atomically.
 */
function db_set_model_permissions(string $model_type, array $access_uids, array $merge_uids): void
{
    $db = get_db();
    $db->beginTransaction();
    try {
        $db->prepare('DELETE FROM users_accesspermissions WHERE model_type = ?')->execute([$model_type]);
        $db->prepare('DELETE FROM users_mergepermissions  WHERE model_type = ?')->execute([$model_type]);

        $ia = $db->prepare('INSERT INTO users_accesspermissions (user_id, model_type) VALUES (?, ?)');
        foreach ($access_uids as $uid) {
            $ia->execute([$uid, $model_type]);
        }

        $im = $db->prepare('INSERT INTO users_mergepermissions (user_id, model_type) VALUES (?, ?)');
        foreach ($merge_uids as $uid) {
            $im->execute([$uid, $model_type]);
        }

        $db->commit();
    } catch (Throwable $e) {
        $db->rollBack();
        throw $e;
    }
}

/**
 * Return the set of valid user IDs (for input validation).
 */
function db_get_valid_user_ids(): array
{
    return array_flip(
        get_db()->query('SELECT id FROM users')->fetchAll(PDO::FETCH_COLUMN)
    );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

// ---------------------------------------------------------------------------
// Auth state – shared session with admin.php
// ---------------------------------------------------------------------------

$logged_in_key    = $_SESSION['admin_api_key'] ?? null;
$current_user     = $logged_in_key !== null ? get_user_by_key($logged_in_key) : null;
$is_authenticated = $current_user !== null && (bool) $current_user['is_admin'];

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

        if ($action === 'update_model') {
            $model_type = sanitize_model_type($_POST['model_type'] ?? '');

            if ($model_type === null || !in_array($model_type, get_model_types(), true)) {
                flash('error', 'Invalid or unknown model type.');
            } else {
                $valid_ids = db_get_valid_user_ids();

                $access_uids = array_values(array_filter(
                    array_map('intval', (array) ($_POST['access_users'] ?? [])),
                    fn($id) => isset($valid_ids[$id])
                ));
                $merge_uids = array_values(array_filter(
                    array_map('intval', (array) ($_POST['merge_users'] ?? [])),
                    fn($id) => isset($valid_ids[$id])
                ));

                try {
                    db_set_model_permissions($model_type, $access_uids, $merge_uids);
                    flash('success', "Permissions for \"{$model_type}\" updated.");
                } catch (Throwable $e) {
                    flash('error', 'Failed to update permissions: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF'] . '#model-' . urlencode($model_type ?? ''));
            exit;
        }
    }
}

// ---------------------------------------------------------------------------
// Collect render data
// ---------------------------------------------------------------------------

$model_types = $is_authenticated ? get_model_types() : [];
$flashes     = pop_flashes();

// Pre-load per-model permissions to avoid N+1 queries in the template.
$model_data = [];
foreach ($model_types as $mt) {
    $model_data[$mt] = db_get_model_user_perms($mt);
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Model Permissions</title>
<link rel="stylesheet" href="css/admin.css">
</head>
<body>

<?php if (!$is_authenticated): ?>

<div class="login-wrap">
  <div class="login-card">
    <h2>Dafne Server</h2>
    <p>Sign in with an administrator API key to manage model permissions.</p>
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
      <a href="admin_models.php" class="active">Models</a>
      <a href="admin_upload.php">Upload</a>
      <a href="admin_log.php">Log</a>
      <a href="admin_bulk_import.php">Bulk Import</a>
    </nav>
  </div>
  <form method="post" style="margin:0">
    <input type="hidden" name="action" value="logout">
    <button type="submit" class="btn btn-outline btn-sm" style="color:#cbd5e1;border-color:#3a5a7c">
      Sign Out (<?= h($current_user['name']) ?>)
    </button>
  </form>
</div>

<div class="container container-medium">

  <?php foreach ($flashes as $f): ?>
    <div class="flash <?= h($f['type']) ?>"><?= h($f['msg']) ?></div>
  <?php endforeach ?>

  <?php if (empty($model_types)): ?>
    <div class="card">
      <div class="empty-notice">
        No model types are configured on this server yet.<br>
        Add model directories under <code>models/</code> to get started.
      </div>
    </div>
  <?php else: ?>

    <?php foreach ($model_types as $mt):
        $users   = $model_data[$mt];
        $form_id = 'form-' . preg_replace('/[^a-zA-Z0-9-]/', '', $mt);
    ?>
    <div class="card" id="model-<?= h(urlencode($mt)) ?>">
      <div class="card-header">
        <h2><?= h($mt) ?></h2>

        <?php if (!empty($users)): ?>
        <div class="quick-actions" data-form="<?= h($form_id) ?>">
          <span>Grant to all:</span>
          <button type="button" class="btn btn-xs access" onclick="grantAll('<?= h($form_id) ?>','access')">
            ✓ Access
          </button>
          <button type="button" class="btn btn-xs merge" onclick="grantAll('<?= h($form_id) ?>','merge')">
            ✓ Merge
          </button>
          <button type="button" class="btn btn-xs both" onclick="grantAll('<?= h($form_id) ?>','both')">
            ✓ Both
          </button>
          <button type="button" class="btn btn-xs clear" onclick="grantAll('<?= h($form_id) ?>','none')">
            ✗ Clear all
          </button>
        </div>
        <?php endif ?>
      </div>

      <div class="card-body card-body-flush">
        <?php if (empty($users)): ?>
          <div class="empty-notice">No users exist yet.</div>
        <?php else: ?>

        <form id="<?= h($form_id) ?>" method="post">
          <input type="hidden" name="action"     value="update_model">
          <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
          <input type="hidden" name="model_type" value="<?= h($mt) ?>">

          <table class="perm-table">
            <thead>
              <tr>
                <th>User</th>
                <th class="center">
                  Access<br>
                  <span style="font-weight:400;text-transform:none;letter-spacing:0">read &amp; upload</span>
                </th>
                <th class="center">
                  Merge<br>
                  <span style="font-weight:400;text-transform:none;letter-spacing:0">merge client</span>
                </th>
              </tr>
            </thead>
            <tbody>
              <?php foreach ($users as $u): ?>
              <tr>
                <td>
                  <div class="user-name"><?= h($u['name']) ?></div>
                  <div class="user-email"><?= h($u['email']) ?></div>
                </td>
                <td class="td-check">
                  <input type="checkbox"
                         name="access_users[]"
                         value="<?= (int) $u['id'] ?>"
                         data-type="access"
                         <?= $u['has_access'] ? 'checked' : '' ?>>
                </td>
                <td class="td-check">
                  <input type="checkbox"
                         name="merge_users[]"
                         value="<?= (int) $u['id'] ?>"
                         data-type="merge"
                         <?= $u['has_merge'] ? 'checked' : '' ?>>
                </td>
              </tr>
              <?php endforeach ?>
            </tbody>
          </table>

          <div class="card-footer">
            <span class="user-count"><?= count($users) ?> user<?= count($users) !== 1 ? 's' : '' ?></span>
            <button type="submit" class="btn btn-primary btn-sm">Save Changes</button>
          </div>
        </form>

        <?php endif ?>
      </div>
    </div>
    <?php endforeach ?>

  <?php endif ?>

</div><!-- /container -->

<script>
/**
 * Grant/revoke checkboxes in bulk for a given form.
 * mode: 'access' | 'merge' | 'both' | 'none'
 */
function grantAll(formId, mode) {
    const form   = document.getElementById(formId);
    const access = form.querySelectorAll('input[data-type="access"]');
    const merge  = form.querySelectorAll('input[data-type="merge"]');

    access.forEach(cb => {
        cb.checked = (mode === 'access' || mode === 'both');
    });
    merge.forEach(cb => {
        cb.checked = (mode === 'merge' || mode === 'both');
    });
}

// Highlight the card referenced by the URL hash (e.g. #model-Thigh).
(function () {
    const hash = decodeURIComponent(window.location.hash.slice(1));
    if (!hash) return;
    const el = document.getElementById(hash);
    if (el) {
        el.classList.add('highlighted');
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setTimeout(() => el.classList.remove('highlighted'), 3000);
    }
})();
</script>

<?php endif ?>
</body>
</html>