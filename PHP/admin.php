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
// DB helpers (admin-only operations)
// ---------------------------------------------------------------------------

function db_get_all_users(): array
{
    return get_db()
        ->query('SELECT id, name, email, is_admin, created_at FROM users ORDER BY name')
        ->fetchAll();
}

function db_get_user_access_models(int $user_id): array
{
    $s = get_db()->prepare('SELECT model_type FROM users_accesspermissions WHERE user_id = ?');
    $s->execute([$user_id]);
    return $s->fetchAll(PDO::FETCH_COLUMN);
}

function db_get_user_merge_models(int $user_id): array
{
    $s = get_db()->prepare('SELECT model_type FROM users_mergepermissions WHERE user_id = ?');
    $s->execute([$user_id]);
    return $s->fetchAll(PDO::FETCH_COLUMN);
}

function db_set_user_permissions(
    int $user_id,
    bool $is_admin,
    array $access_models,
    array $merge_models
): void {
    $db = get_db();
    $db->beginTransaction();
    try {
        $db->prepare('UPDATE users SET is_admin = ? WHERE id = ?')
           ->execute([(int) $is_admin, $user_id]);

        $db->prepare('DELETE FROM users_accesspermissions WHERE user_id = ?')->execute([$user_id]);
        $db->prepare('DELETE FROM users_mergepermissions  WHERE user_id = ?')->execute([$user_id]);

        $ins_a = $db->prepare('INSERT INTO users_accesspermissions (user_id, model_type) VALUES (?, ?)');
        foreach ($access_models as $m) {
            $ins_a->execute([$user_id, $m]);
        }

        $ins_m = $db->prepare('INSERT INTO users_mergepermissions (user_id, model_type) VALUES (?, ?)');
        foreach ($merge_models as $m) {
            $ins_m->execute([$user_id, $m]);
        }

        $db->commit();
    } catch (Throwable $e) {
        $db->rollBack();
        throw $e;
    }
}

function db_create_user(string $name, string $email, string $api_key_hash): int
{
    $s = get_db()->prepare('INSERT INTO users (name, email, api_key_hash) VALUES (?, ?, ?)');
    $s->execute([$name, $email, $api_key_hash]);
    return (int) get_db()->lastInsertId();
}

function db_delete_user(int $user_id): void
{
    get_db()->prepare('DELETE FROM users WHERE id = ?')->execute([$user_id]);
}

function db_get_pending_requests(): array
{
    return get_db()
        ->query("SELECT * FROM user_requests WHERE status = 'pending' ORDER BY created_at ASC")
        ->fetchAll();
}

function db_get_request(int $id): ?array
{
    $s = get_db()->prepare('SELECT * FROM user_requests WHERE id = ?');
    $s->execute([$id]);
    $r = $s->fetch();
    return $r !== false ? $r : null;
}

function db_mark_request_approved(int $id, int $reviewer_id): void
{
    get_db()->prepare(
        "UPDATE user_requests SET status = 'approved', reviewed_at = NOW(), reviewed_by = ? WHERE id = ?"
    )->execute([$reviewer_id, $id]);
}

function db_mark_request_rejected(int $id, int $reviewer_id): void
{
    get_db()->prepare(
        "UPDATE user_requests SET status = 'rejected', reviewed_at = NOW(), reviewed_by = ? WHERE id = ?"
    )->execute([$reviewer_id, $id]);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function generate_api_key(): string
{
    $chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    $len   = strlen($chars);
    $key   = '';
    for ($i = 0; $i < 16; $i++) {
        $key .= $chars[random_int(0, $len - 1)];
    }
    return $key;
}

function send_welcome_email(string $to_email, string $to_name, string $api_key): bool
{
    $subject = 'Your Dafne Server API Key';
    $body    = implode("\r\n", [
        "Hello {$to_name},",
        '',
        'An account has been created for you on the Dafne federated learning server.',
        '',
        'Your API key is:',
        '',
        "    {$api_key}",
        '',
        'Keep this key confidential — it grants access to your assigned models.',
        '',
        'This message was sent automatically by the Dafne server.',
    ]);
    $headers = implode("\r\n", [
        'From: ' . ADMIN_EMAIL,
        'Reply-To: ' . ADMIN_EMAIL,
        'Content-Type: text/plain; charset=UTF-8',
        'MIME-Version: 1.0',
    ]);
    return mail($to_email, $subject, $body, $headers);
}

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

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? '';

    // Login does not require CSRF (session not yet established with known token).
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

        if ($action === 'update_user') {
            $user_id      = (int) ($_POST['user_id'] ?? 0);
            $new_is_admin = isset($_POST['is_admin']);

            // An admin cannot strip their own admin flag to prevent lockout.
            if ($user_id === (int) $current_user['id']) {
                $new_is_admin = true;
            }

            $access_models = sanitize_posted_models((array) ($_POST['access_models'] ?? []));
            $merge_models  = sanitize_posted_models((array) ($_POST['merge_models']  ?? []));

            try {
                db_set_user_permissions($user_id, $new_is_admin, $access_models, $merge_models);
                flash('success', 'Permissions updated successfully.');
            } catch (Throwable $e) {
                flash('error', 'Failed to update permissions: ' . $e->getMessage());
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;

        } elseif ($action === 'create_user') {
            $name       = trim($_POST['name']  ?? '');
            $email      = trim($_POST['email'] ?? '');
            $send_email = isset($_POST['send_email']);

            if ($name === '' || $email === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
                flash('error', 'Please provide a valid name and email address.');
            } else {
                $api_key = generate_api_key();
                try {
                    db_create_user($name, $email, hash_api_key($api_key));
                    $_SESSION['new_user_key'] = [
                        'key'   => $api_key,
                        'name'  => $name,
                        'email' => $email,
                    ];
                    if ($send_email) {
                        $sent = send_welcome_email($email, $name, $api_key);
                        flash('success', "User \"{$name}\" created." .
                            ($sent ? ' Welcome email sent.' : ' Warning: email delivery failed.'));
                    } else {
                        flash('success', "User \"{$name}\" created.");
                    }
                } catch (Throwable $e) {
                    flash('error', 'Failed to create user: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;

        } elseif ($action === 'delete_user') {
            $user_id = (int) ($_POST['user_id'] ?? 0);
            if ($user_id === (int) $current_user['id']) {
                flash('error', 'You cannot delete your own account.');
            } else {
                try {
                    db_delete_user($user_id);
                    flash('success', 'User deleted.');
                } catch (Throwable $e) {
                    flash('error', 'Failed to delete user: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;

        } elseif ($action === 'approve_request') {
            $request_id    = (int) ($_POST['request_id'] ?? 0);
            $new_is_admin  = isset($_POST['is_admin']);
            $send_email    = isset($_POST['send_email']);
            $access_models = sanitize_posted_models((array) ($_POST['access_models'] ?? []));
            $merge_models  = sanitize_posted_models((array) ($_POST['merge_models']  ?? []));

            $req = db_get_request($request_id);
            if ($req === null || $req['status'] !== 'pending') {
                flash('error', 'Request not found or already processed.');
            } else {
                $api_key = generate_api_key();
                try {
                    $uid = db_create_user($req['name'], $req['email'], hash_api_key($api_key));
                    db_set_user_permissions($uid, $new_is_admin, $access_models, $merge_models);
                    db_mark_request_approved($request_id, (int) $current_user['id']);
                    $_SESSION['new_user_key'] = [
                        'key'   => $api_key,
                        'name'  => $req['name'],
                        'email' => $req['email'],
                    ];
                    if ($send_email) {
                        $sent = send_welcome_email($req['email'], $req['name'], $api_key);
                        flash('success', "Request approved — user \"{$req['name']}\" created." .
                            ($sent ? ' Welcome email sent.' : ' Warning: email delivery failed.'));
                    } else {
                        flash('success', "Request approved — user \"{$req['name']}\" created.");
                    }
                } catch (Throwable $e) {
                    flash('error', 'Failed to approve request: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;

        } elseif ($action === 'reject_request') {
            $request_id = (int) ($_POST['request_id'] ?? 0);
            $req = db_get_request($request_id);
            if ($req === null || $req['status'] !== 'pending') {
                flash('error', 'Request not found or already processed.');
            } else {
                try {
                    db_mark_request_rejected($request_id, (int) $current_user['id']);
                    flash('success', "Access request from \"{$req['name']}\" rejected.");
                } catch (Throwable $e) {
                    flash('error', 'Failed to reject request: ' . $e->getMessage());
                }
            }
            header('Location: ' . $_SERVER['PHP_SELF']);
            exit;
        }
    }
}

// ---------------------------------------------------------------------------
// Collect render data
// ---------------------------------------------------------------------------

$model_types  = $is_authenticated ? get_model_types() : [];
$users        = [];
$user_perms   = [];

if ($is_authenticated) {
    $users = db_get_all_users();
    foreach ($users as $u) {
        $uid = (int) $u['id'];
        $user_perms[$uid] = [
            'access' => array_flip(db_get_user_access_models($uid)),
            'merge'  => array_flip(db_get_user_merge_models($uid)),
        ];
    }
}

$flashes          = pop_flashes();
$new_user_key     = $_SESSION['new_user_key'] ?? null;
unset($_SESSION['new_user_key']);
$pending_requests = $is_authenticated ? db_get_pending_requests() : [];

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – User Management</title>
<link rel="stylesheet" href="css/admin.css">
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
      <a href="admin.php" class="active">Users</a>
      <a href="admin_models.php">Models</a>
      <a href="admin_upload.php">Upload</a>
      <a href="admin_log.php">Log</a>
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

  <?php if ($new_user_key): ?>
    <div class="key-banner">
      <p><strong>New user created: <?= h($new_user_key['name']) ?> &lt;<?= h($new_user_key['email']) ?>&gt;</strong></p>
      <p>Copy this API key now — it will not be shown again.</p>
      <div class="key-display">
        <code id="new-key"><?= h($new_user_key['key']) ?></code>
        <button type="button" class="btn btn-outline btn-sm"
                onclick="navigator.clipboard.writeText(document.getElementById('new-key').textContent)">
          Copy
        </button>
      </div>
    </div>
  <?php endif ?>

  <!-- ================================================================ -->
  <!-- Pending access requests                                            -->
  <!-- ================================================================ -->
  <?php if (!empty($pending_requests)): ?>
  <div class="card">
    <div class="card-header">
      <h2>
        Pending Access Requests
        <span class="badge-count"><?= count($pending_requests) ?></span>
      </h2>
    </div>

    <?php foreach ($pending_requests as $req):
        $rid       = (int) $req['id'];
        $requested = json_decode($req['requested_models'], true) ?? [];
    ?>
    <div class="request-item" id="req-<?= $rid ?>">
      <div class="request-head">
        <div>
          <div class="user-meta" style="margin-bottom:4px">
            <span class="user-name"><?= h($req['name']) ?></span>
            <span class="user-email"><?= h($req['email']) ?></span>
          </div>
          <span style="font-size:12px;color:#94a3b8">
            Submitted <?= h(substr($req['created_at'], 0, 16)) ?>
          </span>
        </div>
        <button type="button" class="btn btn-outline btn-sm"
                onclick="toggleReview(<?= $rid ?>)" id="btn-<?= $rid ?>">
          Review ▾
        </button>
      </div>

      <?php if ($req['reason'] !== ''): ?>
      <div class="request-reason">
        <strong>Reason:</strong> <?= h($req['reason']) ?>
      </div>
      <?php endif ?>

      <div class="request-models">
        <strong>Requested models:</strong>
        <?php if (empty($requested)): ?>
          <em style="color:#94a3b8">none specified</em>
        <?php else: ?>
          <?php foreach ($requested as $m): ?>
            <span class="model-tag"><?= h($m) ?></span>
          <?php endforeach ?>
        <?php endif ?>
      </div>

      <!-- Inline approval panel -->
      <div class="review-panel" id="review-<?= $rid ?>">
        <h3>Grant permissions</h3>

        <?php if (!empty($model_types)): ?>
        <table class="perm-table" style="margin-bottom:12px">
          <thead>
            <tr>
              <th>Model Type</th>
              <th style="text-align:center">Access<br><span style="font-weight:400;text-transform:none;letter-spacing:0">read &amp; upload</span></th>
              <th style="text-align:center">Merge<br><span style="font-weight:400;text-transform:none;letter-spacing:0">merge client</span></th>
            </tr>
          </thead>
          <tbody>
            <?php foreach ($model_types as $mt): ?>
            <tr>
              <td><?= h($mt) ?></td>
              <td style="text-align:center">
                <input type="checkbox" form="approve-<?= $rid ?>"
                       name="access_models[]" value="<?= h($mt) ?>"
                       <?= in_array($mt, $requested, true) ? 'checked' : '' ?>>
              </td>
              <td style="text-align:center">
                <input type="checkbox" form="approve-<?= $rid ?>"
                       name="merge_models[]" value="<?= h($mt) ?>">
              </td>
            </tr>
            <?php endforeach ?>
          </tbody>
        </table>
        <?php else: ?>
          <p class="no-models" style="margin-bottom:12px">No model types configured yet — permissions can be set after approval.</p>
        <?php endif ?>

        <div class="admin-row" style="margin-top:0;margin-bottom:10px">
          <input type="checkbox" id="req-admin-<?= $rid ?>" form="approve-<?= $rid ?>" name="is_admin" value="1">
          <label for="req-admin-<?= $rid ?>">Administrator</label>
        </div>
        <div class="checkbox-row" style="margin-top:0;margin-bottom:16px">
          <input type="checkbox" id="req-mail-<?= $rid ?>" form="approve-<?= $rid ?>" name="send_email" value="1" checked>
          <label for="req-mail-<?= $rid ?>">Send welcome email with API key</label>
        </div>

        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button type="submit" form="approve-<?= $rid ?>" class="btn btn-success btn-sm">
            Approve &amp; Create User
          </button>
          <button type="button" class="btn btn-danger btn-sm"
                  onclick="confirmReject(<?= $rid ?>, '<?= h(addslashes($req['name'])) ?>')">
            Reject Request
          </button>
        </div>

        <!-- The actual form elements (can be outside the visual block due to form= attribute) -->
        <form id="approve-<?= $rid ?>" method="post" style="display:none">
          <input type="hidden" name="action"     value="approve_request">
          <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
          <input type="hidden" name="request_id" value="<?= $rid ?>">
        </form>
        <form id="reject-<?= $rid ?>" method="post" style="display:none">
          <input type="hidden" name="action"     value="reject_request">
          <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
          <input type="hidden" name="request_id" value="<?= $rid ?>">
        </form>
      </div>
    </div>
    <?php endforeach ?>
  </div>
  <?php endif ?>

  <!-- ================================================================ -->
  <!-- Create new user                                                    -->
  <!-- ================================================================ -->
  <div class="card">
    <div class="card-header"><h2>Create New User</h2></div>
    <div class="card-body">
      <form method="post">
        <input type="hidden" name="action"     value="create_user">
        <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
        <div class="create-grid">
          <div class="form-row" style="margin:0">
            <label for="new_name">Name</label>
            <input type="text" id="new_name" name="name" autocomplete="off" required>
          </div>
          <div class="form-row" style="margin:0">
            <label for="new_email">Email</label>
            <input type="email" id="new_email" name="email" autocomplete="off" required>
          </div>
        </div>
        <div class="checkbox-row" style="margin-top:14px">
          <input type="checkbox" id="send_email" name="send_email" value="1" checked>
          <label for="send_email">Send welcome email with API key to the new user</label>
        </div>
        <div style="margin-top:16px">
          <button type="submit" class="btn btn-success">Create User &amp; Generate Key</button>
        </div>
      </form>
    </div>
  </div>

  <!-- ================================================================ -->
  <!-- User list                                                          -->
  <!-- ================================================================ -->
  <?php if (empty($users)): ?>
    <div class="card"><div class="card-body" style="color:#64748b">No users found.</div></div>
  <?php else: ?>
    <?php foreach ($users as $u):
        $uid      = (int) $u['id'];
        $is_self  = $uid === (int) $current_user['id'];
        $perms    = $user_perms[$uid];
    ?>
    <div class="card">
      <div class="card-header">
        <div class="user-meta">
          <span class="user-name"><?= h($u['name']) ?></span>
          <span class="user-email"><?= h($u['email']) ?></span>
          <?php if ($is_self): ?><span class="badge you">You</span><?php endif ?>
          <?php if ((bool) $u['is_admin']): ?><span class="badge admin">Admin</span><?php endif ?>
        </div>
        <span style="font-size:12px;color:#94a3b8">
          Since <?= h(substr($u['created_at'], 0, 10)) ?>
        </span>
      </div>
      <div class="card-body">
        <form method="post">
          <input type="hidden" name="action"     value="update_user">
          <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
          <input type="hidden" name="user_id"    value="<?= $uid ?>">

          <div class="admin-row">
            <input type="checkbox" id="admin_<?= $uid ?>" name="is_admin" value="1"
              <?= (bool) $u['is_admin'] ? 'checked' : '' ?>
              <?= $is_self ? 'disabled title="You cannot remove your own admin status"' : '' ?>>
            <?php if ($is_self): ?>
              <!-- Keep the value even though the checkbox is disabled -->
              <input type="hidden" name="is_admin" value="1">
            <?php endif ?>
            <label for="admin_<?= $uid ?>">Administrator
              <span style="font-weight:400;color:#94a3b8">(can manage users)</span>
            </label>
          </div>

          <?php if (empty($model_types)): ?>
            <p class="no-models">No model types are configured on this server yet.</p>
          <?php else: ?>
            <table class="perm-table">
              <thead>
                <tr>
                  <th>Model Type</th>
                  <th style="text-align:center">Access<br><span style="font-weight:400;text-transform:none;letter-spacing:0">read &amp; upload</span></th>
                  <th style="text-align:center">Merge<br><span style="font-weight:400;text-transform:none;letter-spacing:0">merge client</span></th>
                </tr>
              </thead>
              <tbody>
                <?php foreach ($model_types as $mt): ?>
                <tr>
                  <td><?= h($mt) ?></td>
                  <td style="text-align:center">
                    <input type="checkbox" name="access_models[]" value="<?= h($mt) ?>"
                      <?= isset($perms['access'][$mt]) ? 'checked' : '' ?>>
                  </td>
                  <td style="text-align:center">
                    <input type="checkbox" name="merge_models[]" value="<?= h($mt) ?>"
                      <?= isset($perms['merge'][$mt]) ? 'checked' : '' ?>>
                  </td>
                </tr>
                <?php endforeach ?>
              </tbody>
            </table>
          <?php endif ?>

          <div class="user-actions">
            <button type="submit" class="btn btn-primary btn-sm">Save Changes</button>
            <?php if (!$is_self): ?>
              <button type="button" class="btn btn-danger btn-sm"
                      onclick="confirmDelete(<?= $uid ?>, '<?= h(addslashes($u['name'])) ?>')">
                Delete User
              </button>
            <?php else: ?>
              <span></span>
            <?php endif ?>
          </div>
        </form>
      </div>
    </div>
    <?php endforeach ?>
  <?php endif ?>

</div><!-- /container -->

<!-- Hidden delete confirmation form -->
<form id="delete-form" method="post" style="display:none">
  <input type="hidden" name="action"     value="delete_user">
  <input type="hidden" name="csrf_token" value="<?= h($csrf_token) ?>">
  <input type="hidden" name="user_id"    id="delete-user-id">
</form>

<script>
function confirmDelete(userId, userName) {
    if (!confirm('Delete user "' + userName + '"?\n\nThis cannot be undone.')) return;
    document.getElementById('delete-user-id').value = userId;
    document.getElementById('delete-form').submit();
}

function toggleReview(rid) {
    const panel = document.getElementById('review-' + rid);
    const btn   = document.getElementById('btn-' + rid);
    const open  = panel.style.display !== 'none' && panel.style.display !== '';
    panel.style.display = open ? 'none' : 'block';
    btn.textContent = open ? 'Review ▾' : 'Close ▴';
}

function confirmReject(rid, name) {
    if (!confirm('Reject the access request from "' + name + '"?')) return;
    document.getElementById('reject-' + rid).submit();
}

// If the URL contains ?review=ID, scroll to and open that request.
(function () {
    const id = new URLSearchParams(window.location.search).get('review');
    if (!id) return;
    const el = document.getElementById('req-' + id);
    if (!el) return;
    el.classList.add('highlight');
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const panel = document.getElementById('review-' + id);
    const btn   = document.getElementById('btn-' + id);
    if (panel) { panel.style.display = 'block'; }
    if (btn)   { btn.textContent = 'Close ▴'; }
})();
</script>

<?php endif ?>
</body>
</html>
