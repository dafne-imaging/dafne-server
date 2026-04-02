<?php
declare(strict_types=1);
session_start();

/**
 * Dafne Server – web setup wizard.
 *
 * Browse to this page once to initialise the database, apply the schema,
 * create the filesystem layout, and add the first administrator account.
 * Delete or rename this file once setup is complete.
 */

require_once __DIR__ . '/config.php';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function gen_api_key(): string
{
    $chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    $key   = '';
    for ($i = 0; $i < 16; $i++) {
        $key .= $chars[random_int(0, strlen($chars) - 1)];
    }
    return $key;
}

// Paths without realpath() so they work even before the dirs are created.
$models_dir = dirname(__DIR__) . '/models';
$data_dir   = dirname(__DIR__) . '/uploaded_data';

// ---------------------------------------------------------------------------
// State detection
// ---------------------------------------------------------------------------

function probe_state(string $models_dir, string $data_dir): array
{
    $s = [
        'db_ok'     => false,
        'schema_ok' => false,
        'admin_ok'  => false,
        'dirs_ok'   => false,
        'db_err'    => '',
    ];

    try {
        $dsn = sprintf('mysql:host=%s;port=%d;dbname=%s;charset=utf8mb4',
                       MYSQL_HOST, MYSQL_PORT, MYSQL_DB);
        $pdo = new PDO($dsn, MYSQL_USER, MYSQL_PASS,
                       [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
        $s['db_ok'] = true;

        try {
            $s['schema_ok'] = true;
            $s['admin_ok']  = (int) $pdo->query(
                'SELECT COUNT(*) FROM users WHERE is_admin = 1'
            )->fetchColumn() > 0;
        } catch (PDOException) {
            // Schema not yet applied.
        }
    } catch (PDOException $e) {
        $s['db_err'] = $e->getMessage();
    }

    $s['dirs_ok'] = is_dir($models_dir) && is_dir($data_dir);
    return $s;
}

$state = probe_state($models_dir, $data_dir);
$done  = $state['db_ok'] && $state['schema_ok'] && $state['admin_ok'] && $state['dirs_ok'];

// ---------------------------------------------------------------------------
// CSRF
// ---------------------------------------------------------------------------

if (empty($_SESSION['setup_csrf'])) {
    $_SESSION['setup_csrf'] = bin2hex(random_bytes(32));
}
$csrf = $_SESSION['setup_csrf'];

// ---------------------------------------------------------------------------
// POST – run setup
// ---------------------------------------------------------------------------

$log               = [];   // [['level' => 'ok|info|warn|fail', 'msg' => '...']]
$ran               = false;
$api_key_generated = null;
$post_errors       = [];

function slog(string $level, string $msg): void
{
    global $log;
    $log[] = ['level' => $level, 'msg' => $msg];
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {

    if (!hash_equals($csrf, (string) ($_POST['csrf'] ?? ''))) {
        http_response_code(403);
        die('Invalid CSRF token.');
    }

    if ($done) {
        // Already complete – silently ignore stale POST.
        header('Location: ' . $_SERVER['PHP_SELF']);
        exit;
    }

    $ran = true;

    $root_user   = trim((string) ($_POST['root_user']   ?? ''));
    $root_pass   = (string) ($_POST['root_pass']   ?? '');
    $admin_name  = trim((string) ($_POST['admin_name']  ?? ''));
    $admin_email = trim((string) ($_POST['admin_email'] ?? ''));

    // Re-check state in case something changed since page load.
    $state = probe_state($models_dir, $data_dir);

    // ── Step 1 · Database & user ──────────────────────────────────────────

    if ($state['db_ok']) {
        slog('info', "Database '" . MYSQL_DB . "' already accessible as '"
             . MYSQL_USER . "' — skipping creation.");
    } else {
        if ($root_user === '') {
            slog('fail', 'MySQL admin username is required to create the database and user.');
        } else {
            try {
                $dsn_root = sprintf('mysql:host=%s;port=%d;charset=utf8mb4',
                                    MYSQL_HOST, MYSQL_PORT);
                $root_pdo = new PDO($dsn_root, $root_user, $root_pass,
                                    [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
                slog('ok', "Connected to MySQL as '{$root_user}'.");

                // Create database
                $db_q = '`' . str_replace('`', '``', MYSQL_DB) . '`';
                $root_pdo->exec(
                    "CREATE DATABASE IF NOT EXISTS {$db_q}"
                    . " CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                );
                slog('ok', "Database '" . MYSQL_DB . "' is ready.");

                // Create application user
                $app_host  = MYSQL_HOST === 'localhost' ? 'localhost' : '%';
                $chk       = $root_pdo->prepare(
                    "SELECT COUNT(*) FROM mysql.user WHERE User = ? AND Host = ?"
                );
                $chk->execute([MYSQL_USER, $app_host]);
                if ((int) $chk->fetchColumn() === 0) {
                    $root_pdo->exec(sprintf(
                        "CREATE USER %s@%s IDENTIFIED BY %s",
                        $root_pdo->quote(MYSQL_USER),
                        $root_pdo->quote($app_host),
                        $root_pdo->quote(MYSQL_PASS)
                    ));
                    slog('ok', "MySQL user '" . MYSQL_USER . "' created.");
                } else {
                    slog('info', "MySQL user '" . MYSQL_USER . "' already exists.");
                }

                $root_pdo->exec(sprintf(
                    "GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, DROP,"
                    . " INDEX, ALTER, REFERENCES ON %s.* TO %s@%s",
                    $db_q,
                    $root_pdo->quote(MYSQL_USER),
                    $root_pdo->quote($app_host)
                ));
                $root_pdo->exec('FLUSH PRIVILEGES');
                slog('ok', "Privileges granted to '" . MYSQL_USER . "' on '" . MYSQL_DB . "'.");
                unset($root_pdo);

                $state['db_ok'] = true;

            } catch (PDOException $e) {
                slog('fail', "MySQL error: " . $e->getMessage());
            }
        }
    }

    // ── Step 2 · Schema ───────────────────────────────────────────────────

    $pdo = null;
    if ($state['db_ok']) {
        try {
            $dsn = sprintf('mysql:host=%s;port=%d;dbname=%s;charset=utf8mb4',
                           MYSQL_HOST, MYSQL_PORT, MYSQL_DB);
            $pdo = new PDO($dsn, MYSQL_USER, MYSQL_PASS, [
                PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            ]);
        } catch (PDOException $e) {
            slog('fail', "Could not connect as application user: " . $e->getMessage());
        }
    }

    if ($pdo !== null) {
        if ($state['schema_ok']) {
            slog('info', "Schema already applied — skipping.");
        } else {
            $schema_file = __DIR__ . '/schema.sql';
            if (!file_exists($schema_file)) {
                slog('fail', "schema.sql not found at {$schema_file}.");
            } else {
                $raw   = (string) file_get_contents($schema_file);
                $lines = array_filter(
                    explode("\n", $raw),
                    fn($l) => !str_starts_with(ltrim($l), '--') && trim($l) !== ''
                );
                $stmts = array_filter(
                    array_map('trim', explode(';', implode("\n", $lines))),
                    fn($s) => $s !== ''
                );

                $schema_ok = true;
                foreach ($stmts as $sql) {
                    try {
                        $pdo->exec($sql);
                        if (preg_match('/CREATE TABLE\s+IF NOT EXISTS\s+`?(\w+)`?/i', $sql, $m)) {
                            slog('ok', "Table '{$m[1]}' is ready.");
                        }
                    } catch (PDOException $e) {
                        slog('fail', "Schema error: " . $e->getMessage());
                        $schema_ok = false;
                        break;
                    }
                }
                if ($schema_ok) {
                    $state['schema_ok'] = true;
                }
            }
        }
    }

    // ── Step 3 · Filesystem ───────────────────────────────────────────────

    $htaccess_content = "Require all denied\n";
    foreach ([$models_dir, $data_dir] as $dir) {
        if (!is_dir($dir)) {
            if (@mkdir($dir, 0755, true)) {
                slog('ok', "Created directory: {$dir}");
            } else {
                slog('fail', "Could not create directory: {$dir}");
            }
        } else {
            slog('info', "Directory exists: {$dir}");
        }

        $ht = $dir . '/.htaccess';
        if (!file_exists($ht)) {
            if (file_put_contents($ht, $htaccess_content) !== false) {
                slog('ok', "Created .htaccess in {$dir}");
            } else {
                slog('warn', "Could not write .htaccess in {$dir}");
            }
        } else {
            slog('info', ".htaccess already exists in {$dir}");
        }
    }

    // ── Step 4 · First administrator ─────────────────────────────────────

    if ($pdo !== null && $state['schema_ok']) {
        $existing = (int) $pdo->query(
            'SELECT COUNT(*) FROM users WHERE is_admin = 1'
        )->fetchColumn();

        if ($existing > 0) {
            slog('info', "An administrator account already exists — skipping.");
        } elseif ($admin_name === '') {
            slog('warn', "No administrator name provided — skipping admin creation.");
        } elseif (!filter_var($admin_email, FILTER_VALIDATE_EMAIL)) {
            slog('warn', "Invalid or missing email address — skipping admin creation.");
        } else {
            $api_key  = gen_api_key();
            $key_hash = hash('sha256', $api_key);
            $s = $pdo->prepare(
                'INSERT INTO users (name, email, api_key_hash, is_admin) VALUES (?, ?, ?, 1)'
            );
            $s->execute([$admin_name, $admin_email, $key_hash]);
            $uid               = (int) $pdo->lastInsertId();
            $api_key_generated = $api_key;
            slog('ok', "Administrator '{$admin_name}' created (id={$uid}).");
        }
    }

    // Re-probe for the result summary
    $state = probe_state($models_dir, $data_dir);
    $done  = $state['db_ok'] && $state['schema_ok'] && $state['admin_ok'] && $state['dirs_ok'];
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function state_badge(bool $ok, string $yes = 'Done', string $no = 'Pending'): string
{
    $cls  = $ok ? 'badge-done' : 'badge-pending';
    $text = $ok ? $yes : $no;
    return "<span class=\"state-badge {$cls}\">{$text}</span>";
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Setup</title>
<link rel="stylesheet" href="css/admin.css">
<style>
.setup-log        { list-style: none; padding: 0; margin: 0; }
.setup-log li     { padding: 7px 12px; border-radius: 4px; font-size: 13px;
                    margin-bottom: 3px; display: flex; align-items: baseline; gap: 10px; }
.log-ok           { background: #f0fdf4; color: #166534; }
.log-info         { background: #f8fafc; color: #475569; }
.log-warn         { background: #fffbeb; color: #854d0e; }
.log-fail         { background: #fef2f2; color: #991b1b; }
.log-icon         { font-family: monospace; font-weight: 700; flex-shrink: 0; width: 14px; }
.state-badge      { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge-done       { background: #dcfce7; color: #166534; }
.badge-pending    { background: #fef9c3; color: #854d0e; }
.badge-missing    { background: #fee2e2; color: #991b1b; }
.state-grid       { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.state-item       { display: flex; align-items: center; justify-content: space-between;
                    font-size: 13px; padding: 6px 10px; background: #f8fafc;
                    border-radius: 4px; border: 1px solid #e4e8ef; }
.state-item-label { color: #475569; font-weight: 500; }
.cfg-table        { width: 100%; border-collapse: collapse; font-size: 13px; }
.cfg-table td     { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; }
.cfg-table tr:last-child td { border-bottom: none; }
.cfg-table td:first-child   { font-weight: 600; color: #475569; width: 160px; }
.cfg-table code   { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
.key-display-wrap { background: #1e293b; border-radius: 8px; padding: 16px 20px; margin: 12px 0; }
.key-display-wrap p   { color: #94a3b8; font-size: 12px; margin-bottom: 8px; }
.key-code         { font-family: 'SF Mono','Fira Code',monospace; font-size: 20px;
                    color: #fcd34d; letter-spacing: .1em; word-break: break-all; }
.section-head     { font-size: 11px; font-weight: 700; text-transform: uppercase;
                    letter-spacing: .06em; color: #1a3a5c; margin: 18px 0 8px; }
@media (max-width: 500px) { .state-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="topbar">
  <div style="display:flex;align-items:center">
    <h1>Dafne Server</h1>
    <span class="topbar-subtitle" style="opacity:.65;font-size:13px;margin-left:12px">Setup Wizard</span>
  </div>
</div>

<div class="container container-narrow" style="max-width:660px">

<?php if ($done && !$ran): ?>

  <!-- ── Already complete ────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header"><h2>Setup already complete</h2></div>
    <div class="card-body">
      <div class="flash success" style="margin-bottom:12px">
        The Dafne server is fully configured.
      </div>
      <p style="font-size:13px;color:#475569;line-height:1.6">
        For security, <strong>delete or rename <code>setup.php</code></strong> so it cannot
        be accessed again. Then visit
        <a href="admin.php" style="color:#1a3a5c;font-weight:600">admin.php</a>
        to manage users and models.
      </p>
    </div>
  </div>

<?php elseif ($ran): ?>

  <!-- ── Setup result ────────────────────────────────────────────────── -->
  <div class="card">
    <div class="card-header"><h2>Setup <?= $done ? 'complete' : 'finished with issues' ?></h2></div>
    <div class="card-body">

      <?php if ($done): ?>
        <div class="flash success" style="margin-bottom:14px">All steps completed successfully.</div>
      <?php else: ?>
        <div class="flash error" style="margin-bottom:14px">
          One or more steps did not complete. Review the log below and
          <a href="<?= h($_SERVER['PHP_SELF']) ?>" style="color:#991b1b;font-weight:600">run setup again</a>
          after fixing the issue.
        </div>
      <?php endif ?>

      <?php if ($api_key_generated !== null): ?>
      <div class="key-display-wrap">
        <p>Administrator API key — copy it now, it will not be shown again:</p>
        <div class="key-code" id="api-key-val"><?= h($api_key_generated) ?></div>
        <button onclick="copyKey()" class="btn btn-outline btn-sm"
                style="margin-top:10px;color:#94a3b8;border-color:#334155">
          Copy to clipboard
        </button>
      </div>
      <p style="font-size:12px;color:#94a3b8;margin-bottom:14px">
        Store this key securely — it grants full administrator access.
      </p>
      <?php endif ?>

      <p class="section-head">Log</p>
      <ul class="setup-log">
        <?php foreach ($log as $entry): ?>
          <?php
            $icon = match($entry['level']) {
                'ok'   => '✓',
                'warn' => '!',
                'fail' => '✗',
                default => '·',
            };
          ?>
          <li class="log-<?= h($entry['level']) ?>">
            <span class="log-icon"><?= $icon ?></span>
            <span><?= h($entry['msg']) ?></span>
          </li>
        <?php endforeach ?>
      </ul>

      <?php if ($done): ?>
      <p style="margin-top:16px;font-size:13px;color:#475569;line-height:1.6">
        For security, <strong>delete or rename <code>setup.php</code></strong>, then visit
        <a href="admin.php" style="color:#1a3a5c;font-weight:600">admin.php</a>.
      </p>
      <?php endif ?>

    </div>
  </div>

<?php else: ?>

  <!-- ── Setup form ──────────────────────────────────────────────────── -->

  <!-- Current state -->
  <div class="card">
    <div class="card-header"><h2>Current state</h2></div>
    <div class="card-body">
      <div class="state-grid">
        <div class="state-item">
          <span class="state-item-label">Database</span>
          <?= state_badge($state['db_ok']) ?>
        </div>
        <div class="state-item">
          <span class="state-item-label">Schema</span>
          <?= state_badge($state['schema_ok']) ?>
        </div>
        <div class="state-item">
          <span class="state-item-label">Directories</span>
          <?= state_badge($state['dirs_ok']) ?>
        </div>
        <div class="state-item">
          <span class="state-item-label">Administrator</span>
          <?= state_badge($state['admin_ok']) ?>
        </div>
      </div>
      <?php if ($state['db_err'] !== ''): ?>
      <p style="margin-top:10px;font-size:12px;color:#64748b">
        DB error: <code><?= h($state['db_err']) ?></code>
      </p>
      <?php endif ?>
    </div>
  </div>

  <!-- Config summary -->
  <div class="card">
    <div class="card-header"><h2>Configuration (from config.php)</h2></div>
    <div class="card-body" style="padding:0">
      <table class="cfg-table">
        <tr><td>MySQL host</td>      <td><code><?= h((string) MYSQL_HOST) ?>:<?= (int) MYSQL_PORT ?></code></td></tr>
        <tr><td>MySQL database</td>  <td><code><?= h((string) MYSQL_DB) ?></code></td></tr>
        <tr><td>MySQL user</td>      <td><code><?= h((string) MYSQL_USER) ?></code></td></tr>
        <tr><td>Models directory</td><td><code><?= h($models_dir) ?></code></td></tr>
        <tr><td>Data directory</td>  <td><code><?= h($data_dir) ?></code></td></tr>
      </table>
      <p style="font-size:12px;color:#94a3b8;padding:10px 10px 12px">
        Edit <code>config.php</code> to change these values before running setup.
      </p>
    </div>
  </div>

  <!-- Setup form -->
  <div class="card">
    <div class="card-header"><h2>Run setup</h2></div>
    <div class="card-body">
      <form method="post">
        <input type="hidden" name="csrf" value="<?= h($csrf) ?>">

        <?php if (!$state['db_ok']): ?>
        <!-- DB creation needs privileged credentials -->
        <p class="section-head">MySQL admin credentials</p>
        <p style="font-size:12px;color:#64748b;margin-bottom:10px">
          The application database and user do not exist yet. Provide a
          privileged MySQL account (e.g. <code>root</code>) to create them.
        </p>
        <div class="form-row">
          <label for="root_user">Admin username</label>
          <input type="text" id="root_user" name="root_user"
                 value="<?= h($_POST['root_user'] ?? 'root') ?>"
                 autocomplete="off" placeholder="root">
        </div>
        <div class="form-row">
          <label for="root_pass">Admin password</label>
          <input type="password" id="root_pass" name="root_pass" autocomplete="off">
        </div>
        <?php endif ?>

        <?php if (!$state['admin_ok']): ?>
        <!-- First admin creation -->
        <p class="section-head">First administrator account</p>
        <p style="font-size:12px;color:#64748b;margin-bottom:10px">
          Leave blank to skip — you can create an admin later directly in MySQL.
        </p>
        <div class="form-row">
          <label for="admin_name">Full name</label>
          <input type="text" id="admin_name" name="admin_name"
                 value="<?= h($_POST['admin_name'] ?? '') ?>"
                 autocomplete="name" placeholder="Alice Smith">
        </div>
        <div class="form-row">
          <label for="admin_email">Email address</label>
          <input type="email" id="admin_email" name="admin_email"
                 value="<?= h($_POST['admin_email'] ?? '') ?>"
                 autocomplete="email" placeholder="alice@example.com">
        </div>
        <?php else: ?>
        <div class="flash success" style="margin-bottom:14px">
          An administrator account already exists — no action needed for this step.
        </div>
        <?php endif ?>

        <button type="submit" class="btn btn-primary btn-lg" style="margin-top:6px">
          Run Setup
        </button>
      </form>
    </div>
  </div>

<?php endif ?>

</div><!-- /container -->

<?php if ($api_key_generated !== null): ?>
<script>
function copyKey() {
    const txt = document.getElementById('api-key-val').textContent.trim();
    navigator.clipboard.writeText(txt).then(() => {
        const btn = event.target;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy to clipboard'; }, 2000);
    });
}
</script>
<?php endif ?>

</body>
</html>