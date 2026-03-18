<?php
declare(strict_types=1);
session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/auth.php';

// ---------------------------------------------------------------------------
// CSRF
// ---------------------------------------------------------------------------

if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['csrf_token'];

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

$logged_in_key    = $_SESSION['admin_api_key'] ?? null;
$current_user     = $logged_in_key !== null ? get_user_by_key($logged_in_key) : null;
$is_authenticated = $current_user !== null && (bool) $current_user['is_admin'];

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

// ---------------------------------------------------------------------------
// Login / logout
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
    }
}

// ---------------------------------------------------------------------------
// CSV export (before any HTML output)
// ---------------------------------------------------------------------------

if ($is_authenticated && isset($_GET['export']) && $_GET['export'] === 'csv') {
    $search     = trim($_GET['search']   ?? '');
    $date_from  = trim($_GET['date_from'] ?? '');
    $date_to    = trim($_GET['date_to']   ?? '');

    $where  = [];
    $params = [];

    if ($search !== '') {
        $where[]  = 'message LIKE ?';
        $params[] = '%' . $search . '%';
    }
    if ($date_from !== '') {
        $where[]  = 'created_at >= ?';
        $params[] = $date_from . ' 00:00:00';
    }
    if ($date_to !== '') {
        $where[]  = 'created_at <= ?';
        $params[] = $date_to . ' 23:59:59';
    }

    $sql = 'SELECT id, created_at, message FROM server_log'
         . ($where ? ' WHERE ' . implode(' AND ', $where) : '')
         . ' ORDER BY created_at DESC';

    $stmt = get_db()->prepare($sql);
    $stmt->execute($params);
    $rows = $stmt->fetchAll();

    $filename = 'dafne_log_' . date('Ymd_His') . '.csv';
    header('Content-Type: text/csv; charset=UTF-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, no-store');

    // Prefix cells that start with formula-triggering characters so that
    // spreadsheet applications (Excel, LibreOffice) do not execute them.
    $csv_safe = static function (string $value): string {
        if ($value !== '' && strpos('=+-@', $value[0]) !== false) {
            return "\t" . $value;
        }
        return $value;
    };

    $out = fopen('php://output', 'w');
    fprintf($out, chr(0xEF) . chr(0xBB) . chr(0xBF));   // UTF-8 BOM for Excel
    fputcsv($out, ['ID', 'Timestamp', 'Message']);
    foreach ($rows as $row) {
        fputcsv($out, [
            $row['id'],
            $row['created_at'],
            $csv_safe($row['message']),
        ]);
    }
    fclose($out);
    exit;
}

// ---------------------------------------------------------------------------
// Log query (paginated)
// ---------------------------------------------------------------------------

const PAGE_SIZE = 100;

$search     = trim($_GET['search']    ?? '');
$date_from  = trim($_GET['date_from'] ?? '');
$date_to    = trim($_GET['date_to']   ?? '');
$page       = max(1, (int) ($_GET['page'] ?? 1));

$log_rows   = [];
$total_rows = 0;
$total_pages = 1;

if ($is_authenticated) {
    $where  = [];
    $params = [];

    if ($search !== '') {
        $where[]  = 'message LIKE ?';
        $params[] = '%' . $search . '%';
    }
    if ($date_from !== '') {
        $where[]  = 'created_at >= ?';
        $params[] = $date_from . ' 00:00:00';
    }
    if ($date_to !== '') {
        $where[]  = 'created_at <= ?';
        $params[] = $date_to . ' 23:59:59';
    }

    $where_sql = $where ? ' WHERE ' . implode(' AND ', $where) : '';

    $count_stmt = get_db()->prepare('SELECT COUNT(*) FROM server_log' . $where_sql);
    $count_stmt->execute($params);
    $total_rows  = (int) $count_stmt->fetchColumn();
    $total_pages = max(1, (int) ceil($total_rows / PAGE_SIZE));
    $page        = min($page, $total_pages);
    $offset      = ($page - 1) * PAGE_SIZE;

    $data_stmt = get_db()->prepare(
        'SELECT id, created_at, message FROM server_log'
        . $where_sql
        . ' ORDER BY created_at DESC LIMIT ' . PAGE_SIZE . ' OFFSET ' . $offset
    );
    $data_stmt->execute($params);
    $log_rows = $data_stmt->fetchAll();
}

// Build the current filter query string (without page) for links
function filter_qs(array $extra = []): string
{
    global $search, $date_from, $date_to;
    $p = array_filter([
        'search'    => $search,
        'date_from' => $date_from,
        'date_to'   => $date_to,
    ], fn($v) => $v !== '');
    return http_build_query(array_merge($p, $extra));
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Server Log</title>
<link rel="stylesheet" href="css/admin.css">
<style>
.log-table          { width:100%; border-collapse:collapse; font-size:13px; }
.log-table th,
.log-table td       { padding:7px 10px; border-bottom:1px solid #e2e8f0; vertical-align:top; }
.log-table th       { background:#f8fafc; font-weight:600; color:#475569;
                       text-transform:uppercase; font-size:11px; letter-spacing:.04em; }
.log-table tr:hover td { background:#f1f5f9; }
.log-ts             { white-space:nowrap; color:#64748b; font-size:12px; }
.log-id             { color:#94a3b8; font-size:11px; text-align:right; }
.filter-bar         { display:flex; gap:8px; flex-wrap:wrap; align-items:flex-end; margin-bottom:16px; }
.filter-bar .form-row { margin:0; flex:1; min-width:140px; }
.filter-bar label   { font-size:12px; color:#64748b; display:block; margin-bottom:3px; }
.filter-bar input   { width:100%; }
.pagination         { display:flex; gap:4px; align-items:center; justify-content:flex-end;
                       margin-top:12px; flex-wrap:wrap; }
.pagination a,
.pagination span    { padding:4px 10px; border-radius:4px; font-size:13px; text-decoration:none; }
.pagination a       { background:#e2e8f0; color:#1a1a2e; }
.pagination a:hover { background:#cbd5e1; }
.pagination .current{ background:#1a3a5c; color:#fff; font-weight:600; }
.pagination .dots   { color:#94a3b8; }
.row-count          { font-size:12px; color:#64748b; margin-top:8px; }
</style>
</head>
<body>

<?php if (!$is_authenticated): ?>

<div class="login-wrap">
  <div class="login-card">
    <h2>Dafne Server</h2>
    <p>Sign in with an administrator API key to view the server log.</p>
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
      <a href="admin_log.php" class="active">Log</a>
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

  <div class="card">
    <div class="card-header" style="justify-content:space-between;align-items:center">
      <h2>Server Log</h2>
      <a href="?<?= h(filter_qs(['export' => 'csv'])) ?>" class="btn btn-outline btn-sm">
        Export CSV
      </a>
    </div>
    <div class="card-body">

      <!-- Filter bar -->
      <form method="get" action="admin_log.php">
        <div class="filter-bar">
          <div class="form-row">
            <label for="f-search">Search</label>
            <input type="text" id="f-search" name="search"
                   value="<?= h($search) ?>" placeholder="keyword…">
          </div>
          <div class="form-row">
            <label for="f-from">From</label>
            <input type="date" id="f-from" name="date_from" value="<?= h($date_from) ?>">
          </div>
          <div class="form-row">
            <label for="f-to">To</label>
            <input type="date" id="f-to" name="date_to" value="<?= h($date_to) ?>">
          </div>
          <div style="display:flex;gap:6px;padding-bottom:1px">
            <button type="submit" class="btn btn-primary btn-sm">Filter</button>
            <?php if ($search !== '' || $date_from !== '' || $date_to !== ''): ?>
              <a href="admin_log.php" class="btn btn-outline btn-sm">Clear</a>
            <?php endif ?>
          </div>
        </div>
      </form>

      <?php if (empty($log_rows)): ?>
        <p style="color:#64748b">No log entries found.</p>
      <?php else: ?>
        <table class="log-table">
          <thead>
            <tr>
              <th class="log-id">#</th>
              <th style="white-space:nowrap">Timestamp</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            <?php foreach ($log_rows as $row): ?>
            <tr>
              <td class="log-id"><?= (int) $row['id'] ?></td>
              <td class="log-ts"><?= h($row['created_at']) ?></td>
              <td><?= h($row['message']) ?></td>
            </tr>
            <?php endforeach ?>
          </tbody>
        </table>

        <p class="row-count">
          Showing <?= count($log_rows) ?> of <?= number_format($total_rows) ?> entries
          (page <?= $page ?> / <?= $total_pages ?>)
        </p>

        <?php if ($total_pages > 1): ?>
        <div class="pagination">
          <?php if ($page > 1): ?>
            <a href="?<?= h(filter_qs(['page' => $page - 1])) ?>">&laquo; Prev</a>
          <?php endif ?>

          <?php
          // Show at most 7 page links: first, last, current ±2, with ellipsis
          $links = array_unique(array_filter([
              1, 2,
              $page - 2, $page - 1, $page, $page + 1, $page + 2,
              $total_pages - 1, $total_pages,
          ], fn($p) => $p >= 1 && $p <= $total_pages));
          sort($links);
          $prev_link = null;
          foreach ($links as $lp) {
              if ($prev_link !== null && $lp - $prev_link > 1) {
                  echo '<span class="dots">…</span>';
              }
              if ($lp === $page) {
                  echo '<span class="current">' . $lp . '</span>';
              } else {
                  echo '<a href="?' . h(filter_qs(['page' => $lp])) . '">' . $lp . '</a>';
              }
              $prev_link = $lp;
          }
          ?>

          <?php if ($page < $total_pages): ?>
            <a href="?<?= h(filter_qs(['page' => $page + 1])) ?>">Next &raquo;</a>
          <?php endif ?>
        </div>
        <?php endif ?>
      <?php endif ?>

    </div>
  </div>

</div><!-- /container -->

<?php endif ?>
</body>
</html>