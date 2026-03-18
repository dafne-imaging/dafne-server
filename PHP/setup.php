#!/usr/bin/env php
<?php
/**
 * Dafne Server – one-time setup script.
 *
 * Run from the PHP/ directory:
 *   php setup.php
 *
 * All settings are read from config.php — edit that file before running.
 *
 * What this script does:
 *   1. Reads all configuration values from config.php.
 *   2. Connects to MySQL with a privileged account and creates the database
 *      and application user if they do not already exist.
 *   3. Applies schema.sql (all statements are idempotent – CREATE IF NOT EXISTS).
 *   4. Creates the db/ directory tree.
 *   5. Optionally creates the first administrator account.
 */

declare(strict_types=1);

if (PHP_SAPI !== 'cli') {
    http_response_code(403);
    exit('This script must be run from the command line.');
}

require_once __DIR__ . '/config.php';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ESC = "\033[";

function c(string $text, string $colour): string
{
    $codes = ['reset' => '0', 'bold' => '1', 'red' => '31', 'green' => '32',
              'yellow' => '33', 'cyan' => '36', 'white' => '37', 'dim' => '2'];
    return ESC . ($codes[$colour] ?? '0') . 'm' . $text . ESC . "0m";
}

function info(string $msg): void  { echo c('  · ', 'dim')  . $msg . "\n"; }
function ok(string $msg): void    { echo c('  ✓ ', 'green') . $msg . "\n"; }
function warn(string $msg): void  { echo c('  ! ', 'yellow') . $msg . "\n"; }
function fail(string $msg): void  { echo c('  ✗ ', 'red')   . $msg . "\n"; }

function abort(string $msg): never
{
    echo "\n" . c('Error: ', 'red') . $msg . "\n\n";
    exit(1);
}

function prompt(string $label, string $default = '', bool $secret = false): string
{
    $hint = $default !== '' ? " [{$default}]" : '';
    echo c("  → {$label}{$hint}: ", 'cyan');
    if ($secret) {
        // Suppress echo on supported platforms
        if (PHP_OS_FAMILY !== 'Windows') {
            system('stty -echo');
        }
        $val = trim((string) fgets(STDIN));
        if (PHP_OS_FAMILY !== 'Windows') {
            system('stty echo');
        }
        echo "\n";
    } else {
        $val = trim((string) fgets(STDIN));
    }
    return $val !== '' ? $val : $default;
}

function ask_yes(string $question, bool $default = true): bool
{
    $hint = $default ? '[Y/n]' : '[y/N]';
    echo c("  → {$question} {$hint}: ", 'cyan');
    $val = strtolower(trim((string) fgets(STDIN)));
    if ($val === '') return $default;
    return $val === 'y' || $val === 'yes';
}

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

function hash_api_key(string $key): string
{
    return hash('sha256', $key);
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

echo "\n";
echo c(' ╔══════════════════════════════════════╗', 'cyan') . "\n";
echo c(' ║   Dafne Server  –  Setup             ║', 'cyan') . "\n";
echo c(' ╚══════════════════════════════════════╝', 'cyan') . "\n\n";

// ---------------------------------------------------------------------------
// Step 1 – Configuration
// ---------------------------------------------------------------------------

echo c(' Step 1 – Configuration', 'bold') . "\n\n";

$mysql_host = MYSQL_HOST;
$mysql_port = MYSQL_PORT;
$mysql_db   = MYSQL_DB;
$mysql_user = MYSQL_USER;
$mysql_pass = MYSQL_PASS;

info("MySQL host    : {$mysql_host}:{$mysql_port}");
info("MySQL database: {$mysql_db}");
info("MySQL user    : {$mysql_user}");
echo "\n";
info("Edit config.php to change these values.");
echo "\n";

// ---------------------------------------------------------------------------
// Step 2 – Privileged MySQL connection (create DB + user)
// ---------------------------------------------------------------------------

echo c(' Step 2 – Database & user creation', 'bold') . "\n\n";

$root_user = '';
$root_pass = '';

if ($root_user === '') {
    info("A privileged MySQL account is needed to create the database and");
    info("application user (if they do not already exist).");
    info("Leave the username blank to skip this step (e.g. DB already exists).");
    echo "\n";
    $root_user = prompt('MySQL admin username (e.g. root)', 'root');
}

$skip_db_create = ($root_user === '');

if (!$skip_db_create) {
    if ($root_pass === '') {
        $root_pass = prompt('MySQL admin password', '', secret: true);
    }

    try {
        // Connect without selecting a database
        $dsn_root = sprintf('mysql:host=%s;port=%d;charset=utf8mb4', $mysql_host, $mysql_port);
        $root_pdo = new PDO($dsn_root, $root_user, $root_pass, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        ]);
        ok("Connected to MySQL as '{$root_user}'.");
    } catch (PDOException $e) {
        abort("Could not connect to MySQL: " . $e->getMessage());
    }

    // Create database
    $db_quoted = '`' . str_replace('`', '``', $mysql_db) . '`';
    $root_pdo->exec("CREATE DATABASE IF NOT EXISTS {$db_quoted} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
    ok("Database '{$mysql_db}' is ready.");

    // Create user and grant privileges (MySQL 5.7 / 8 compatible)
    $escaped_user = $root_pdo->quote($mysql_user);
    $escaped_pass = $root_pdo->quote($mysql_pass);
    $escaped_host = $root_pdo->quote($mysql_host === 'localhost' ? 'localhost' : '%');

    // Check if user exists (works on MySQL 5.7+)
    $stmt = $root_pdo->prepare("SELECT COUNT(*) FROM mysql.user WHERE User = ? AND Host = ?");
    $stmt->execute([$mysql_user, $mysql_host === 'localhost' ? 'localhost' : '%']);
    $user_exists = (int) $stmt->fetchColumn() > 0;

    if (!$user_exists) {
        $root_pdo->exec(
            "CREATE USER {$escaped_user}@{$escaped_host} IDENTIFIED BY {$escaped_pass}"
        );
        ok("MySQL user '{$mysql_user}' created.");
    } else {
        info("MySQL user '{$mysql_user}' already exists — skipping creation.");
    }

    $root_pdo->exec(
        "GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, INDEX, ALTER, REFERENCES "
        . "ON {$db_quoted}.* TO {$escaped_user}@{$escaped_host}"
    );
    $root_pdo->exec('FLUSH PRIVILEGES');
    ok("Privileges granted to '{$mysql_user}' on '{$mysql_db}'.");

    unset($root_pdo);
} else {
    warn("Skipping database/user creation — assuming they already exist.");
}

echo "\n";

// ---------------------------------------------------------------------------
// Step 3 – Apply schema
// ---------------------------------------------------------------------------

echo c(' Step 3 – Schema', 'bold') . "\n\n";

try {
    $dsn = sprintf('mysql:host=%s;port=%d;dbname=%s;charset=utf8mb4',
                   $mysql_host, $mysql_port, $mysql_db);
    $pdo = new PDO($dsn, $mysql_user, $mysql_pass, [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    ]);
    ok("Connected to '{$mysql_db}' as '{$mysql_user}'.");
} catch (PDOException $e) {
    abort("Could not connect as application user: " . $e->getMessage());
}

$schema_file = __DIR__ . '/schema.sql';
if (!file_exists($schema_file)) {
    abort("schema.sql not found at {$schema_file}");
}

// Split on semicolons, skip comments and blank statements
$raw_sql    = (string) file_get_contents($schema_file);
$statements = array_filter(
    array_map('trim', explode(';', $raw_sql)),
    fn($s) => $s !== '' && !str_starts_with(ltrim($s), '--')
);

foreach ($statements as $sql) {
    try {
        $pdo->exec($sql);
        // Extract the table name for feedback
        if (preg_match('/CREATE TABLE\s+IF NOT EXISTS\s+`?(\w+)`?/i', $sql, $m)) {
            ok("Table '{$m[1]}' is ready.");
        }
    } catch (PDOException $e) {
        fail("Failed to execute statement: " . $e->getMessage());
        fail("SQL: " . substr($sql, 0, 120) . '…');
        abort("Schema application failed.");
    }
}

echo "\n";

// ---------------------------------------------------------------------------
// Step 4 – Filesystem layout
// ---------------------------------------------------------------------------

echo c(' Step 4 – Filesystem', 'bold') . "\n\n";

$dirs = [
    MODELS_DIR,
    UPLOAD_DATA_DIR,
];

$htaccess = "Require all denied\n";

foreach ($dirs as $dir) {
    if (!is_dir($dir)) {
        if (!mkdir($dir, 0755, true)) {
            abort("Could not create directory: {$dir}");
        }
        ok("Created {$dir}");
    } else {
        info("Exists  {$dir}");
    }

    $ht = $dir . '/.htaccess';
    if (!file_exists($ht)) {
        file_put_contents($ht, $htaccess);
        ok("Created {$ht}");
    } else {
        info("Exists  {$ht}");
    }
}

echo "\n";

// ---------------------------------------------------------------------------
// Step 5 – First admin user
// ---------------------------------------------------------------------------

echo c(' Step 5 – Administrator account', 'bold') . "\n\n";

$admin_count_stmt = $pdo->query('SELECT COUNT(*) FROM users WHERE is_admin = 1');
$admin_count      = (int) $admin_count_stmt->fetchColumn();

if ($admin_count > 0) {
    info("An administrator account already exists — skipping.");
} else {
    info("No administrator found. Create one now so you can log into the admin panel.");
    echo "\n";
    if (ask_yes("Create first administrator?", default: true)) {
        $name  = '';
        $email = '';
        while ($name === '') {
            $name = prompt('Full name');
            if ($name === '') warn("Name cannot be empty.");
        }
        while ($email === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $email = prompt('Email address');
            if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
                warn("Please enter a valid email address.");
                $email = '';
            }
        }

        $api_key     = generate_api_key();
        $key_hash    = hash_api_key($api_key);

        $s = $pdo->prepare(
            'INSERT INTO users (name, email, api_key_hash, is_admin) VALUES (?, ?, ?, 1)'
        );
        $s->execute([$name, $email, $key_hash]);
        $uid = (int) $pdo->lastInsertId();

        echo "\n";
        ok("Administrator '{$name}' created (id={$uid}).");
        echo "\n";
        echo c(' ┌─────────────────────────────────────────┐', 'yellow') . "\n";
        echo c(' │  API Key (copy now – shown only once)   │', 'yellow') . "\n";
        echo c(' │                                         │', 'yellow') . "\n";
        printf(c(' │  %-41s│', 'yellow') . "\n", $api_key);
        echo c(' └─────────────────────────────────────────┘', 'yellow') . "\n";
        echo "\n";
        warn("Store this key securely. It grants full admin access.");
    } else {
        warn("Skipped. You can create an admin via the CLI or directly in MySQL.");
    }
}

echo "\n";

// ---------------------------------------------------------------------------
// Done
// ---------------------------------------------------------------------------

echo c(' ══════════════════════════════════════════', 'green') . "\n";
echo c('  Setup complete.', 'green') . "\n";
echo c(' ══════════════════════════════════════════', 'green') . "\n\n";

info("Start the web server and visit /admin.php to manage users.");
echo "\n";