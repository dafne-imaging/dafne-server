<?php
declare(strict_types=1);

// Path to the shared db directory.
// Can be overridden via the DAFNE_DB_DIR environment variable.
// Default assumes PHP/ is a subdirectory of the repo root (i.e. db/ is at ../db).
$_db_dir_raw = getenv('DAFNE_DB_DIR') ?: __DIR__ . '/../db';
$_db_dir     = realpath($_db_dir_raw) ?: $_db_dir_raw;

define('DB_DIR',            rtrim($_db_dir, '/'));
define('MODELS_DIR',        DB_DIR . '/models');
define('SERVER_CONFIG_FILE', DB_DIR . '/server_config.json');

// MySQL connection settings – override via environment variables.
define('MYSQL_HOST', getenv('DAFNE_MYSQL_HOST') ?: 'localhost');
define('MYSQL_PORT', (int) (getenv('DAFNE_MYSQL_PORT') ?: 3306));
define('MYSQL_DB',   getenv('DAFNE_MYSQL_DB')   ?: 'dafne');
define('MYSQL_USER', getenv('DAFNE_MYSQL_USER')  ?: 'dafne');
define('MYSQL_PASS', getenv('DAFNE_MYSQL_PASS')  ?: '');

// Administrator email address.
// Used as the From: address for outgoing emails (e.g. new-user welcome messages)
// and as the recipient for server-side admin notifications.
define('ADMIN_EMAIL', getenv('DAFNE_ADMIN_EMAIL') ?: 'admin@example.com');

// Google reCAPTCHA v3 credentials.
// Obtain a key pair at https://www.google.com/recaptcha/admin/create (type: v3).
// Leave SITE_KEY empty to disable reCAPTCHA entirely (e.g. in development).
define('RECAPTCHA_SITE_KEY',   getenv('DAFNE_RECAPTCHA_SITE_KEY')   ?: '');
define('RECAPTCHA_SECRET_KEY', getenv('DAFNE_RECAPTCHA_SECRET_KEY') ?: '');
// Minimum score to accept (0.0 = all traffic, 1.0 = humans only; 0.5 is Google's default).
define('RECAPTCHA_MIN_SCORE',  (float) (getenv('DAFNE_RECAPTCHA_MIN_SCORE') ?: 0.5));
