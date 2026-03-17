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
