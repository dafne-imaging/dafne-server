<?php
declare(strict_types=1);

// Path to the shared db directory.
// Can be overridden via the DAFNE_DB_DIR environment variable.
// Default assumes PHP/ is a subdirectory of the repo root (i.e. db/ is at ../db).
$_db_dir_raw = getenv('DAFNE_DB_DIR') ?: __DIR__ . '/../db';
$_db_dir     = realpath($_db_dir_raw) ?: $_db_dir_raw;

define('DB_DIR',            rtrim($_db_dir, '/'));
define('MODELS_DIR',        DB_DIR . '/models');
define('API_KEYS_FILE',     DB_DIR . '/api_keys.txt');
define('MERGE_API_KEYS_FILE', DB_DIR . '/merge_api_keys.txt');
define('SERVER_CONFIG_FILE', DB_DIR . '/server_config.json');
