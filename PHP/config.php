<?php
declare(strict_types=1);

// ---------------------------------------------------------------------------
// Filesystem paths
// ---------------------------------------------------------------------------

define('MODELS_DIR',       realpath( __DIR__ . '/../models'));
define('UPLOAD_DATA_DIR',  realpath( __DIR__ . '/../uploaded_data'));

// How many canonical model files to retain per model type when pruning.
define('NR_MODELS_TO_KEEP', 10);

// ---------------------------------------------------------------------------
// MySQL connection
// ---------------------------------------------------------------------------

define('MYSQL_HOST', 'localhost');
define('MYSQL_PORT', 3306);
define('MYSQL_DB',   'dafne');
define('MYSQL_USER', 'dafne');
define('MYSQL_PASS', '');

// ---------------------------------------------------------------------------
// Email
// ---------------------------------------------------------------------------

// From: address for outgoing emails (welcome messages, admin notifications).
define('ADMIN_EMAIL', 'admin@example.com');

// ---------------------------------------------------------------------------
// Google reCAPTCHA v3  (leave SITE_KEY empty to disable)
// ---------------------------------------------------------------------------

define('RECAPTCHA_SITE_KEY',   '');
define('RECAPTCHA_SECRET_KEY', '');
// Minimum score to accept (0.0 = all traffic, 1.0 = humans only).
define('RECAPTCHA_MIN_SCORE',  0.5);