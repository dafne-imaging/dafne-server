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
// Google reCAPTCHA Enterprise  (leave SITE_KEY empty to disable)
// ---------------------------------------------------------------------------

// Site key from the reCAPTCHA Enterprise console.
define('RECAPTCHA_SITE_KEY',   '');
// Google Cloud project ID that owns the reCAPTCHA Enterprise site key.
define('RECAPTCHA_PROJECT_ID', '');
// Google Cloud API key with the reCAPTCHA Enterprise API enabled.
define('RECAPTCHA_API_KEY',    '');
// Minimum score to accept (0.0 = all traffic, 1.0 = humans only).
define('RECAPTCHA_MIN_SCORE',  0.5);

// ---------------------------------------------------------------------------
// File download
// ---------------------------------------------------------------------------

// Set to true to delegate large file downloads to the web server instead of
// streaming them through PHP.  Requires the appropriate server module:
//   Apache:    mod_xsendfile  → header 'X-Sendfile'
//   nginx:     ngx_http_proxy → header 'X-Accel-Redirect'
//   lighttpd:  mod_fastcgi    → header 'X-Sendfile'
// When false, PHP streams the file in 1 MB chunks (works on any server).
define('XSENDFILE_ENABLED', false);
define('XSENDFILE_HEADER',  'X-Sendfile'); // change to 'X-Accel-Redirect' for nginx