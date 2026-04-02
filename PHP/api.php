<?php
declare(strict_types=1);

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/logger.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';

// ---------------------------------------------------------------------------
// Request bootstrapping
// ---------------------------------------------------------------------------

// Build the request body from whichever source the client used.
// Some reverse proxies silently convert POST to GET, so we accept both.
// Priority: multipart POST > JSON POST body > GET query string.
$method       = $_SERVER['REQUEST_METHOD'];
$content_type = $_SERVER['CONTENT_TYPE'] ?? '';
$is_multipart = str_contains($content_type, 'multipart/form-data');

if ($method === 'POST' && $is_multipart) {
    // File upload: fields from $_POST, files from $_FILES.
    $body = $_POST;
} elseif ($method === 'POST') {
    $raw  = (string) file_get_contents('php://input');
    $body = json_decode($raw, true) ?? [];
} elseif ($method === 'GET') {
    $body = $_GET;
} else {
    json_respond(['message' => 'Method not allowed'], 405);
}

// Strip the query string, the subdirectory prefix (if deployed under one),
// and any trailing slash from the request path so route keys always start
// with a single slash (e.g. "/get_model" regardless of base directory).
$full_path  = (string) parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$script_dir = rtrim(dirname($_SERVER['SCRIPT_NAME']), '/');
$path       = $script_dir !== ''
    ? '/' . ltrim(substr($full_path, strlen($script_dir)), '/')
    : $full_path;
$path = rtrim($path, '/') ?: '/';

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

/**
 * Send a JSON response and terminate.
 */
function json_respond(array $data, int $status = 200): never
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($data);
    exit;
}

/**
 * Stream a binary file with SHA-256 checksum header and terminate.
 */
function file_respond(string $path, string $sha256): never
{
    header('Content-Type: application/octet-stream');
    header('Content-Length: ' . filesize($path));
    if ($sha256) {
        header('X-SHA256-Checksum: ' . $sha256);
    }
    header('Content-Disposition: attachment; filename="' . basename($path) . '"');
    readfile($path);
    exit;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

// Maps URL path → handler file (under api/) and function name (handle_{name}).
$routes = [
    '/get_available_models' => 'get_available_models',
    '/info_model'           => 'info_model',
    '/get_model'            => 'get_model',
    '/upload_model'         => 'upload_model',
    '/upload_data'          => 'upload_data',
    '/evaluate_model'       => 'evaluate_model',
    '/log'                  => 'log',
    '/get_pending_uploads'  => 'get_pending_uploads',
    '/download_upload'      => 'download_upload',
    '/delete_upload'        => 'delete_upload',
    '/upload_merged_model'  => 'upload_merged_model',
    '/upload_new_model'     => 'upload_new_model',
];

if (!isset($routes[$path])) {
    json_respond(['message' => 'not found'], 404);
}

$handler = $routes[$path];
require_once __DIR__ . '/api/' . $handler . '.php';

$fn     = 'handle_' . $handler;
$result = $fn($body);

// Dispatch the handler's return value.
if (isset($result['__file'])) {
    file_respond($result['__file'], $result['__sha256'] ?? '');
}

$status = $result['__status'] ?? 200;
unset($result['__status']);
json_respond($result, $status);