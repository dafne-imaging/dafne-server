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

// Only POST is accepted on all routes.
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    json_respond(['message' => 'Method not allowed'], 405);
}

// Determine whether this is a multipart/form-data upload or a JSON body request.
$content_type = $_SERVER['CONTENT_TYPE'] ?? '';
$is_multipart = str_contains($content_type, 'multipart/form-data');

if ($is_multipart) {
    // Fields come from $_POST; files from $_FILES.
    $body = $_POST;
} else {
    $raw  = (string) file_get_contents('php://input');
    $body = json_decode($raw, true) ?? [];
}

// Strip the query string and any trailing slash from the request path.
$path = rtrim((string) parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH), '/');

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
    header('X-SHA256-Checksum: ' . $sha256);
    header('Content-Disposition: attachment; filename="' . basename($path) . '"');
    readfile($path);
    exit;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

switch ($path) {
    // --- Existing routes (identical behaviour to the Python FastAPI server) ---
    case '/get_available_models':
        handle_get_available_models($body);

    case '/info_model':
        handle_info_model($body);

    case '/get_model':
        handle_get_model($body);

    case '/upload_model':
        handle_upload_model($body);

    case '/upload_data':
        handle_upload_data($body);

    case '/evaluate_model':
        handle_evaluate_model($body);

    case '/log':
        handle_log_message($body);

    // --- New routes for the merge client ---
    case '/get_pending_uploads':
        handle_get_pending_uploads($body);

    case '/download_upload':
        handle_download_upload($body);

    case '/delete_upload':
        handle_delete_upload($body);

    case '/upload_merged_model':
        handle_upload_merged_model($body);

    default:
        json_respond(['message' => 'not found'], 404);
}

// ---------------------------------------------------------------------------
// Handlers – existing routes
// ---------------------------------------------------------------------------

function handle_get_available_models(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        json_respond(['message' => 'invalid access code'], 401);
    }
    $all_types  = get_model_types();
    $accessible = get_accessible_models($api_key);
    json_respond(['models' => array_values(array_intersect($all_types, $accessible))]);
}

function handle_info_model(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    $model_type = sanitize_model_type($body['model_type'] ?? '');
    if ($model_type === null) {
        json_respond(['message' => 'invalid model type'], 400);
    }

    if (!can_access_model($api_key, $model_type)) {
        json_respond(['message' => 'access denied for this model'], 403);
    }

    $timestamps = get_canonical_models($model_type);
    if (empty($timestamps)) {
        json_respond(['message' => 'no models found for this type'], 404);
    }

    $hashes = [];
    foreach ($timestamps as $stamp) {
        $model_path    = MODELS_DIR . "/{$model_type}/{$stamp}.model";
        $hashes[$stamp] = file_sha256($model_path, true); // cache alongside the model file
    }

    $latest = end($timestamps);
    $out = [
        'latest_timestamp' => $latest,
        'hash'             => $hashes[$latest],
        'hashes'           => $hashes,
        'timestamps'       => $timestamps,
    ];

    // Merge model.json metadata (model.json keys override computed keys,
    // matching Python's out_dict.update(json.load(...)) behaviour).
    $json_path = MODELS_DIR . "/{$model_type}/model.json";
    if (is_file($json_path)) {
        $model_info = json_decode((string) file_get_contents($json_path), true) ?? [];
        $out        = array_merge($out, $model_info);
    }

    json_respond($out);
}

function handle_get_model(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    $model_type = sanitize_model_type($body['model_type'] ?? '');
    $timestamp  = sanitize_timestamp($body['timestamp'] ?? '');

    if ($model_type === null || $timestamp === null) {
        json_respond(['message' => 'invalid parameters'], 400);
    }

    if (!can_access_model($api_key, $model_type)) {
        json_respond(['message' => 'access denied for this model'], 403);
    }

    $model_path = MODELS_DIR . "/{$model_type}/{$timestamp}.model";
    if (!is_file($model_path)) {
        json_respond(['message' => 'invalid model - not found'], 500);
    }

    $username = get_username($api_key);
    server_log("get_model accessed by {$username} - {$model_type} - {$timestamp}");

    file_respond($model_path, file_sha256($model_path, true));
}

function handle_upload_model(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        server_log("Upload rejected: invalid api key {$api_key}");
        json_respond(['message' => 'invalid access code'], 401);
    }

    $username   = get_username($api_key);
    $model_type = sanitize_model_type($body['model_type'] ?? '');

    if ($model_type === null) {
        json_respond(['message' => 'invalid model type'], 400);
    }

    if (!in_array($model_type, get_model_types(), true)) {
        server_log("Upload of {$model_type} from {$username} rejected: unknown model type");
        json_respond(['message' => 'invalid model type'], 500);
    }

    if (!can_access_model($api_key, $model_type)) {
        server_log("Upload of {$model_type} from {$username} rejected: access denied");
        json_respond(['message' => 'access denied for this model'], 403);
    }

    if (!isset($_FILES['model_binary']) || $_FILES['model_binary']['error'] !== UPLOAD_ERR_OK) {
        json_respond(['message' => 'file upload error'], 400);
    }

    $uploads_dir = MODELS_DIR . "/{$model_type}/uploads";
    if (!is_dir($uploads_dir)) {
        mkdir($uploads_dir, 0755, true);
    }

    $model_path = "{$uploads_dir}/" . time() . "_{$username}.model";

    if (!move_uploaded_file($_FILES['model_binary']['tmp_name'], $model_path)) {
        json_respond(['message' => 'failed to save uploaded file'], 500);
    }

    $dice = $body['dice'] ?? null;
    server_log("upload_model accessed by {$username} - {$model_type} - {$model_path} - client dice {$dice}");

    // Verify integrity checksum if the client provided one.
    $original_hash = $body['original_hash'] ?? null;
    if ($original_hash !== null && $original_hash !== '') {
        if (file_sha256($model_path, false) !== $original_hash) {
            server_log("Upload integrity check failed for {$model_path}: hash mismatch");
            @unlink($model_path);
            json_respond(['message' => 'Communication error during upload'], 500);
        }
    }

    json_respond(['message' => 'upload successful']);
}

function handle_upload_data(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        server_log("Data upload rejected: invalid api key {$api_key}");
        json_respond(['message' => 'invalid access code'], 401);
    }

    $username = get_username($api_key);

    if (!isset($_FILES['data_binary']) || $_FILES['data_binary']['error'] !== UPLOAD_ERR_OK) {
        json_respond(['message' => 'file upload error'], 400);
    }

    $data_dir = DB_DIR . "/uploaded_data/{$username}";
    if (!is_dir($data_dir)) {
        mkdir($data_dir, 0755, true);
    }

    $dest = "{$data_dir}/" . time() . '.npz';
    if (!move_uploaded_file($_FILES['data_binary']['tmp_name'], $dest)) {
        json_respond(['message' => 'failed to save uploaded file'], 500);
    }

    server_log("upload_data accessed by {$username} - upload successful");
    json_respond(['message' => 'upload successful']);
}

function handle_evaluate_model(array $body): never
{
    // Authentication check kept for API compatibility.
    if (!valid_credentials($body['api_key'] ?? '')) {
        json_respond(['message' => 'invalid access code'], 401);
    }
    // No-op: model evaluation is now performed by the external merge client.
    json_respond(['message' => 'starting evaluation successful']);
}

function handle_log_message(array $body): never
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    $username = get_username($api_key);
    $message  = $body['message'] ?? '';
    server_log("Log message from {$username} - {$message}", true);
    json_respond(['message' => 'ok']);
}

// ---------------------------------------------------------------------------
// Handlers – merge client routes
// ---------------------------------------------------------------------------

/**
 * POST /get_pending_uploads
 * Returns the list of upload files available for a given model type.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type }
 * Response: { uploads: [ { filename, timestamp, username, size, hash }, ... ] }
 */
function handle_get_pending_uploads(array $body): never
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');

    if ($model_type === null) {
        json_respond(['message' => 'invalid model type'], 400);
    }

    if (!can_merge_model($api_key, $model_type)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    if (!in_array($model_type, get_model_types(), true)) {
        json_respond(['message' => 'unknown model type'], 404);
    }

    json_respond(['uploads' => get_pending_uploads($model_type)]);
}

/**
 * POST /download_upload
 * Stream a specific upload file to the merge client.
 * SHA-256 checksum is provided in the X-SHA256-Checksum response header.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type, filename }
 * Response: binary file stream
 */
function handle_download_upload(array $body): never
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');
    $filename   = sanitize_upload_filename($body['filename'] ?? '');

    if ($model_type === null || $filename === null) {
        json_respond(['message' => 'invalid parameters'], 400);
    }

    if (!can_merge_model($api_key, $model_type)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    $file_path = MODELS_DIR . "/{$model_type}/uploads/{$filename}";
    if (!is_file($file_path)) {
        json_respond(['message' => 'file not found'], 404);
    }

    // Hash is not cached for uploads since they are transient.
    file_respond($file_path, file_sha256($file_path, false));
}

/**
 * POST /delete_upload
 * Delete a specific upload file once the merge client has finished with it.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type, filename }
 * Response: { message }
 */
function handle_delete_upload(array $body): never
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');
    $filename   = sanitize_upload_filename($body['filename'] ?? '');

    if ($model_type === null || $filename === null) {
        json_respond(['message' => 'invalid parameters'], 400);
    }

    if (!can_merge_model($api_key, $model_type)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    $file_path = MODELS_DIR . "/{$model_type}/uploads/{$filename}";
    if (!is_file($file_path)) {
        json_respond(['message' => 'file not found'], 404);
    }

    if (!unlink($file_path)) {
        json_respond(['message' => 'failed to delete file'], 500);
    }
    @unlink($file_path . '.sha256'); // clean up sidecar if present

    $username = get_merge_username($api_key);
    server_log("delete_upload: {$model_type}/{$filename} by {$username}");
    json_respond(['message' => 'deleted successfully']);
}

/**
 * POST /upload_merged_model
 * Accept a merged canonical model from the merge client.
 * The client MUST supply the SHA-256 hex digest of the file; the server
 * rejects the upload if the digest does not match.
 * Requires a merge-client API key.
 *
 * Request (multipart): api_key, model_type, sha256, model_binary (file)
 * Response: { message, timestamp }
 */
function handle_upload_merged_model(array $body): never
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');

    if ($model_type === null) {
        json_respond(['message' => 'invalid model type'], 400);
    }

    if (!can_merge_model($api_key, $model_type)) {
        json_respond(['message' => 'invalid access code'], 401);
    }

    if (!in_array($model_type, get_model_types(), true)) {
        json_respond(['message' => 'unknown model type'], 404);
    }

    if (!isset($_FILES['model_binary']) || $_FILES['model_binary']['error'] !== UPLOAD_ERR_OK) {
        json_respond(['message' => 'file upload error'], 400);
    }

    $provided_sha256 = trim($body['sha256'] ?? '');
    if (strlen($provided_sha256) !== 64 || !ctype_xdigit($provided_sha256)) {
        json_respond(['message' => 'sha256 field is required (64-char hex digest)'], 400);
    }

    // Verify checksum against the received bytes before committing to disk.
    $tmp_path    = $_FILES['model_binary']['tmp_name'];
    $actual_sha256 = (string) hash_file('sha256', $tmp_path);
    if ($actual_sha256 !== $provided_sha256) {
        server_log("upload_merged_model: SHA-256 mismatch for {$model_type} - upload rejected");
        json_respond(['message' => 'SHA-256 checksum mismatch - upload corrupted or tampered'], 422);
    }

    $timestamp  = time();
    $dest_path  = MODELS_DIR . "/{$model_type}/{$timestamp}.model";
    $tmp_dest   = $dest_path . '.tmp';

    // Write to a .tmp file first; rename is atomic on POSIX filesystems,
    // preventing clients from reading a partially-written model.
    if (!move_uploaded_file($tmp_path, $tmp_dest)) {
        json_respond(['message' => 'failed to save merged model'], 500);
    }

    if (!rename($tmp_dest, $dest_path)) {
        @unlink($tmp_dest);
        json_respond(['message' => 'failed to commit merged model'], 500);
    }

    // Store the verified checksum sidecar so info_model can serve it immediately.
    file_put_contents($dest_path . '.sha256', $actual_sha256);

    $username = get_merge_username($api_key);
    server_log("upload_merged_model: saved {$model_type}/{$timestamp}.model by {$username}", true);

    // Prune old canonical models (uncomment to enable).
    // delete_older_canonical_models($model_type);

    json_respond(['message' => 'merged model uploaded successfully', 'timestamp' => $timestamp]);
}
