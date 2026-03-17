<?php
declare(strict_types=1);

/**
 * POST /upload_new_model  (also used by admin_upload.php UI)
 *
 * Upload a new canonical model and its metadata file.
 * Requires an admin API key.
 *
 * Request (multipart): { api_key, model_type } + model_binary + model_json files
 * Response: { message, model_type, timestamp, sha256, already_existed }
 */
function handle_upload_new_model(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    $user    = get_user_by_key($api_key);

    if ($user === null || !(bool) $user['is_admin']) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $model_type = sanitize_model_type(trim($body['model_type'] ?? ''));
    if ($model_type === null || $model_type === '') {
        return ['__status' => 400, 'message' => 'model_type is required and must match [a-zA-Z0-9_-]+'];
    }

    if (!isset($_FILES['model_binary']) || !isset($_FILES['model_json'])) {
        return ['__status' => 400, 'message' => 'Both model_binary and model_json file fields are required'];
    }

    try {
        $result = perform_model_upload(
            $model_type,
            $_FILES['model_binary'],
            $_FILES['model_json'],
            $user['name']
        );
        return array_merge(['message' => 'Model uploaded successfully'], $result);
    } catch (InvalidArgumentException $e) {
        return ['__status' => 400, 'message' => $e->getMessage()];
    } catch (Throwable $e) {
        return ['__status' => 500, 'message' => 'Server error: ' . $e->getMessage()];
    }
}

/**
 * Validate uploaded files, create the model directory tree, and place the
 * files in the correct locations.  Throws on any error.
 *
 * @return array{model_type:string, timestamp:int, sha256:string, already_existed:bool}
 */
function perform_model_upload(
    string $model_type,
    array  $model_file,   // entry from $_FILES['model_binary']
    array  $json_file,    // entry from $_FILES['model_json']
    string $uploaded_by
): array {
    // Validate model type (redundant if caller already sanitised, but be safe)
    if (sanitize_model_type($model_type) === null) {
        throw new InvalidArgumentException('Invalid model type name.');
    }

    // File upload errors
    if ($model_file['error'] !== UPLOAD_ERR_OK) {
        throw new InvalidArgumentException('Model file upload error (code ' . $model_file['error'] . ').');
    }
    if ($json_file['error'] !== UPLOAD_ERR_OK) {
        throw new InvalidArgumentException('JSON file upload error (code ' . $json_file['error'] . ').');
    }

    // Extension checks
    $model_ext = strtolower(pathinfo($model_file['name'], PATHINFO_EXTENSION));
    $json_ext  = strtolower(pathinfo($json_file['name'],  PATHINFO_EXTENSION));
    if ($model_ext !== 'model') {
        throw new InvalidArgumentException('Model file must have a .model extension.');
    }
    if ($json_ext !== 'json') {
        throw new InvalidArgumentException('Metadata file must have a .json extension.');
    }

    // Validate JSON content before touching the filesystem
    $json_raw = file_get_contents($json_file['tmp_name']);
    if ($json_raw === false || json_decode($json_raw) === null) {
        throw new InvalidArgumentException('model.json is not valid JSON.');
    }

    // Directory setup
    $model_dir       = MODELS_DIR . "/{$model_type}";
    $uploads_dir     = $model_dir . '/uploads';
    $already_existed = is_dir($model_dir);

    foreach ([$model_dir, $uploads_dir] as $dir) {
        if (!is_dir($dir) && !mkdir($dir, 0755, true)) {
            throw new RuntimeException("Could not create directory: {$dir}");
        }
    }

    // Save model binary as a new timestamped canonical model
    $timestamp  = time();
    $model_path = "{$model_dir}/{$timestamp}.model";

    if (!move_uploaded_file($model_file['tmp_name'], $model_path)) {
        throw new RuntimeException('Failed to save model binary.');
    }

    // SHA-256 sidecar (allows info_model to serve the hash without re-hashing)
    $sha256 = (string) hash_file('sha256', $model_path);
    if (file_put_contents("{$model_path}.sha256", $sha256) === false) {
        // Non-fatal: hash will be computed on-demand if sidecar is missing
        server_log("upload_new_model: warning – could not write sha256 sidecar for {$model_path}");
    }

    // Save / overwrite model.json
    $json_path = "{$model_dir}/model.json";
    if (!move_uploaded_file($json_file['tmp_name'], $json_path)) {
        // Roll back the model binary
        @unlink($model_path);
        @unlink("{$model_path}.sha256");
        throw new RuntimeException('Failed to save model.json.');
    }

    server_log(
        "upload_new_model: {$model_type}/{$timestamp}.model saved by {$uploaded_by}" .
        ($already_existed ? ' (model type already existed)' : ' (new model type)'),
        true
    );

    return [
        'model_type'      => $model_type,
        'timestamp'       => $timestamp,
        'sha256'          => $sha256,
        'already_existed' => $already_existed,
    ];
}