<?php
declare(strict_types=1);

/**
 * POST /download_upload
 *
 * Stream a specific upload file to the merge client.
 * SHA-256 checksum is provided in the X-SHA256-Checksum response header.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type, filename }
 * Response: binary file stream
 */
function handle_download_upload(array $body): array
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');
    $filename   = sanitize_upload_filename($body['filename'] ?? '');

    if ($model_type === null || $filename === null) {
        return ['__status' => 400, 'message' => 'invalid parameters'];
    }

    if (!can_merge_model($api_key, $model_type)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $file_path = MODELS_DIR . "/{$model_type}/uploads/{$filename}";
    if (!is_file($file_path)) {
        return ['__status' => 404, 'message' => 'file not found'];
    }

    // Hash is not cached for uploads since they are transient.
    return ['__file' => $file_path, '__sha256' => file_sha256($file_path, false)];
}