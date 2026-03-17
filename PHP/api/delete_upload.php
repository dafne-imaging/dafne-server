<?php
declare(strict_types=1);

/**
 * POST /delete_upload
 *
 * Delete a specific upload file once the merge client has finished with it.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type, filename }
 * Response: { message }
 */
function handle_delete_upload(array $body): array
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

    if (!unlink($file_path)) {
        return ['__status' => 500, 'message' => 'failed to delete file'];
    }
    @unlink($file_path . '.sha256'); // clean up sidecar if present

    $username = get_merge_username($api_key);
    server_log("delete_upload: {$model_type}/{$filename} by {$username}");
    return ['message' => 'deleted successfully'];
}