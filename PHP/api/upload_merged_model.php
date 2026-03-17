<?php
declare(strict_types=1);

/**
 * POST /upload_merged_model
 *
 * Accept a merged canonical model from the merge client.
 * The client MUST supply the SHA-256 hex digest of the file; the server
 * rejects the upload if the digest does not match.
 * Requires a merge-client API key.
 *
 * Request (multipart): { api_key, model_type, sha256 } + model_binary file
 * Response: { message, timestamp }
 */
function handle_upload_merged_model(array $body): array
{
    $api_key    = $body['api_key'] ?? '';
    $model_type = sanitize_model_type($body['model_type'] ?? '');

    if ($model_type === null) {
        return ['__status' => 400, 'message' => 'invalid model type'];
    }

    if (!can_merge_model($api_key, $model_type)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    if (!in_array($model_type, get_model_types(), true)) {
        return ['__status' => 404, 'message' => 'unknown model type'];
    }

    if (!isset($_FILES['model_binary']) || $_FILES['model_binary']['error'] !== UPLOAD_ERR_OK) {
        return ['__status' => 400, 'message' => 'file upload error'];
    }

    $provided_sha256 = trim($body['sha256'] ?? '');
    if (strlen($provided_sha256) !== 64 || !ctype_xdigit($provided_sha256)) {
        return ['__status' => 400, 'message' => 'sha256 field is required (64-char hex digest)'];
    }

    // Verify checksum against the received bytes before committing to disk.
    $tmp_path      = $_FILES['model_binary']['tmp_name'];
    $actual_sha256 = (string) hash_file('sha256', $tmp_path);
    if ($actual_sha256 !== $provided_sha256) {
        server_log("upload_merged_model: SHA-256 mismatch for {$model_type} - upload rejected");
        return ['__status' => 422, 'message' => 'SHA-256 checksum mismatch - upload corrupted or tampered'];
    }

    $timestamp = time();
    $dest_path = MODELS_DIR . "/{$model_type}/{$timestamp}.model";
    $tmp_dest  = $dest_path . '.tmp';

    // Write to a .tmp file first; rename is atomic on POSIX filesystems,
    // preventing clients from reading a partially-written model.
    if (!move_uploaded_file($tmp_path, $tmp_dest)) {
        return ['__status' => 500, 'message' => 'failed to save merged model'];
    }

    if (!rename($tmp_dest, $dest_path)) {
        @unlink($tmp_dest);
        return ['__status' => 500, 'message' => 'failed to commit merged model'];
    }

    // Store the verified checksum sidecar so info_model can serve it immediately.
    file_put_contents($dest_path . '.sha256', $actual_sha256);

    $username = get_merge_username($api_key);
    server_log("upload_merged_model: saved {$model_type}/{$timestamp}.model by {$username}", true);

    // Prune old canonical models (uncomment to enable).
    // delete_older_canonical_models($model_type);

    return ['message' => 'merged model uploaded successfully', 'timestamp' => $timestamp];
}