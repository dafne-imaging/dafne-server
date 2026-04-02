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
 * Request (multipart): { api_key, model_type, sha256,
 *                        [chunk_index], [total_chunks], [filename] } + model_binary file
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

    $chunk_index  = isset($body['chunk_index'])  ? (int) $body['chunk_index']  : 0;
    $total_chunks = isset($body['total_chunks']) ? (int) $body['total_chunks'] : 1;

    $model_dir  = MODELS_DIR . "/{$model_type}";
    $assembled  = false;
    $chunks_dir = null;

    if ($total_chunks > 1) {
        // Chunked upload: store each chunk in a temp directory, assemble on the last one.
        $safe_id    = preg_replace('/[^a-zA-Z0-9_\-.]/', '_', (string) ($body['filename'] ?? ''));
        $chunks_dir = "{$model_dir}/chunks_{$safe_id}";

        if (!is_dir($chunks_dir)) {
            mkdir($chunks_dir, 0755, true);
        }

        if (!move_uploaded_file($_FILES['model_binary']['tmp_name'], "{$chunks_dir}/{$chunk_index}.chunk")) {
            return ['__status' => 500, 'message' => 'failed to save chunk'];
        }

        if ($chunk_index < $total_chunks - 1) {
            return ['message' => 'chunk received'];
        }

        // Last chunk arrived: assemble all chunks.
        $assembled_path = "{$chunks_dir}/assembled.model";
        $out = fopen($assembled_path, 'wb');
        if ($out === false) {
            return ['__status' => 500, 'message' => 'failed to create assembled file'];
        }
        for ($i = 0; $i < $total_chunks; $i++) {
            $cp   = "{$chunks_dir}/{$i}.chunk";
            $data = file_get_contents($cp);
            if ($data === false) {
                fclose($out);
                @unlink($assembled_path);
                return ['__status' => 500, 'message' => "missing chunk {$i}"];
            }
            fwrite($out, $data);
            unlink($cp);
        }
        fclose($out);
        $file_to_verify = $assembled_path;
        $assembled      = true;
    } else {
        $file_to_verify = $_FILES['model_binary']['tmp_name'];
    }

    $provided_sha256 = trim($body['sha256'] ?? '');
    if (strlen($provided_sha256) !== 64 || !ctype_xdigit($provided_sha256)) {
        if ($assembled) { @unlink($file_to_verify); @rmdir($chunks_dir); }
        return ['__status' => 400, 'message' => 'sha256 field is required (64-char hex digest)'];
    }

    // Verify checksum against the received bytes before committing to disk.
    $actual_sha256 = (string) hash_file('sha256', $file_to_verify);
    if ($actual_sha256 !== $provided_sha256) {
        if ($assembled) { @unlink($file_to_verify); @rmdir($chunks_dir); }
        server_log("upload_merged_model: SHA-256 mismatch for {$model_type} - upload rejected");
        return ['__status' => 422, 'message' => 'SHA-256 checksum mismatch - upload corrupted or tampered'];
    }

    $timestamp = time();
    $dest_path = "{$model_dir}/{$timestamp}.model";
    $tmp_dest  = $dest_path . '.tmp';

    // Write to a .tmp file first; rename is atomic on POSIX filesystems,
    // preventing clients from reading a partially-written model.
    if ($assembled) {
        if (!rename($file_to_verify, $tmp_dest)) {
            @unlink($file_to_verify);
            @rmdir($chunks_dir);
            return ['__status' => 500, 'message' => 'failed to save merged model'];
        }
        @rmdir($chunks_dir);
    } else {
        if (!move_uploaded_file($file_to_verify, $tmp_dest)) {
            return ['__status' => 500, 'message' => 'failed to save merged model'];
        }
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