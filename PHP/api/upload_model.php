<?php
declare(strict_types=1);

/**
 * POST /upload_model
 *
 * Request (multipart): { api_key, model_type, [dice], [original_hash],
 *                        [chunk_index], [total_chunks], [filename] } + model_binary file
 * Response: { message }
 */
function handle_upload_model(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        server_log("Upload rejected: invalid api key {$api_key}");
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $username   = sanitize_username_for_filename((string) get_username($api_key));
    $model_type = sanitize_model_type($body['model_type'] ?? '');

    if ($username === null) {
        server_log("Upload rejected: username cannot be sanitized to a valid filename component");
        return ['__status' => 500, 'message' => 'invalid username for file storage'];
    }

    if ($model_type === null) {
        return ['__status' => 400, 'message' => 'invalid model type'];
    }

    if (!in_array($model_type, get_model_types(), true)) {
        server_log("Upload of {$model_type} from {$username} rejected: unknown model type");
        return ['__status' => 500, 'message' => 'invalid model type'];
    }

    if (!can_access_model($api_key, $model_type)) {
        server_log("Upload of {$model_type} from {$username} rejected: access denied");
        return ['__status' => 403, 'message' => 'access denied for this model'];
    }

    if (!isset($_FILES['model_binary']) || $_FILES['model_binary']['error'] !== UPLOAD_ERR_OK) {
        return ['__status' => 400, 'message' => 'file upload error'];
    }

    $uploads_dir = MODELS_DIR . "/{$model_type}/uploads";
    if (!is_dir($uploads_dir)) {
        mkdir($uploads_dir, 0755, true);
    }

    $chunk_index  = isset($body['chunk_index'])  ? (int) $body['chunk_index']  : 0;
    $total_chunks = isset($body['total_chunks']) ? (int) $body['total_chunks'] : 1;

    if ($total_chunks > 1) {
        // Chunked upload: store each chunk in a temp directory, assemble on the last one.
        $safe_id    = preg_replace('/[^a-zA-Z0-9_\-.]/', '_', (string) ($body['filename'] ?? ''));
        $chunks_dir = "{$uploads_dir}/chunks_{$safe_id}";

        if (!is_dir($chunks_dir)) {
            mkdir($chunks_dir, 0755, true);
        }

        if (!move_uploaded_file($_FILES['model_binary']['tmp_name'], "{$chunks_dir}/{$chunk_index}.chunk")) {
            return ['__status' => 500, 'message' => 'failed to save chunk'];
        }

        if ($chunk_index < $total_chunks - 1) {
            return ['message' => 'chunk received'];
        }

        // Last chunk arrived: assemble all chunks into the final model file.
        $model_path = "{$uploads_dir}/" . time() . "_{$username}.model";
        $out = fopen($model_path, 'wb');
        if ($out === false) {
            return ['__status' => 500, 'message' => 'failed to create assembled file'];
        }
        for ($i = 0; $i < $total_chunks; $i++) {
            $cp   = "{$chunks_dir}/{$i}.chunk";
            $data = file_get_contents($cp);
            if ($data === false) {
                fclose($out);
                @unlink($model_path);
                return ['__status' => 500, 'message' => "missing chunk {$i}"];
            }
            fwrite($out, $data);
            unlink($cp);
        }
        fclose($out);
        rmdir($chunks_dir);
    } else {
        $model_path = "{$uploads_dir}/" . time() . "_{$username}.model";
        if (!move_uploaded_file($_FILES['model_binary']['tmp_name'], $model_path)) {
            return ['__status' => 500, 'message' => 'failed to save uploaded file'];
        }
    }

    $dice = $body['dice'] ?? null;
    server_log("upload_model accessed by {$username} - {$model_type} - {$model_path} - client dice {$dice}");

    // Verify integrity checksum if the client provided one.
    $original_hash = $body['original_hash'] ?? null;
    if ($original_hash !== null && $original_hash !== '') {
        if (file_sha256($model_path, false) !== $original_hash) {
            server_log("Upload integrity check failed for {$model_path}: hash mismatch");
            @unlink($model_path);
            return ['__status' => 500, 'message' => 'Communication error during upload'];
        }
    }

    return ['message' => 'upload successful'];
}