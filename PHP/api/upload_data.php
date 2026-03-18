<?php
declare(strict_types=1);

/**
 * POST /upload_data
 *
 * Request (multipart): { api_key } + data_binary file
 * Response: { message }
 */
function handle_upload_data(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        server_log("Data upload rejected: invalid api key {$api_key}");
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $username = get_username($api_key);

    if (!isset($_FILES['data_binary']) || $_FILES['data_binary']['error'] !== UPLOAD_ERR_OK) {
        return ['__status' => 400, 'message' => 'file upload error'];
    }

    $data_dir = UPLOAD_DATA_DIR . "/{$username}";
    if (!is_dir($data_dir)) {
        mkdir($data_dir, 0755, true);
    }

    $dest = "{$data_dir}/" . time() . '.npz';
    if (!move_uploaded_file($_FILES['data_binary']['tmp_name'], $dest)) {
        return ['__status' => 500, 'message' => 'failed to save uploaded file'];
    }

    server_log("upload_data accessed by {$username} - upload successful");
    return ['message' => 'upload successful'];
}