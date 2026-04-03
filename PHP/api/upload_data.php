<?php
declare(strict_types=1);

/**
 * POST /upload_data
 *
 * Request (multipart): { api_key, [chunk_index], [total_chunks], [filename] } + data_binary file
 * Response: { message }
 */
function handle_upload_data(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        server_log("Data upload rejected: invalid api key {$api_key}");
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $username = sanitize_username_for_filename((string) get_username($api_key));

    if ($username === null) {
        server_log("Data upload rejected: username cannot be sanitized to a valid filename component");
        return ['__status' => 500, 'message' => 'invalid username for file storage'];
    }

    if (!isset($_FILES['data_binary']) || $_FILES['data_binary']['error'] !== UPLOAD_ERR_OK) {
        return ['__status' => 400, 'message' => 'file upload error'];
    }

    $data_dir = UPLOAD_DATA_DIR . "/{$username}";
    if (!is_dir($data_dir)) {
        mkdir($data_dir, 0755, true);
    }

    $chunk_index  = isset($body['chunk_index'])  ? (int) $body['chunk_index']  : 0;
    $total_chunks = isset($body['total_chunks']) ? (int) $body['total_chunks'] : 1;

    if ($total_chunks > 1) {
        // Chunked upload: store each chunk in a temp directory, assemble on the last one.
        $safe_id    = preg_replace('/[^a-zA-Z0-9_\-.]/', '_', (string) ($body['filename'] ?? ''));
        $chunks_dir = "{$data_dir}/chunks_{$safe_id}";

        if (!is_dir($chunks_dir)) {
            mkdir($chunks_dir, 0755, true);
        }

        if (!move_uploaded_file($_FILES['data_binary']['tmp_name'], "{$chunks_dir}/{$chunk_index}.chunk")) {
            return ['__status' => 500, 'message' => 'failed to save chunk'];
        }

        if ($chunk_index < $total_chunks - 1) {
            return ['message' => 'chunk received'];
        }

        // Last chunk arrived: assemble all chunks into the final data file.
        $dest = "{$data_dir}/" . time() . '.npz';
        $out  = fopen($dest, 'wb');
        if ($out === false) {
            return ['__status' => 500, 'message' => 'failed to create assembled file'];
        }
        for ($i = 0; $i < $total_chunks; $i++) {
            $cp   = "{$chunks_dir}/{$i}.chunk";
            $data = file_get_contents($cp);
            if ($data === false) {
                fclose($out);
                @unlink($dest);
                return ['__status' => 500, 'message' => "missing chunk {$i}"];
            }
            fwrite($out, $data);
            unlink($cp);
        }
        fclose($out);
        rmdir($chunks_dir);
    } else {
        $dest = "{$data_dir}/" . time() . '.npz';
        if (!move_uploaded_file($_FILES['data_binary']['tmp_name'], $dest)) {
            return ['__status' => 500, 'message' => 'failed to save uploaded file'];
        }
    }

    server_log("upload_data accessed by {$username} - upload successful");

    $user = get_user_by_key($api_key);
    $display_name = $user['name'] ?? $username;
    $subject = "New data upload from {$display_name}";
    $body = implode("\r\n", [
        'A new data package has been uploaded.',
        '',
        str_repeat('─', 50),
        "User:      {$display_name}",
        'Timestamp: ' . date('Y-m-d H:i:s') . ' UTC',
        'File:      ' . basename($dest),
        str_repeat('─', 50),
        '',
        'This message was sent automatically by the Dafne server.',
    ]);
    $headers = implode("\r\n", [
        'From: '        . ADMIN_EMAIL,
        'Content-Type: text/plain; charset=UTF-8',
        'MIME-Version: 1.0',
    ]);
    mail(ADMIN_EMAIL, $subject, $body, $headers);

    return ['message' => 'upload successful'];
}