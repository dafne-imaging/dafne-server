<?php
declare(strict_types=1);

/**
 * POST /log
 *
 * Request (JSON): { api_key, message }
 * Response: { message }
 */
function handle_log(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $username = get_username($api_key);
    $message  = $body['message'] ?? '';
    server_log("Log message from {$username} - {$message}", true);
    return ['message' => 'ok'];
}