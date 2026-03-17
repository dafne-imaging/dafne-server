<?php
declare(strict_types=1);

/**
 * POST /get_model
 *
 * Request (JSON): { api_key, model_type, timestamp }
 * Response: binary file stream
 */
function handle_get_model(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $model_type = sanitize_model_type($body['model_type'] ?? '');
    $timestamp  = sanitize_timestamp($body['timestamp'] ?? '');

    if ($model_type === null || $timestamp === null) {
        return ['__status' => 400, 'message' => 'invalid parameters'];
    }

    if (!can_access_model($api_key, $model_type)) {
        return ['__status' => 403, 'message' => 'access denied for this model'];
    }

    $model_path = MODELS_DIR . "/{$model_type}/{$timestamp}.model";
    if (!is_file($model_path)) {
        return ['__status' => 500, 'message' => 'invalid model - not found'];
    }

    $username = get_username($api_key);
    server_log("get_model accessed by {$username} - {$model_type} - {$timestamp}");

    return ['__file' => $model_path, '__sha256' => file_sha256($model_path, true)];
}