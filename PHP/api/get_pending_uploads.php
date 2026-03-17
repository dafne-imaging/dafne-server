<?php
declare(strict_types=1);

/**
 * POST /get_pending_uploads
 *
 * Returns the list of upload files available for a given model type.
 * Requires a merge-client API key.
 *
 * Request (JSON): { api_key, model_type }
 * Response: { uploads: [ { filename, timestamp, username, size, hash }, ... ] }
 */
function handle_get_pending_uploads(array $body): array
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

    return ['uploads' => get_pending_uploads($model_type)];
}