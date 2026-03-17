<?php
declare(strict_types=1);

/**
 * POST /evaluate_model
 *
 * Request (JSON): { api_key }
 * Response: { message }
 *
 * No-op: model evaluation is performed by the external merge client.
 */
function handle_evaluate_model(array $body): array
{
    if (!valid_credentials($body['api_key'] ?? '')) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }
    return ['message' => 'starting evaluation successful'];
}