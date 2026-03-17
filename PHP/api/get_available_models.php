<?php
declare(strict_types=1);

/**
 * POST /get_available_models
 *
 * Request (JSON): { api_key }
 * Response: { models: [...] }
 */
function handle_get_available_models(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }
    $all_types  = get_model_types();
    $accessible = get_accessible_models($api_key);
    return ['models' => array_values(array_intersect($all_types, $accessible))];
}