<?php
declare(strict_types=1);

/**
 * POST /info_model
 *
 * Request (JSON): { api_key, model_type }
 * Response: { latest_timestamp, hash, hashes, timestamps, ...model.json fields }
 */
function handle_info_model(array $body): array
{
    $api_key = $body['api_key'] ?? '';
    if (!valid_credentials($api_key)) {
        return ['__status' => 401, 'message' => 'invalid access code'];
    }

    $model_type = sanitize_model_type($body['model_type'] ?? '');
    if ($model_type === null) {
        return ['__status' => 400, 'message' => 'invalid model type'];
    }

    if (!can_access_model($api_key, $model_type)) {
        return ['__status' => 403, 'message' => 'access denied for this model'];
    }

    $timestamps = get_canonical_models($model_type);
    if (empty($timestamps)) {
        return ['__status' => 404, 'message' => 'no models found for this type'];
    }

    $hashes = [];
    foreach ($timestamps as $stamp) {
        $model_path    = MODELS_DIR . "/{$model_type}/{$stamp}.model";
        $hashes[$stamp] = file_sha256($model_path, true);
    }

    $latest      = end($timestamps);
    $latest_path = MODELS_DIR . "/{$model_type}/{$latest}.model";
    $out = [
        'latest_timestamp' => $latest,
        'latest_size'      => filesize($latest_path),
        'hash'             => $hashes[$latest],
        'hashes'           => $hashes,
        'timestamps'       => $timestamps,
    ];

    // Merge model.json metadata (model.json keys override computed keys,
    // matching Python's out_dict.update(json.load(...)) behaviour).
    $json_path = MODELS_DIR . "/{$model_type}/model.json";
    if (is_file($json_path)) {
        $model_info = json_decode((string) file_get_contents($json_path), true) ?? [];
        $out        = array_merge($out, $model_info);
    }

    return $out;
}