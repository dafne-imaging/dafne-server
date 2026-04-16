<?php
declare(strict_types=1);

// ---------------------------------------------------------------------------
// Model directory helpers
// ---------------------------------------------------------------------------

/**
 * Return the list of available model type names (subdirectory names inside MODELS_DIR).
 */
function get_model_types(): array
{
    $dirs = glob(MODELS_DIR . '/*', GLOB_ONLYDIR);
    if ($dirs === false) {
        return [];
    }
    return array_values(array_map('basename', $dirs));
}

/**
 * Return sorted array of timestamp strings for the canonical models of $model_type.
 * Canonical models live directly inside MODELS_DIR/{model_type}/ as {timestamp}.model files.
 */
function get_canonical_models(string $model_type): array
{
    $files = glob(MODELS_DIR . "/{$model_type}/*.model");
    if (empty($files)) {
        return [];
    }
    $timestamps = array_map(static function (string $f): string {
        return pathinfo(basename($f), PATHINFO_FILENAME);
    }, $files);
    sort($timestamps);
    return $timestamps;
}

/**
 * Return metadata for every file currently sitting in the uploads/ directory
 * for $model_type.  Each entry contains:
 *   filename  – basename of the file (e.g. "1710001000_jakob.model")
 *   timestamp – Unix timestamp extracted from the filename
 *   username  – uploader username extracted from the filename
 *   size      – file size in bytes
 *   hash      – SHA-256 hex digest of the file
 *
 * The list is ordered by timestamp ascending.
 */
function get_pending_uploads(string $model_type): array
{
    $uploads_dir = MODELS_DIR . "/{$model_type}/uploads";
    $files       = glob("{$uploads_dir}/*.model");
    if (empty($files)) {
        return [];
    }

    $result = [];
    foreach ($files as $file) {
        $basename = basename($file);
        // Expected filename format: {timestamp}_{username}.model
        $stem   = pathinfo($basename, PATHINFO_FILENAME);
        $parts  = explode('_', $stem, 2);
        $result[] = [
            'filename'  => $basename,
            'timestamp' => $parts[0] ?? '',
            'username'  => $parts[1] ?? '',
            'size'      => filesize($file),
        ];
    }

    usort($result, static fn ($a, $b) => strcmp($a['timestamp'], $b['timestamp']));
    return $result;
}

// ---------------------------------------------------------------------------
// SHA-256 helpers
// ---------------------------------------------------------------------------

/**
 * Return the SHA-256 hex digest of $path.
 *
 * When $cache is true the result is stored in / read from a "{path}.sha256"
 * sidecar file, matching the behaviour of Python's
 * calculate_file_hash(path, cache_results=True).
 */
function file_sha256(string $path, bool $cache = false): string
{
    if ($cache) {
        $sidecar = $path . '.sha256';
        if (is_file($sidecar)) {
            $stored = trim((string) file_get_contents($sidecar));
            if (strlen($stored) === 64) {
                return $stored;
            }
        }
    }

    $hash = (string) hash_file('sha256', $path);

    if ($cache) {
        file_put_contents($path . '.sha256', $hash);
    }

    return $hash;
}

// ---------------------------------------------------------------------------
// Canonical model pruning  (mirrored from Python utils.py)
// ---------------------------------------------------------------------------

/**
 * Delete the oldest canonical models for $model_type, keeping only the newest $keep.
 * Uses NR_MODELS_TO_KEEP from config.php when not supplied explicitly.
 *
 * NOTE: This function is currently COMMENTED OUT at the call-site in
 * handle_upload_merged_model(). Uncomment that call once you are ready to
 * enable automatic pruning.
 */
function delete_older_canonical_models(string $model_type, int $keep = -1): void
{
    if ($keep < 0) {
        $keep = NR_MODELS_TO_KEEP;
    }

    $timestamps = get_canonical_models($model_type);
    if (count($timestamps) <= $keep) {
        return;
    }

    $to_delete = array_slice($timestamps, 0, count($timestamps) - $keep);
    foreach ($to_delete as $stamp) {
        $path = MODELS_DIR . "/{$model_type}/{$stamp}.model";
        @unlink($path);
        @unlink($path . '.sha256');
        server_log("Pruned old canonical model: {$model_type}/{$stamp}.model");
    }
}

// ---------------------------------------------------------------------------
// Input sanitization
// ---------------------------------------------------------------------------

/**
 * Validate and return a model-type string (alphanumeric, underscore, hyphen only).
 * Returns null on invalid input to prevent path-traversal attacks.
 */
function sanitize_model_type(string $input): ?string
{
    return preg_match('/^[a-zA-Z0-9\() -]+$/', $input) ? $input : null;
}

/**
 * Validate and return a timestamp string (digits only).
 */
function sanitize_timestamp(string $input): ?string
{
    return preg_match('/^\d+$/', $input) ? $input : null;
}

/**
 * Sanitize a username for use as a filename component.
 * Replaces any character outside [a-zA-Z0-9_- ] with an underscore,
 * collapses consecutive underscores, and trims leading/trailing underscores/spaces.
 * Returns null if the result is empty (e.g. input was all special characters).
 */
function sanitize_username_for_filename(string $input): ?string
{
    $sanitized = (string) preg_replace('/[^a-zA-Z0-9_\- ]/', '_', $input);
    $sanitized = (string) preg_replace('/_+/', '_', $sanitized);
    $sanitized = trim($sanitized, '_ ');
    return $sanitized !== '' ? $sanitized : null;
}

/**
 * Validate and return an upload filename.
 * Expected format: {digits}_{alphanumeric/underscore/hyphen/space}.model
 * Use sanitize_username_for_filename() before building the name.
 */
function sanitize_upload_filename(string $input): ?string
{
    return preg_match('/^\d+_[a-zA-Z0-9_\() -]+\.model$/', $input) ? $input : null;
}
