<?php
declare(strict_types=1);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Hash an API key for comparison against the database.
 * Keys are stored as unsalted SHA-256 hex digests.
 */
function hash_api_key(string $api_key): string
{
    return hash('sha256', $api_key);
}

/**
 * Fetch the user row for the given raw API key, or null if not found.
 * Result is cached for the lifetime of the current request to avoid
 * redundant DB round-trips within a single handler call chain.
 *
 * Returned keys: id (int), name (string), email (string), is_admin (bool).
 */
function get_user_by_key(string $api_key): ?array
{
    static $cache = [];

    if ($api_key === '') {
        return null;
    }

    if (!array_key_exists($api_key, $cache)) {
        $stmt = get_db()->prepare(
            'SELECT id, name, email, is_admin FROM users WHERE api_key_hash = ?'
        );
        $stmt->execute([hash_api_key($api_key)]);
        $row = $stmt->fetch();
        $cache[$api_key] = $row !== false ? $row : null;
    }

    return $cache[$api_key];
}

// ---------------------------------------------------------------------------
// Basic credential checks
// ---------------------------------------------------------------------------

/**
 * Return true if the API key belongs to a registered user.
 */
function valid_credentials(string $api_key): bool
{
    return get_user_by_key($api_key) !== null;
}

/**
 * Return the display name for the given API key, or null if not found.
 */
function get_username(string $api_key): ?string
{
    $user = get_user_by_key($api_key);
    return $user !== null ? $user['name'] : null;
}

// ---------------------------------------------------------------------------
// Per-model access permissions
// ---------------------------------------------------------------------------

/**
 * Return true if the user may access (read/upload) the given model type.
 * Access is granted by a row in users_accesspermissions.
 */
function can_access_model(string $api_key, string $model_type): bool
{
    $user = get_user_by_key($api_key);
    if ($user === null) {
        return false;
    }
    $stmt = get_db()->prepare(
        'SELECT 1 FROM users_accesspermissions WHERE user_id = ? AND model_type = ?'
    );
    $stmt->execute([$user['id'], $model_type]);
    return $stmt->fetch() !== false;
}

/**
 * Return the list of model types the user is allowed to access.
 */
function get_accessible_models(string $api_key): array
{
    $user = get_user_by_key($api_key);
    if ($user === null) {
        return [];
    }
    $stmt = get_db()->prepare(
        'SELECT model_type FROM users_accesspermissions WHERE user_id = ?'
    );
    $stmt->execute([$user['id']]);
    return $stmt->fetchAll(PDO::FETCH_COLUMN);
}

// ---------------------------------------------------------------------------
// Per-model merge permissions
// ---------------------------------------------------------------------------

/**
 * Return true if the user may merge (download uploads / push merged models)
 * for the given model type.
 * Merge access is granted by a row in users_mergepermissions.
 */
function can_merge_model(string $api_key, string $model_type): bool
{
    $user = get_user_by_key($api_key);
    if ($user === null) {
        return false;
    }
    $stmt = get_db()->prepare(
        'SELECT 1 FROM users_mergepermissions WHERE user_id = ? AND model_type = ?'
    );
    $stmt->execute([$user['id'], $model_type]);
    return $stmt->fetch() !== false;
}

/**
 * Return the display name for a merge-client API key (same as get_username).
 * Kept for call-site symmetry with the previous file-based implementation.
 */
function get_merge_username(string $api_key): ?string
{
    return get_username($api_key);
}

// ---------------------------------------------------------------------------
// Administration
// ---------------------------------------------------------------------------

/**
 * Return true if the user has the administration permission.
 * Admins may change permissions for other users.
 */
function is_admin(string $api_key): bool
{
    $user = get_user_by_key($api_key);
    return $user !== null && (bool) $user['is_admin'];
}
