<?php
declare(strict_types=1);

/**
 * Check whether $api_key belongs to a standard (user) credential.
 */
function valid_credentials(string $api_key): bool
{
    return _lookup_key(API_KEYS_FILE, $api_key) !== null;
}

/**
 * Return the username associated with $api_key, or null if not found.
 */
function get_username(string $api_key): ?string
{
    return _lookup_key(API_KEYS_FILE, $api_key);
}

/**
 * Check whether $api_key belongs to a merge-client credential.
 * Merge client keys live in a separate file (db/merge_api_keys.txt) so
 * they can never be confused with standard user keys.
 */
function valid_merge_credentials(string $api_key): bool
{
    return _lookup_key(MERGE_API_KEYS_FILE, $api_key) !== null;
}

/**
 * Return the username associated with a merge-client $api_key, or null if not found.
 */
function get_merge_username(string $api_key): ?string
{
    return _lookup_key(MERGE_API_KEYS_FILE, $api_key);
}

/**
 * Internal helper: scan a key file for a matching key and return the username.
 * File format: one entry per line, "username:api_key".
 */
function _lookup_key(string $file, string $api_key): ?string
{
    if (!is_file($file)) {
        return null;
    }
    $lines = file($file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if ($lines === false) {
        return null;
    }
    foreach ($lines as $line) {
        $parts = explode(':', trim($line), 2);
        if (count($parts) === 2 && $parts[1] === $api_key) {
            return $parts[0];
        }
    }
    return null;
}
