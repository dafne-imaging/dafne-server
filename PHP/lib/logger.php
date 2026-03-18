<?php
declare(strict_types=1);

/**
 * Sanitise a log message before storage.
 * - Removes null bytes and ASCII control characters (keeps printable + UTF-8).
 * - Collapses runs of whitespace to a single space and trims.
 * - Truncates to 500 characters so a crafted payload cannot bloat the DB.
 */
function sanitize_log_message(string $text): string
{
    // Strip null bytes and ASCII control characters (0x00–0x1F, 0x7F),
    // but keep horizontal tab (0x09) for readability.
    $text = preg_replace('/[\x00-\x08\x0A-\x1F\x7F]/', '', $text) ?? '';
    // Collapse whitespace and trim
    $text = trim(preg_replace('/\s+/', ' ', $text) ?? '');
    // Hard length cap
    if (mb_strlen($text) > 500) {
        $text = mb_substr($text, 0, 500) . ' [truncated]';
    }
    return $text;
}

/**
 * Append a timestamped row to the server_log MySQL table.
 * If $print is true, also forward the message to the web-server error log.
 * Falls back to the error log if the database is unavailable.
 */
function server_log(string $text, bool $print = false): void
{
    $text = sanitize_log_message($text);

    if ($print) {
        error_log('[dafne] ' . $text);
    }
    try {
        $s = get_db()->prepare('INSERT INTO server_log (message) VALUES (?)');
        $s->execute([$text]);
    } catch (Throwable $e) {
        error_log('[dafne] DB log write failed: ' . $e->getMessage());
        error_log('[dafne] ' . $text);
    }
}