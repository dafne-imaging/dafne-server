<?php
declare(strict_types=1);

/**
 * Append a timestamped row to the server_log MySQL table.
 * If $print is true, also forward the message to the web-server error log.
 * Falls back to the error log if the database is unavailable.
 */
function server_log(string $text, bool $print = false): void
{
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