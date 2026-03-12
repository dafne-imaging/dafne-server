<?php
declare(strict_types=1);

/**
 * Append a timestamped line to db/models/log.txt.
 * If $print is true, also forward the message to the web-server error log.
 */
function server_log(string $text, bool $print = false): void
{
    if ($print) {
        error_log('[dafne] ' . $text);
    }
    $line     = date('Y-m-d H:i:s') . ' ' . $text . PHP_EOL;
    $log_file = MODELS_DIR . '/log.txt';
    file_put_contents($log_file, $line, FILE_APPEND | LOCK_EX);
}
