#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# merge_client.py – Federated learning merge client for the Dafne PHP server.
#
# Downloads the oldest pending upload for a given model type, validates it,
# merges it with the current canonical model, and publishes the result back.
# Designed to be called repeatedly (e.g. via run_merge.sh) until the queue
# is empty.
#
# Exit codes:
#   0  – merge completed successfully
#   1  – error (download failure, validation failure, upload failure, …)
#   2  – no pending uploads; queue is empty (use as loop-termination sentinel)
#
# Typical cron usage: run_merge.sh handles the loop; this script handles one
# upload per invocation.
#
# Copyright (c) 2026 Dafne-Imaging Team – GPLv3

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import requests

# Reuse evaluation logic from utils.py.
# evaluate_model(path_or_dir, model, save_log, comment, cleanup) → float
# Passing a directory as first argument makes it glob *.npz from there directly,
# bypassing the server-side TEST_DATA_DIR constant.
try:
    from utils import evaluate_model
except ImportError as exc:
    print(f"ERROR: cannot import from utils.py: {exc}", file=sys.stderr)
    print("Ensure utils.py is on the path and its imports are satisfied.", file=sys.stderr)
    sys.exit(1)

from dafne_dl.model_loaders import generic_load_model

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_OK             = 0
EXIT_ERROR          = 1
EXIT_NOTHING_TO_DO  = 2   # caller (run_merge.sh) uses this as the stop signal

# ---------------------------------------------------------------------------
# Defaults (mirror db/server_config.json)
# ---------------------------------------------------------------------------
DEFAULT_DICE_THRESHOLD  = -0.001
DEFAULT_ORIGINAL_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _check_download(path: Path, server_hash: str | None) -> None:
    """
    Verify the downloaded file against the server-supplied SHA-256 digest.
    On mismatch the local file is deleted before raising, so the caller
    always exits with a clean state.
    """
    if server_hash is None:
        return
    local = sha256_file(path)
    if local != server_hash:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}:\n"
            f"  server: {server_hash}\n"
            f"  local:  {local}"
        )


# ---------------------------------------------------------------------------
# Merge log  – tracks uploads that have already been successfully merged so
# that stale server-side entries are cleaned up without being re-processed.
# File: merge_cache/<model_type>/merged_uploads.txt  (one filename per line)
# ---------------------------------------------------------------------------

MERGED_LOG_FILE = 'merged_uploads.txt'


def load_merged_log(log_path: Path) -> set[str]:
    """Return the set of upload filenames recorded as already merged."""
    if not log_path.exists():
        return set()
    return set(log_path.read_text().splitlines())


def record_merged(log_path: Path, filename: str) -> None:
    """Append *filename* to the merge log."""
    with log_path.open('a') as fh:
        fh.write(filename + '\n')


# ---------------------------------------------------------------------------
# Server API
# ---------------------------------------------------------------------------

def _post_json(server: str, route: str, payload: dict, timeout: int = 30) -> dict:
    r = requests.post(f'{server}{route}', json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_info(server: str, api_key: str, model_type: str) -> dict:
    return _post_json(server, '/info_model',
                      {'api_key': api_key, 'model_type': model_type})


def api_get_pending_uploads(server: str, api_key: str, model_type: str) -> list[dict]:
    data = _post_json(server, '/get_pending_uploads',
                      {'api_key': api_key, 'model_type': model_type})
    return data.get('uploads', [])


def api_download_file(server: str, route: str, payload: dict, dest: Path) -> None:
    """Download a binary endpoint into *dest*, then verify the SHA-256 header."""
    r = requests.post(f'{server}{route}', json=payload, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    _check_download(dest, r.headers.get('X-SHA256-Checksum'))


def api_delete_upload(server: str, api_key: str, model_type: str,
                      filename: str) -> None:
    _post_json(server, '/delete_upload',
               {'api_key': api_key, 'model_type': model_type, 'filename': filename})


UPLOAD_CHUNK_SIZE = 50 * 1024 * 1024   # 50 MiB per chunk


def api_upload_merged(server: str, api_key: str, model_type: str,
                      path: Path) -> int:
    """Upload merged model with SHA-256 checksum; return server-assigned timestamp.

    Files larger than UPLOAD_CHUNK_SIZE are sent in multiple chunks so that
    each individual HTTP request stays within server upload-size limits.
    """
    sha256       = sha256_file(path)
    file_size    = path.stat().st_size
    total_chunks = max(1, -(-file_size // UPLOAD_CHUNK_SIZE))  # ceiling division
    filename     = path.name

    with path.open('rb') as fh:
        for chunk_index in range(total_chunks):
            chunk_data = fh.read(UPLOAD_CHUNK_SIZE)
            r = requests.post(
                f'{server}/upload_merged_model',
                data={
                    'api_key':      api_key,
                    'model_type':   model_type,
                    'sha256':       sha256,
                    'chunk_index':  chunk_index,
                    'total_chunks': total_chunks,
                    'filename':     filename,
                },
                files={'model_binary': (filename, chunk_data, 'application/octet-stream')},
                timeout=300,
            )
            r.raise_for_status()
            if chunk_index < total_chunks - 1:
                print(f"    chunk {chunk_index + 1}/{total_chunks} uploaded …")

    return int(r.json().get('timestamp', 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dafne merge client – processes ONE pending upload per run. "
            "Exit code 2 means the queue is empty. "
            "Use run_merge.sh to drain the queue automatically."
        )
    )
    parser.add_argument('--server',          '-s', required=True,
                        help='Server base URL, e.g. http://my-server.example.com')
    parser.add_argument('--model-type',      '-m', required=True,
                        help='Model type to process, e.g. Thigh')
    parser.add_argument('--validation-data', '-v', required=True,
                        help='Directory containing .npz validation files')
    parser.add_argument('--api-key',         '-k', required=True,
                        help='Merge-client API key (from db/merge_api_keys.txt)')
    parser.add_argument('--cache-dir',       '-c', default='merge_cache',
                        help='Local directory for downloaded/processed models '
                             '(default: ./merge_cache)')
    parser.add_argument('--dice-threshold',  '-t', type=float,
                        default=DEFAULT_DICE_THRESHOLD,
                        help=f'Minimum Dice score to accept a merge '
                             f'(default: {DEFAULT_DICE_THRESHOLD})')
    parser.add_argument('--original-weight', '-w', type=float,
                        default=DEFAULT_ORIGINAL_WEIGHT,
                        help=f'Weight of the canonical model in the weighted average '
                             f'(default: {DEFAULT_ORIGINAL_WEIGHT})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and merge locally; skip server upload/deletion')
    args = parser.parse_args()

    server      = args.server.rstrip('/')
    model_type  = args.model_type
    api_key     = args.api_key
    val_dir     = Path(args.validation_data)
    dice_thr    = args.dice_threshold
    orig_weight = args.original_weight

    if not val_dir.is_dir():
        print(f"ERROR: validation data directory not found: {val_dir}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    # Cache layout: merge_cache/<model_type>/{canonical,uploads,merged}/
    cache_root    = Path(args.cache_dir) / model_type
    canonical_dir = cache_root / 'canonical'
    uploads_dir   = cache_root / 'uploads'
    merged_dir    = cache_root / 'merged'
    for d in (canonical_dir, uploads_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    merged_log = cache_root / MERGED_LOG_FILE

    # ------------------------------------------------------------------
    # 1. Check for pending uploads – exit immediately if queue is empty
    # ------------------------------------------------------------------
    print(f"Checking pending uploads for '{model_type}' …")
    try:
        pending = api_get_pending_uploads(server, api_key, model_type)
    except Exception as exc:
        print(f"ERROR: could not fetch pending uploads: {exc}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    if not pending:
        print("Queue is empty – nothing to do.")
        sys.exit(EXIT_NOTHING_TO_DO)

    # ------------------------------------------------------------------
    # 2. Skip uploads that are already recorded in the local merge log.
    #    They linger on the server only because a previous deletion failed;
    #    delete them now and move on to the first unprocessed entry.
    # ------------------------------------------------------------------
    already_merged = load_merged_log(merged_log)
    entry = None
    for candidate in pending:
        if candidate['filename'] in already_merged:
            print(f"  {candidate['filename']} already merged – deleting from server.")
            try:
                api_delete_upload(server, api_key, model_type, candidate['filename'])
            except Exception as exc:
                print(f"  WARNING: could not delete {candidate['filename']}: {exc}")
        else:
            entry = candidate
            break   # found the oldest unprocessed upload

    if entry is None:
        print("All pending uploads have already been merged. Queue is empty.")
        sys.exit(EXIT_NOTHING_TO_DO)

    filename = entry['filename']
    username = entry.get('username', '?')
    print(f"Processing oldest unmerged upload: {filename}  (from {username})")

    # ------------------------------------------------------------------
    # 3. Download the upload (keep local copy even after server deletion)
    # ------------------------------------------------------------------
    upload_path = uploads_dir / filename
    if not upload_path.exists():
        print("  Downloading upload …")
        try:
            api_download_file(server, '/download_upload',
                              {'api_key': api_key, 'model_type': model_type,
                               'filename': filename},
                              upload_path)
        except Exception as exc:
            print(f"ERROR: download failed: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
    else:
        print("  Using cached local copy.")

    # ------------------------------------------------------------------
    # 4. Fetch info and download the current canonical model if not cached
    # ------------------------------------------------------------------
    try:
        info      = api_info(server, api_key, model_type)
        latest_ts = info['latest_timestamp']
    except Exception as exc:
        print(f"ERROR: could not fetch model info: {exc}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    canonical_path = canonical_dir / f'{latest_ts}.model'
    if not canonical_path.exists():
        print(f"  Downloading canonical model {latest_ts} …")
        try:
            api_download_file(server, '/get_model',
                              {'api_key': api_key, 'model_type': model_type,
                               'timestamp': latest_ts},
                              canonical_path)
        except Exception as exc:
            print(f"ERROR: canonical model download failed: {exc}", file=sys.stderr)
            sys.exit(EXIT_ERROR)
    else:
        print(f"  Using cached canonical model {latest_ts}.model")

    # ------------------------------------------------------------------
    # 5. Load both models
    # ------------------------------------------------------------------
    print("  Loading models …")
    try:
        canonical_model = generic_load_model(open(canonical_path, 'rb'))
        upload_model    = generic_load_model(open(upload_path, 'rb'))
    except Exception as exc:
        print(f"ERROR: failed to load model: {exc}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    if canonical_model.model_id != upload_model.model_id:
        print(
            f"ERROR: model ID mismatch – cannot merge\n"
            f"  canonical: {canonical_model.model_id}\n"
            f"  upload:    {upload_model.model_id}",
            file=sys.stderr,
        )
        sys.exit(EXIT_ERROR)

    # ------------------------------------------------------------------
    # 6. Pre-merge validation of the uploaded model
    # ------------------------------------------------------------------
    print("  Validating uploaded model …")
    upload_score = evaluate_model(val_dir, upload_model, save_log=False, cleanup=False)
    print(f"  Upload Dice: {upload_score:.6f}  (threshold: {dice_thr})")
    if upload_score < dice_thr:
        print("  Upload score below threshold – merge rejected.")
        sys.exit(EXIT_ERROR)

    # ------------------------------------------------------------------
    # 7. Merge  (mirrors utils.merge_model weighted average)
    # ------------------------------------------------------------------
    print(f"  Merging  (canonical weight={orig_weight}) …")
    merged_model = canonical_model * orig_weight + upload_model * (1 - orig_weight)
    merged_model.reset_timestamp()
    del upload_model
    gc.collect()

    # ------------------------------------------------------------------
    # 8. Post-merge validation
    # ------------------------------------------------------------------
    print("  Validating merged model …")
    merged_score = evaluate_model(val_dir, merged_model, save_log=False, cleanup=True)
    print(f"  Merged Dice: {merged_score:.6f}")
    if merged_score < dice_thr:
        print("  Merged model score below threshold – merge rejected.")
        sys.exit(EXIT_ERROR)

    # ------------------------------------------------------------------
    # 9. Save merged model locally (atomic write via .tmp)
    # ------------------------------------------------------------------
    merged_path = merged_dir / f'{merged_model.timestamp_id}.model'
    tmp_path    = merged_path.with_suffix('.tmp')
    print(f"  Saving merged model → {merged_path} …")
    with tmp_path.open('wb') as fh:
        merged_model.dump(fh)
    tmp_path.rename(merged_path)

    del canonical_model, merged_model
    gc.collect()

    if args.dry_run:
        print("[dry-run] Skipping server upload and upload deletion.")
        sys.exit(EXIT_OK)

    # ------------------------------------------------------------------
    # 10. Upload merged model to server
    # ------------------------------------------------------------------
    print("  Uploading merged model to server …")
    try:
        new_ts = api_upload_merged(server, api_key, model_type, merged_path)
        print(f"  Server accepted merged model (timestamp: {new_ts}).")
    except Exception as exc:
        print(f"ERROR: upload failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    # ------------------------------------------------------------------
    # 11. Record the merge and delete the upload from the server.
    #     The log is written first: even if the server deletion fails,
    #     the next run will recognise the file as already merged and
    #     retry the deletion without re-processing it.
    # ------------------------------------------------------------------
    record_merged(merged_log, filename)

    print(f"  Deleting {filename} from server …")
    try:
        api_delete_upload(server, api_key, model_type, filename)
    except Exception as exc:
        print(f"WARNING: could not delete {filename} from server: {exc}")

    print("Done.")
    sys.exit(EXIT_OK)


if __name__ == '__main__':
    main()
