#!/usr/bin/env bash
# run_merge.sh – Drain the merge queue by calling merge_client.py in a loop.
#
# All arguments are forwarded to merge_client.py unchanged, so the flags are
# identical:
#   ./run_merge.sh -s http://server -m Thigh -v /data/val -k MY_KEY
#
# The loop stops when:
#   • merge_client.py exits with code 2 (queue empty)  → success
#   • merge_client.py exits with code 1 (error)        → propagate the error
#
# A safety cap (MAX_ITERATIONS) prevents an infinite loop in case the server
# keeps returning uploads faster than they are consumed.

set -euo pipefail

MAX_ITERATIONS=500
PYTHON="${PYTHON:-python3}"   # override with: PYTHON=/path/to/venv/bin/python ./run_merge.sh …

count=0

echo "[run_merge] Starting merge loop (max iterations: ${MAX_ITERATIONS})"

while [ "$count" -lt "$MAX_ITERATIONS" ]; do
    count=$(( count + 1 ))
    echo ""
    echo "[run_merge] Iteration ${count} / ${MAX_ITERATIONS}"

    set +e
    "$PYTHON" merge_client.py "$@"
    code=$?
    set -e

    case "$code" in
        0)
            # Merged one upload successfully; continue to the next.
            ;;
        2)
            # Queue is empty – normal termination.
            echo ""
            echo "[run_merge] Queue is empty after ${count} iteration(s). All done."
            exit 0
            ;;
        *)
            echo "[run_merge] merge_client.py exited with error code ${code}." >&2
            exit "$code"
            sleep 10  # wait a bit before retrying, in case of transient errors
            ;;
    esac
done

echo "[run_merge] WARNING: reached safety cap of ${MAX_ITERATIONS} iterations." >&2
exit 1
