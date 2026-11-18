#!/bin/sh
set -eu
usage() { printf '%s\n' 'Usage: examples/nano-household/run.sh [--prepare] NEW_OUTPUT_DIRECTORY'; }
if [ "${1:-}" = --help ]; then usage; exit 0; fi
prepare=false
if [ "${1:-}" = --prepare ]; then prepare=true; shift; fi
[ "$#" -eq 1 ] || { usage >&2; exit 2; }
: "${AISLE_NANO_MODEL:?set to the Studio Nano URDF with its meshes}"
: "${AISLE_NANO_SIMULATOR:?set to dorobot-nano-runtime with aisle-node}"
require_executable() {
    [ -f "$1" ] && [ -x "$1" ] || { printf 'Missing executable: %s\n' "$1" >&2; exit 2; }
}
absolute() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
require_executable "$AISLE_NANO_SIMULATOR"
if [ "$prepare" = false ]; then
    : "${AISLE_DORA:?set to the Dora 1.0.1 executable}"
    require_executable "$AISLE_DORA"
    AISLE_DORA=$(absolute "$AISLE_DORA")
    case "$("$AISLE_DORA" --version)" in *' 1.0.1') ;; *) echo 'Dora 1.0.1 is required' >&2; exit 2 ;; esac
    if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        # An exported empty key would stop the planner from reading the profile.
        unset DEEPSEEK_API_KEY
        [ -r "${AISLE_OCTOS_PROFILE:-}" ] || { echo 'Set DEEPSEEK_API_KEY or AISLE_OCTOS_PROFILE' >&2; exit 2; }
    fi
    if [ -n "${AISLE_OCTOS_PROFILE:-}" ]; then export AISLE_OCTOS_PROFILE="$(absolute "$AISLE_OCTOS_PROFILE")"; fi
    if [ -n "${AISLE_STUDIO_JIT:-}" ]; then require_executable "$AISLE_STUDIO_JIT"; fi
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
output=$(absolute "$1")
set -- --output "$output" --model "$(absolute "$AISLE_NANO_MODEL")" --simulator "$(absolute "$AISLE_NANO_SIMULATOR")"
if [ -n "${AISLE_NANO_HELPER:-}" ]; then
    require_executable "$AISLE_NANO_HELPER"
    set -- "$@" --helper "$(absolute "$AISLE_NANO_HELPER")"
fi
if [ -n "${AISLE_HOUSEHOLD_START_AT:-}" ]; then set -- "$@" --start-at "$AISLE_HOUSEHOLD_START_AT"; fi
"$root/tools/aisle-jit" nano run household "$@"
if [ "$prepare" = true ]; then exit 0; fi
output=$(CDPATH= cd -- "$output" && pwd -P)
printf 'Studio live socket: %s/preview.sock\n' "$output" >&2
if [ -n "${AISLE_STUDIO_JIT:-}" ]; then
    set -- --live-socket "$output/preview.sock" --screen live
    if [ "${AISLE_STUDIO_VIDEO:-0}" = 1 ]; then set -- --video "$@"; fi
    DOROBOT_STUDIO_STATE_DIR="$output/studio-state" "$AISLE_STUDIO_JIT" "$@" >"$output/studio.log" 2>&1 &
    studio=$!
fi
# A failed run must not leave the viewer running. It stays open after a completed one.
(cd "$output" && "$AISLE_DORA" run dataflow.yaml) || {
    status=$?
    if [ -n "${studio:-}" ]; then kill "$studio" 2>/dev/null || true; fi
    printf 'Dora exited with status %s; the recording was not audited.\n' "$status" >&2
    exit "$status"
}
exec "$root/tools/aisle-jit" nano run audit-household --label-replay --output "$output"
