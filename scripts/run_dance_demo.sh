#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: run_dance_demo.sh [--headless] [--save PATH] [--xml PATH]' \
        '' \
        'Run the bundled local ONNX dance example in CPU MuJoCo.' \
        'The interactive viewer opens by default and stays open at the end.' \
        '' \
        '  --headless   Run without a viewer or a display.' \
        '  --save PATH  Save the inferred trajectory (parent directory must exist).' \
        '  --xml PATH   Use a G1 XML instead of locating it in mjlab.' \
        '  -h, --help   Show this help without loading dependencies.' \
        '' \
        'Requires Bash, uv, and local assets at the paths listed in README.' \
        'Relative --save and --xml paths are relative to your current directory.'
}

fail() {
    printf 'Dance demo: %s\n' "$*" >&2
    exit 2
}

viewer=(--viewer)
forward=()
while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --headless)
            viewer=()
            shift
            ;;
        --save|--xml)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "$1 requires a path."
            forward+=("$1" "$2")
            shift 2
            ;;
        *)
            fail "Unknown argument: $1. Use --help."
            ;;
    esac
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
policy="$repo/checkpoints/onnx/dance/dance_ckpt20000.onnx"
metadata="$repo/checkpoints/onnx/dance/dance_ckpt20000.json"
motion="$repo/data/motions/dance/zhongzhuangyuan1_umr_50hz.npz"
entry="$repo/scripts/infer_onnx_minimal.py"

for path in "$policy" "$metadata" "$motion" "$entry"; do
    [[ -r "$path" && -s "$path" ]] || fail "Missing or empty file: $path. See README for asset placement."
done
command -v uv >/dev/null 2>&1 || fail 'uv is not installed or is not on PATH.'
if ((${#viewer[@]})) && [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    fail 'No graphical display detected. Use --headless or run from a desktop session.'
fi

printf 'Policy: %s\nMotion: %s\n' "$policy" "$motion"
# Keep the temporary runtime alive for the entire viewer process; do not edit uv.lock.
exec uv run --project "$repo" --frozen --with onnxruntime==1.31.1 \
    python -B "$entry" "$policy" "$motion" "${viewer[@]}" "${forward[@]}"
