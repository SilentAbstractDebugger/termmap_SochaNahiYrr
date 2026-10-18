#!/usr/bin/env bash
# Fails (exit 1) if git tracks or stages any data, model, cache or secret file.
# Use before every push:  bash scripts/check_no_data_tracked.sh
# Optional pre-commit hook: ln -s ../../scripts/check_no_data_tracked.sh .git/hooks/pre-commit
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PATTERN='\.(csv|tsv|xlsx|xls|pkl|pickle|npy|npz|safetensors|bin|pt|pth|onnx|faiss)$|(^|/)\.env$|(^|/)\.env\.[^e]|^(data|resources|models|outputs|logs)/|^reports/private/|^reports/diagnostics\.csv$'
BAD=$( { git ls-files; git diff --cached --name-only; } | sort -u | grep -E "$PATTERN" || true )
if [ -n "$BAD" ]; then
  echo "BLOCKED: these files must not be in git:"
  echo "$BAD" | sed 's/^/  - /'
  echo "Remove them with: git rm --cached <file>   (then commit)"
  exit 1
fi
# large files are suspicious too (weights / dumps)
BIG=$(git ls-files -z | xargs -0 -I{} sh -c 'test -f "{}" && s=$(wc -c < "{}") && [ "$s" -gt 5000000 ] && echo "{} ($s bytes)"' || true)
if [ -n "$BIG" ]; then echo "BLOCKED: files larger than 5 MB are tracked:"; echo "$BIG"; exit 1; fi
echo "OK: no data, model, cache or secret files are tracked."
