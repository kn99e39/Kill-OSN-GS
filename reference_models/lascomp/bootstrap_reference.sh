#!/usr/bin/env bash
# Deterministically obtain the exact upstream source; does not install or run it.
set -euo pipefail

UPSTREAM_URL="https://github.com/DavidYan2001/LaS-Comp.git"
UPSTREAM_SHA="0da46f2569b484986d3737d97213428db962161f"
TARGET_DIR="${1:-$(pwd)/reference_models/lascomp/upstream}"

if [ -e "$TARGET_DIR" ] && [ ! -d "$TARGET_DIR/.git" ]; then
  echo "Refusing non-repository target: $TARGET_DIR" >&2
  exit 2
fi
if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone "$UPSTREAM_URL" "$TARGET_DIR"
fi

git -C "$TARGET_DIR" remote set-url origin "$UPSTREAM_URL"
git -C "$TARGET_DIR" fetch --no-tags origin "$UPSTREAM_SHA"
git -C "$TARGET_DIR" checkout --detach "$UPSTREAM_SHA"

actual="$(git -C "$TARGET_DIR" rev-parse HEAD)"
[ "$actual" = "$UPSTREAM_SHA" ] || { echo "wrong upstream commit: $actual" >&2; exit 3; }
[ -z "$(git -C "$TARGET_DIR" status --porcelain)" ] || { echo "upstream working tree is not clean" >&2; exit 4; }

verify_blob() {
  local expected="$1"
  local path="$2"
  local actual_hash
  actual_hash="$(git -C "$TARGET_DIR" show "$UPSTREAM_SHA:$path" | sha256sum | awk '{print $1}')"
  [ "$actual_hash" = "$expected" ] || { echo "blob fingerprint mismatch for $path: $actual_hash" >&2; exit 5; }
}

# Hash Git blobs, not a platform working-tree checkout; this avoids CRLF differences.
verify_blob "fbcad84708cf54d6c67f7dd92c85c56c9e1b13e035ef360e27ab75fba3794dff" "environment.yml"
verify_blob "b61e85f97fdb60b2a74ce65498089423e9bcb6798f87d054979bc3519e8329e5" "requirements.txt"
verify_blob "18724a629514b405588d41d9d86c52cd84cf1deead1a5d5a4195846d0764a7b1" "run_lascomp_text_condition_single.py"
verify_blob "a91087e2374d646ed5ddbf7504e0212a59c91432eb1ad854fd7c5e3b91581d67" "trellis/pipelines/samplers/flow_euler.py"

echo "LaS-Comp source locked: $UPSTREAM_SHA"
