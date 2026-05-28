#!/bin/bash
# Build kvcached wheel + Docker image
# Usage: ./build.sh [--push] [--push-image]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="vllm-weight-share:full-sardeenz-kvcached-upstream"
TAG="vllm-weight-share:kvcached-v2"
DOCKERFILE="docker/Dockerfile.kvcached-v2"

PUSH_CODE=false
PUSH_IMAGE=false
for arg in "$@"; do
  case "$arg" in
    --push)      PUSH_CODE=true ;;
    --push-image) PUSH_IMAGE=true ;;
  esac
done

echo "=========================================="
echo " kvcached build script"
echo " base image: $BASE_IMAGE"
echo " target tag: $TAG"
echo "=========================================="

# ------------------------------------------------------------------
# Step 1: Build wheel (in container, avoids docker build seccomp limits)
# ------------------------------------------------------------------
echo ""
echo "[1/3] Building wheel..."
mkdir -p dist/

docker run --rm --entrypoint bash \
  -v "$SCRIPT_DIR:/tmp/kvcached" \
  "$BASE_IMAGE" \
  -c "
    set -e
    pip3 uninstall ninja --yes 2>/dev/null || true
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:\${LIBRARY_PATH}
    export MAX_JOBS=1
    pip3 wheel /tmp/kvcached/ --no-build-isolation --no-deps -w /tmp/kvcached/dist/
  " 2>&1 | grep -v "LD_PRELOAD\|ld.so"

WHEEL=$(ls -1 dist/kvcached-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
  echo "ERROR: wheel build failed"
  exit 1
fi
echo "  -> $WHEEL ($(du -h "$WHEEL" | cut -f1))"

# ------------------------------------------------------------------
# Step 2: Build Docker image (docker commit, more reliable in this env)
# ------------------------------------------------------------------
echo ""
echo "[2/3] Building Docker image..."
CID="kvcached-build-$$"

docker run -d --name "$CID" --entrypoint bash "$BASE_IMAGE" -c "sleep 300" 2>/dev/null
trap 'docker rm -f '"$CID"' 2>/dev/null' EXIT

docker cp "$WHEEL" "$CID:/tmp/" 2>/dev/null
docker exec "$CID" pip3 install "/tmp/$(basename "$WHEEL")" \
  --force-reinstall --no-deps --no-cache-dir 2>&1 | grep -v "LD_PRELOAD\|ld.so"

docker commit \
  -c 'ENV ENABLE_KVCACHED=true' \
  -c 'ENV KVCACHED_AUTOPATCH=1' \
  -c 'ENTRYPOINT []' \
  -c 'CMD ["bash"]' \
  "$CID" "$TAG" 2>/dev/null

docker rm -f "$CID" 2>/dev/null
trap - EXIT

echo "  -> $TAG ($(docker images --format '{{.Size}}' "$TAG"))"

# ------------------------------------------------------------------
# Step 3: Push (optional)
# ------------------------------------------------------------------
echo ""
echo "[3/3] Push..."

if $PUSH_CODE; then
  echo "  Pushing code to git..."
  git push
else
  echo "  Skipping code push (use --push)"
fi

if $PUSH_IMAGE; then
  echo "  Pushing image..."
  docker push "$TAG"
else
  echo "  Skipping image push (use --push-image)"
fi

echo ""
echo "=========================================="
echo " Build complete: $TAG"
echo "=========================================="
