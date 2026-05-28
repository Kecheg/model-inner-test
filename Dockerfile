FROM vllm/vllm-openai:v0.18.0

ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NODE_ENV=production \
    PORT=3000 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info \
    HF_HUB_OFFLINE=0 \
    HOME=/opt/app-root/src \
    XDG_CACHE_HOME=/opt/app-root/src/.cache \
    HF_HOME=/opt/app-root/src/.cache/huggingface \
    FLASHINFER_WORKSPACE_DIR=/opt/app-root/src/.cache/flashinfer

USER root

COPY node-v22.16.0-linux-x64/ /usr/local/
COPY cuda-stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so

COPY vllm/config/weight_sharing.py /tmp/ws/config/weight_sharing.py
COPY vllm/config/vllm.py /tmp/ws/config/vllm.py
COPY vllm/config/__init__.py /tmp/ws/config/__init__.py
COPY vllm/engine/arg_utils.py /tmp/ws/engine/arg_utils.py
COPY vllm/distributed/weight_sharing/ /tmp/ws/distributed/weight_sharing/
COPY vllm/model_executor/model_loader/base_loader.py /tmp/ws/model_executor/model_loader/base_loader.py
COPY vllm/model_executor/model_loader/default_loader.py /tmp/ws/model_executor/model_loader/default_loader.py
COPY vllm/model_executor/model_loader/utils.py /tmp/ws/model_executor/model_loader/utils.py
COPY vllm/v1/worker/gpu_model_runner.py /tmp/ws/v1/worker/gpu_model_runner.py
COPY kvcached/ /opt/kvcached/

RUN SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))") \
    && mkdir -p "$SITE/distributed/weight_sharing" \
    && cp /tmp/ws/config/weight_sharing.py "$SITE/config/weight_sharing.py" \
    && cp /tmp/ws/config/vllm.py "$SITE/config/vllm.py" \
    && cp /tmp/ws/config/__init__.py "$SITE/config/__init__.py" \
    && cp /tmp/ws/engine/arg_utils.py "$SITE/engine/arg_utils.py" \
    && cp /tmp/ws/model_executor/model_loader/base_loader.py "$SITE/model_executor/model_loader/base_loader.py" \
    && cp /tmp/ws/model_executor/model_loader/default_loader.py "$SITE/model_executor/model_loader/default_loader.py" \
    && cp /tmp/ws/model_executor/model_loader/utils.py "$SITE/model_executor/model_loader/utils.py" \
    && cp /tmp/ws/v1/worker/gpu_model_runner.py "$SITE/v1/worker/gpu_model_runner.py" \
    && cp /tmp/ws/distributed/weight_sharing/*.py "$SITE/distributed/weight_sharing/" \
    && rm -rf /tmp/ws \
    && python3 -m pip install --progress-bar off --no-cache-dir "wrapt>=1.15" "posix_ipc>=1.0" \
    && python3 -c "from pathlib import Path; p=Path('/opt/kvcached/setup.py'); s=p.read_text(); s=s.replace('{\"build_ext\": BuildExtension}', '{\"build_ext\": BuildExtension.with_options(use_ninja=False)}'); p.write_text(s)" \
    && LIBRARY_PATH=/usr/local/cuda/lib64/stubs USE_NINJA=0 MAX_JOBS=1 CUDA_HOME=/usr/local/cuda python3 -m pip install --progress-bar off --no-cache-dir --no-deps --no-build-isolation --force-reinstall /opt/kvcached \
    && python3 -m py_compile \
        "$SITE/distributed/weight_sharing/manager.py" \
        "$SITE/distributed/weight_sharing/parameter_filter.py" \
        /usr/local/lib/python3.12/dist-packages/kvcached/kv_cache_manager.py \
        /usr/local/lib/python3.12/dist-packages/kvcached/integration/vllm/patches.py \
    && python3 -c "from vllm.distributed.weight_sharing.parameter_filter import is_shareable_parameter; from vllm.config.weight_sharing import WeightSharingConfig; assert is_shareable_parameter('model.layers.0.qkv.weight') is True; assert is_shareable_parameter('model.embed_tokens.weight') is False; print('Weight sharing verified OK')"

# Sardeenz must be pre-built in the build context because this environment's
# docker build containers cannot start Node worker threads reliably.
COPY sardeenz/ /app/
COPY sardeenz/ /opt/sardeenz/

RUN test -f /app/apps/backend/dist/server.js \
    && test -f /app/apps/frontend/dist/index.html \
    && sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /opt/app-root/src/.cache/flashinfer /opt/app-root/src/.cache/vllm

ENV ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    KVCACHED_KVCTL_BIN=/usr/local/bin/kvctl \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /app
EXPOSE 3000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
