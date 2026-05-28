FROM vllm/vllm-openai:v0.18.0

ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

COPY vllm/config/weight_sharing.py         /tmp/ws/config/weight_sharing.py
COPY vllm/config/vllm.py                   /tmp/ws/config/vllm.py
COPY vllm/config/__init__.py               /tmp/ws/config/__init__.py
COPY vllm/engine/arg_utils.py              /tmp/ws/engine/arg_utils.py
COPY vllm/distributed/weight_sharing/      /tmp/ws/distributed/weight_sharing/
COPY vllm/model_executor/model_loader/base_loader.py    /tmp/ws/model_executor/model_loader/base_loader.py
COPY vllm/model_executor/model_loader/default_loader.py /tmp/ws/model_executor/model_loader/default_loader.py
COPY vllm/model_executor/model_loader/utils.py          /tmp/ws/model_executor/model_loader/utils.py
COPY vllm/v1/worker/gpu_model_runner.py    /tmp/ws/v1/worker/gpu_model_runner.py
COPY kvcached/                             /opt/kvcached/
COPY sardeenz/                             /opt/sardeenz/

RUN SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))") && \
    mkdir -p "$SITE/distributed/weight_sharing" && \
    cp /tmp/ws/config/weight_sharing.py "$SITE/config/weight_sharing.py" && \
    cp /tmp/ws/config/vllm.py "$SITE/config/vllm.py" && \
    cp /tmp/ws/config/__init__.py "$SITE/config/__init__.py" && \
    cp /tmp/ws/engine/arg_utils.py "$SITE/engine/arg_utils.py" && \
    cp /tmp/ws/model_executor/model_loader/base_loader.py "$SITE/model_executor/model_loader/base_loader.py" && \
    cp /tmp/ws/model_executor/model_loader/default_loader.py "$SITE/model_executor/model_loader/default_loader.py" && \
    cp /tmp/ws/model_executor/model_loader/utils.py "$SITE/model_executor/model_loader/utils.py" && \
    cp /tmp/ws/v1/worker/gpu_model_runner.py "$SITE/v1/worker/gpu_model_runner.py" && \
    cp /tmp/ws/distributed/weight_sharing/*.py "$SITE/distributed/weight_sharing/" && \
    rm -rf /tmp/ws

# kvcached is copied as source because its CUDA extension must be built against
# the final image's torch/CUDA toolchain. Enable this once the base image is
# known to include the matching CUDA compiler/runtime headers:
# RUN python3 -m pip install --no-build-isolation -e /opt/kvcached

RUN python3 -c "from vllm.distributed.weight_sharing.parameter_filter import is_shareable_parameter; from vllm.config.weight_sharing import WeightSharingConfig; assert is_shareable_parameter('model.layers.0.qkv.weight') == True; assert is_shareable_parameter('model.embed_tokens.weight') == False; print('Weight sharing verified OK')"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
