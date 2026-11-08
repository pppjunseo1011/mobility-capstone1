# Reuse the already validated local image (Python 3.9 + ffmpeg/Pillow).
# No dependencies are installed on the host, and no GPU is needed.
ARG ADAPTER_BASE=scout-mtmc/posetrack-all:ba727e6
FROM ${ADAPTER_BASE}
LABEL project=scout-mtmc org.opencontainers.image.title="SCOUT first-pass data adapter"
ENV PYTHONPATH=/opt/scout-adapter PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /opt/scout-adapter
COPY adapters/scout_adapter /opt/scout-adapter/scout_adapter
COPY tests /opt/scout-adapter/tests
ENTRYPOINT ["python", "-m", "scout_adapter"]
CMD ["--help"]
