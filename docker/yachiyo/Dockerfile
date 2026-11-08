ARG TORCH_IMAGE=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel@sha256:9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02
FROM ${TORCH_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST=7.5 NVIDIA_VISIBLE_DEVICES=void
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pip==24.0 setuptools==59.5.0 wheel==0.41.3
COPY docker/yachiyo/requirements-base.txt /opt/scout/requirements-base.txt
RUN python -m pip install --no-cache-dir -r /opt/scout/requirements-base.txt
COPY repos/yachiyo /opt/yachiyo
COPY docker/yachiyo/smoke.py /opt/scout/smoke.py
WORKDIR /opt/yachiyo

FROM base AS detection
ENV SCOUT_STAGE=detection PYTHONPATH=/opt/yachiyo/BoT-SORT
COPY docker/yachiyo/requirements-detection.txt /opt/scout/requirements-detection.txt
RUN python -m pip install --no-cache-dir -r /opt/scout/requirements-detection.txt
COPY docker/yachiyo/vendor/BoT-SORT /opt/yachiyo/BoT-SORT
WORKDIR /opt/yachiyo/BoT-SORT
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
    && cd fast_reid/fastreid/evaluation/rank_cylib && python setup.py build_ext --inplace
RUN cp /opt/yachiyo/detector/aic24_get_detection.py tools/aic24_get_detection.py
RUN python -m pip check && python /opt/scout/smoke.py --device cpu
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]

FROM base AS reid
ENV SCOUT_STAGE=reid PYTHONPATH=/opt/yachiyo/deep-person-reid:/opt/yachiyo/deep-person-reid/torchreid
COPY docker/yachiyo/requirements-reid.txt /opt/scout/requirements-reid.txt
RUN python -m pip install --no-cache-dir -r /opt/scout/requirements-reid.txt
COPY docker/yachiyo/vendor/deep-person-reid /opt/yachiyo/deep-person-reid
WORKDIR /opt/yachiyo/deep-person-reid
# Stable tensorboard exposes the same SummaryWriter API used here; avoid nightly drift.
RUN sed -i 's/^tb-nightly$/tensorboard==2.13.0/' requirements.txt \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
    && cp /opt/yachiyo/embedder/aic24_extract.py torchreid/aic24_extract.py
RUN python -m pip check && python /opt/scout/smoke.py --device cpu
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]

FROM base AS pose
ENV SCOUT_STAGE=pose
COPY docker/yachiyo/requirements-pose.txt /opt/scout/requirements-pose.txt
# The full wheel includes the C++/CUDA ops required by MMDetection.
RUN python -m pip install --no-cache-dir --only-binary=mmcv-full mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html \
    && python -m pip install --no-cache-dir --no-build-isolation -r /opt/scout/requirements-pose.txt
COPY docker/yachiyo/vendor/mmpose /opt/yachiyo/mmpose
WORKDIR /opt/yachiyo/mmpose
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
    && cp /opt/yachiyo/poser/*.py demo/
RUN python -m pip check && python /opt/scout/smoke.py --device cpu
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]

FROM python:3.10.14-slim-bookworm@sha256:2407c61b1a18067393fecd8a22cf6fceede893b6aaca817bf9fbfe65e33614a3 AS tracking
ENV SCOUT_STAGE=tracking PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY repos/yachiyo /opt/yachiyo
COPY docker/yachiyo/smoke.py /opt/scout/smoke.py
RUN python -m pip install --no-cache-dir pip==24.0 \
    && python -m pip install --no-cache-dir -r /opt/yachiyo/tracking/requirements.txt
WORKDIR /opt/yachiyo
RUN python -m pip check && python /opt/scout/smoke.py --device cpu
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]

