# Official BoT-SORT tested torch 1.11.0/cu113; vendored OpenMMLab stages are separated.
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel@sha256:9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02 AS base
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    TORCH_CUDA_ARCH_LIST="7.5" MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=void
# Old NVIDIA apt repositories have expired keys; CUDA already exists in the base image.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       build-essential git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/uwipl/requirements-common.txt /opt/scout/requirements-common.txt
RUN python -m pip install --no-cache-dir pip==24.0 setuptools==59.5.0 wheel==0.41.3 \
    && python -m pip install --no-cache-dir -r /opt/scout/requirements-common.txt
ENV PIP_CONSTRAINT=/opt/scout/requirements-common.txt
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]

FROM base AS botsort
ENV SCOUT_STAGE=botsort PYTHONPATH=/opt/uwipl/BoT-SORT:/opt/uwipl/STCRA
COPY docker/uwipl/requirements-botsort.txt /opt/scout/requirements-stage.txt
RUN python -m pip install --no-cache-dir --no-build-isolation -r /opt/scout/requirements-stage.txt
COPY repos/uwipl/BoT-SORT /opt/uwipl/BoT-SORT
COPY repos/uwipl/STCRA /opt/uwipl/STCRA
WORKDIR /opt/uwipl/BoT-SORT
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -m pip check
COPY docker/uwipl/smoke.py /opt/scout/smoke.py
RUN python /opt/scout/smoke.py --device cpu

FROM base AS reid
ENV SCOUT_STAGE=reid PYTHONPATH=/opt/uwipl/deep-person-reid:/opt/uwipl/deep-person-reid/torchreid
COPY docker/uwipl/requirements-reid.txt /opt/scout/requirements-stage.txt
RUN python -m pip install --no-cache-dir -r /opt/scout/requirements-stage.txt
COPY repos/uwipl/deep-person-reid /opt/uwipl/deep-person-reid
WORKDIR /opt/uwipl/deep-person-reid
# Stable tensorboard provides the SummaryWriter API used by this source; avoid tb-nightly drift.
RUN sed -i 's/^tb-nightly$/tensorboard==2.13.0/' requirements.txt \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -m pip check
COPY docker/uwipl/smoke.py /opt/scout/smoke.py
RUN python /opt/scout/smoke.py --device cpu

FROM base AS mmyolo
ENV SCOUT_STAGE=mmyolo PYTHONPATH=/opt/uwipl/mmyolo
COPY docker/uwipl/requirements-mmyolo.txt /opt/scout/requirements-stage.txt
RUN python -m pip install --no-cache-dir --only-binary=mmcv mmcv==2.0.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html \
    && python -m pip install --no-cache-dir -r /opt/scout/requirements-stage.txt
COPY repos/uwipl/mmyolo /opt/uwipl/mmyolo
WORKDIR /opt/uwipl/mmyolo
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -m pip check
COPY docker/uwipl/smoke.py /opt/scout/smoke.py
RUN python /opt/scout/smoke.py --device cpu

FROM base AS pose
ENV SCOUT_STAGE=pose PYTHONPATH=/opt/uwipl/mmpose
COPY docker/uwipl/requirements-pose.txt /opt/scout/requirements-stage.txt
RUN python -m pip install --no-cache-dir --only-binary=mmcv-full mmcv-full==1.7.1 \
      -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html \
    && python -m pip install --no-cache-dir --no-build-isolation -r /opt/scout/requirements-stage.txt
COPY repos/uwipl/mmpose /opt/uwipl/mmpose
WORKDIR /opt/uwipl/mmpose
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && python -m pip check
COPY docker/uwipl/smoke.py /opt/scout/smoke.py
RUN python /opt/scout/smoke.py --device cpu
