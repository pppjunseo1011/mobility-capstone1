FROM python:3.9.23-slim-bookworm@sha256:7bffea15bcc3d7fb87cf10a027986203e4281e078fa2f5b234c30fca291f0834
ARG SOURCE_COMMIT=ba727e6bc6e45e44cd6e006b73e00864892173c2
LABEL org.opencontainers.image.source="https://github.com/ZhenyuX1E/PoseTrack" \
      org.opencontainers.image.revision=$SOURCE_COMMIT
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    NVIDIA_VISIBLE_DEVICES=void NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN sed -i "s|http://deb.debian.org|https://deb.debian.org|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o Acquire::http::No-Cache=true -o Acquire::https::No-Cache=true \
    && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates libglib2.0-0 libgl1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0 Cython==0.29.37 numpy==1.23.5
RUN python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
COPY docker/posetrack/requirements.txt /opt/scout/requirements.txt
RUN python -m pip install --no-build-isolation -r /opt/scout/requirements.txt
# Require a CUDA wheel: never silently fall back to compiling MMCV without CUDA.
RUN python -m pip install mmcv==2.0.1 --no-deps --only-binary=:all: \
    --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
COPY repos/posetrack /opt/posetrack
WORKDIR /opt/posetrack
RUN python -m pip install --no-deps --no-build-isolation ./track/aic_cpp \
    && python -m pip install --no-deps --no-build-isolation -e ./mmpose \
    && cd fast-reid/fastreid/evaluation/rank_cylib && python setup.py build_ext --inplace
ENV PYTHONPATH=/opt/posetrack/detection:/opt/posetrack/fast-reid:/opt/posetrack/track \
    SCOUT_DATA_DIR=/data/scout SCOUT_OUTPUT_DIR=/outputs
COPY docker/posetrack/smoke.py /opt/scout/smoke.py
RUN python -m pip check && python /opt/scout/smoke.py --device cpu \
    && python -m pip freeze > /opt/scout/installed-requirements.txt
CMD ["python", "/opt/scout/smoke.py", "--device", "cpu"]
