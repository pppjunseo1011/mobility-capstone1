# YACHIYO RIIPS 환경

검토 기준은 [공식 저장소](https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS)의 `main`, 커밋 `f881fe0edf60fe697efb2bf4c935b80a80861230`이다. 원본은 `repos/yachiyo`에 보존했다. README뿐 아니라 `scripts/*.sh`, `detector/aic24_get_detection.py`, `embedder/aic24_extract.py`, `poser/*.py`, `tracking/src/*.py`, `tracking/requirements.txt`를 확인했다.

## 단계와 환경 분리

| 단계/이미지 접미사 | 구현 | 선택 환경 | 확장 모듈 |
|---|---|---|---|
| `detection` | BoT-SORT 내부 YOLOX + FastReID import | Python 3.8.12, PyTorch 1.11.0, torchvision 0.12.0, CUDA 11.3, NumPy 1.23.5 | `yolox._C` C++, `cython_bbox`, `lap`, pycocotools, FastReID `rank_cy`/`roc_cy`, FAISS CPU |
| `reid` | torchreid OSNet | Python 3.8.12, PyTorch 1.11.0, torchvision 0.12.0, CUDA 11.3, NumPy 1.23.5 | torchreid `rank_cy` Cython |
| `pose` | MMPose 0.29.0 / HRNet | Python 3.8.12, PyTorch 1.11.0, CUDA 11.3, `mmcv-full` 1.7.0, MMDetection 2.28.2, NumPy 1.23.5 | MMCV C++/CUDA `_ext`, xtcocotools |
| `tracking` | 오프라인 SCPT/MCPT 클러스터링 | Python 3.10.14, NumPy 1.26.4, SciPy 1.13.0, scikit-learn 1.4.2 | wheel 제공 수치 연산 확장, PyTorch/CUDA 불필요 |

공식 README는 검출·ReID·포즈의 세 환경과 별도 추적 venv를 안내한다. 특히 추적의 NumPy 1.26/SciPy 1.13은 구형 BoT-SORT의 `np.float` 코드 및 Python 3.8 환경과 충돌하므로 네 이미지로 분리했다. 검출·ReID·포즈 이미지는 동일한 PyTorch 기반 레이어를 공유하지만 설치 환경은 각각 독립적이다.

공식 BoT-SORT의 시험 조합인 PyTorch 1.11.0+cu113 / torchvision 0.12.0을 유지했다. 공식 안내의 Python 3.7 대신 PyTorch 공식 이미지에 포함된 Python 3.8.12를 사용한다. 포즈는 공식 YACHIYO의 Python 3.8 계열과 맞추고, 추적은 요구사항의 최소 Python을 충족하는 3.10으로 분리했다. GPU 세 이미지의 기반 OS는 Ubuntu 18.04, 추적 이미지는 Debian 12이며 호스트 OS는 변경하지 않는다. TITAN RTX의 CUDA 아키텍처는 7.5이며, CUDA 11.3은 호스트 580 드라이버의 하위 호환 범위에 있다. `nvidia-smi`의 CUDA 13.0 표시는 컨테이너 CUDA 버전과 다르다. [NVIDIA 호환성 문서](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch 과거 버전](https://pytorch.org/get-started/previous-versions/).

## 버전 고정과 변경

외부 구현은 소스만 `docker/yachiyo/vendor`에 보관한다. 재확보 및 기존 작업 보호 검사는 `bash docker/yachiyo/fetch-vendor.sh`로 실행한다. 기존 폴더의 커밋 또는 수정 상태가 다르면 덮어쓰지 않고 중단한다.

| 소스 | 고정 커밋 | 선택 이유 |
|---|---|---|
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | `251985436d6712aaf682aaaf5f71edb4987224bd` | YACHIYO README가 지정한 외부 구현, 확인 시 HEAD |
| [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` | README 지정 torchreid, 확인 시 HEAD |
| [MMPose v0.29.0](https://github.com/open-mmlab/mmpose/tree/v0.29.0) | `0d9a00c1d64b68999e4b2a4ee5ed0d5ac2c3e296` | YACHIYO의 구형 API와 맞는 공식 지정 태그 |

상위 저장소는 외부 BoT-SORT/torchreid의 커밋을 제공하지 않으므로, 이 고정값이 저자의 당시 정확한 커밋이라는 의미는 아니다.

- 구형 코드의 NumPy 별칭을 보존하도록 전처리 세 환경은 NumPy 1.23.5를 선택했다. 추적 requirements는 그대로 설치한다.
- 공식 안내의 `mmcv==1.7.0` 대신 동일 버전 `mmcv-full==1.7.0`을 설치한다. MMDetection import 및 NMS에 필요한 바이너리 연산이 일반 mmcv 패키지에는 없다. cu113/torch1.11 공식 wheel을 사용하고, wheel 부재 시 임의 소스 빌드로 넘어가지 않는다.
- torchreid의 무고정 `tb-nightly`는 API가 같은 안정판 tensorboard 2.13.0으로 변경한다. 변경은 이미지 내부 `requirements.txt`에만 적용한다. 호스트 원본은 보존한다.
- ReID의 chardet은 기본 이미지 requests와 호환되는 4.0.0으로 고정했다. 최초 5.2.0 검사에서 나온 `RequestsDependencyWarning`을 해소하고 해당 이미지를 다시 빌드·CPU/GPU 검증했다.
- NumPy ABI/Cython 충돌을 피하도록 Cython 0.29.36, setuptools 59.5.0, pip 24.0을 선택한다. `lap`은 Python 3.8 wheel이 있는 0.5.12로 고정했다.
- BoT-SORT의 ONNX 1.8.1/onnxruntime 1.8.0/onnx-simplifier 0.3.5는 모델 export용이며 이번 PyTorch 실행 경로에서 import되지 않아 제외했다. TensorRT/ONNX export는 검증 대상이 아니다.
- FAISS는 검출 코드의 강제 import를 충족하는 CPU 1.7.4를 설치한다. FAISS GPU 검색이나 GPU 재순위화는 이번 환경의 검증 범위가 아니다.
- `MAX_JOBS=2`, BLAS/OpenMP 스레드 제한으로 공용 서버의 빌드 부하를 제한했다.
- 구형 CUDA 기본 이미지의 오래된 NVIDIA apt 저장소 목록은 이미지 내부에서만 제거한다. CUDA/cuDNN은 기본 이미지에 이미 포함되어 있고 Ubuntu 패키지 설치만 필요하므로, 만료된 서명키로 인한 apt 오류를 피한다. 호스트 apt 설정은 변경하지 않는다.

## 빌드와 환경 검사

프로젝트 루트 `scout-mtmc`에서 실행한다. 외부 소스 확보는 최초 또는 검증 시 수행한다.

```bash
bash docker/yachiyo/fetch-vendor.sh
for stage in detection reid pose tracking; do
  docker build --progress=plain --target "$stage" \
    -f docker/yachiyo/Dockerfile \
    -t "scout-mtmc/yachiyo-${stage}:f881fe0" .
done

docker run --rm --runtime=runc --network=none \
  scout-mtmc/yachiyo-detection:f881fe0 \
  python /opt/scout/smoke.py --device cpu

# 승인받은 GPU 번호 또는 UUID를 호출 시 지정한다.
GPU_DEVICE=0
docker run --rm --runtime=nvidia --gpus "device=${GPU_DEVICE}" \
  --env "NVIDIA_VISIBLE_DEVICES=${GPU_DEVICE}" \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --network=none --cpus=2 --memory=4g \
  scout-mtmc/yachiyo-detection:f881fe0 \
  python /opt/scout/smoke.py --device cuda
```

위 검사 명령의 이미지에서 `detection`을 `reid`, `pose`로 바꿔 각 환경을 검사한다. `tracking`은 CPU 전용이므로 `--device cpu`만 사용한다. 기본 CMD는 CPU 환경 검사이며 원본 추론 스크립트를 자동 실행하지 않는다. 검사는 작은 행렬곱, torchvision NMS, 해당 단계 확장 로딩과 상위 실행 코드 import를 수행한다. 포즈는 MMCV NMS의 CPU/CUDA 경로도 확인한다. 데이터/체크포인트 다운로드 없이 네트워크를 끈 컨테이너로 확인한다.

## SCOUT 연결 전에 필요한 작업

공통 실행 래퍼는 호스트에서 지정한 데이터 폴더를 읽기 전용 `/data/scout`로, 별도 결과 폴더를 `/outputs`로 연결한다. 이 마운트가 원본 AIC 입력을 SCOUT로 자동 변환하지는 않는다. 원본 `scripts/*.sh`는 conda 환경 활성화, 코드 복사, 상대경로, AIC scene/camera 번호를 사용하므로 컨테이너에서 그대로 실행하는 운영 경로는 아직 제공하지 않는다. 필요한 코드 복사는 이미지 빌드 때 완료했다.

- 검출: `Original/scene_NNN/camera_NNNN/Frame/*.jpg`, `Detection` 텍스트/JSON 경로와 입력 프레임 규약을 어댑터가 정해야 한다. `bytetrack_x_mot17.pth.tar` YOLOX-X 가중치가 필요하다.
- ReID: 검출 파일과 crop 좌표, 프레임 파일명이 연결되어야 한다. `osnet_ms_m_c.pth.tar` 경로 및 학습 도메인을 팀에서 확정해야 한다. 원본 `FeatureExtractor`는 파일이 없으면 ImageNet 사전학습 가중치를 자동 다운로드할 수 있어 실제 실행 전에 명시적 존재 검사가 필요하다.
- 포즈: `hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth`와 HRNet config가 필요하다. 공식 shell은 HTTP checkpoint URL을 전달하여 실행 시 자동 다운로드한다. 다음 단계에서는 로컬 읽기 전용 weights mount와 명시적 경로를 사용해야 한다.
- 추적: camera/scene 매핑(`tracking/config/scene_2_camera_id_file.json`), FPS(기본 30), 동기화, feature/pose 경로, 카메라별 `calibration.json`의 `homography matrix`를 SCOUT에 맞게 검토해야 한다. 일부 코드가 calibration 미존재를 경고하더라도 `measure_world_coordinate`는 파일을 직접 열므로 전체 파이프라인이 자동으로 calibration 없이 동작한다고 가정하면 안 된다.
- 추적 결과를 공통 평가 포맷으로 변환하고, frame/person/camera ID의 의미를 확정하는 작업은 다음 단계다.

이번 작업에서는 가중치, SCOUT 데이터, 모델 추론, 학습, 성능 평가를 수행하지 않는다. 환경 검사 성공은 실제 추적 성공이나 SCOUT 적합성을 증명하지 않는다.

## 검증 결과

2026-10-25 KST에 **네 이미지 빌드와 CPU 실행 4/4, CUDA 대상 GPU 실행 3/3이 성공**했다. 각 빌드의 `pip check`도 통과했다. 실행 검사는 UID/GID 1000, `--network none`, CPU 2개/메모리 8GiB 제한으로 수행했다. CUDA 검사는 승인된 호스트 GPU 0의 TITAN RTX를 사용했으며 capability 7.5로 확인했다. 실제 모델 가중치 추론은 수행하지 않았다.

| 이미지 | 최종 이미지 ID 앞 12자리 | 빌드·CPU | GPU 0 |
|---|---|---|---|
| `scout-mtmc/yachiyo-detection:f881fe0` | `2094117ed652` | PASS | PASS |
| `scout-mtmc/yachiyo-reid:f881fe0` | `dbf917dca7cf` | PASS | PASS |
| `scout-mtmc/yachiyo-pose:f881fe0` | `9497ec27ae53` | PASS | PASS |
| `scout-mtmc/yachiyo-tracking:f881fe0` | `84bcd82ef6ce` | PASS | 해당 없음: CPU 전용 |

최종 실행 로그:

- 검출: [CPU](../logs/yachiyo-detection-cpu-20261025T170707001856.log), [GPU](../logs/yachiyo-detection-gpu-20261025T170708853248.log)
- ReID: [CPU](../logs/yachiyo-reid-cpu-20261025T170950665944.log), [GPU](../logs/yachiyo-reid-gpu-20261025T170951657402.log)
- 포즈: [CPU](../logs/yachiyo-pose-cpu-20261025T170716860323.log), [GPU](../logs/yachiyo-pose-gpu-20261025T170718737020.log)
- 추적: [CPU](../logs/yachiyo-tracking-cpu-20261025T170011878912.log)

각 단계의 `logs/yachiyo-<stage>-build.log`에 빌드·확장 컴파일·pip check 결과가 있고, tracking의 최종 digest 고정 재빌드 기록은 `logs/yachiyo-tracking-build-final.log`이다. `*-packages.json`은 실제 설치 버전 목록, `*-freeze.txt`는 원본 pip freeze, `*-image.json`은 Docker 이미지 설정과 전체 ID를 기록한다. 기본 이미지의 Conda 패키지는 freeze에 빌드 당시의 `file://` 경로로 표시되므로 freeze를 그대로 재설치 입력으로 쓰지 않는다. 재빌드는 digest가 고정된 Dockerfile과 단계별 requirements를 사용한다.

포즈 CPU 검사에서 나오는 `No CUDA runtime is found`는 GPU를 노출하지 않은 검사에 따른 메시지이며 GPU 검사는 별도로 통과했다. MMCV의 2.0 전환 안내는 구형 라이브러리의 공통 경고다. 남은 작업은 가중치 준비와 SCOUT 입력/카메라/보정 어댑터, 실제 추론 및 공통 평가 검증이며 이번 환경 검사로 완료되었다고 처리하지 않았다.
