# MIG 사전 생성 컨테이너 4개로 이관

2026-11-08 갱신: UWIPL V3까지 포함한 실제 전송 목록 3개와 용량을 [최신 이동 안내](migration/README.md)에 정리했다. 전체 권장 묶음은 약 20.6 GiB이며 SCOUT 원본은 제외한다. 아래 기존 실험 목록에 더해 `outputs/uwipl-v3-20261107/`와 `deliverables/UWIPL_SCOUT_V3_20261107.zip`도 이동한다.

작성: 2026-11-06. 현재 서버의 Dockerfile, requirements, 코드와 실제 파일을 확인한 이관 계획이다. 새 서버에는 접속하지 않았으므로 새 환경 설치·H200 실행 완료를 뜻하지 않는다. SCOUT 원본은 이미 새 서버에 있다.

## 배정과 실행 방식

| 배정 | 담당 | 컨테이너 내부 환경 |
|---|---|---|
| 1번 | OSC = YACHIYO_RIIPS | osc-det / osc-reid / osc-pose / osc-track |
| 2번 | PoseTrack | posetrack; 필요하면 CPU 평가 환경 별도 |
| 3번 | UWIPL·ETRI | uwipl-botsort / uwipl-reid / uwipl-pose / uwipl-mmyolo(선택) |
| 4번 | SCOUT 어댑터·공통 평가·향후 baseline | scout-adapter / scout-eval / baseline(모델 결정 후) |

위 이름은 Conda/micromamba 환경 또는 venv의 이름이다. Docker 컨테이너를 추가 생성하지 않는다. 한 컨테이너 안의 여러 환경은 그 컨테이너에 배정된 CPU/RAM/GPU/VRAM 한도를 함께 사용한다. 가상환경을 늘려도 자원이 늘어나지 않는다. 단계별 순차 실행을 기본으로 한다.

GPU 분리는 MIG, CPU/RAM 제한은 관리자가 설정한 컨테이너/cgroup 배정에 따른다. GPU를 4개로 나눴다는 정보만으로 각각의 VRAM을 전체의 1/4로 계산하지 않는다. 실제 MIG profile과 컨테이너 한도를 확인한다. 한 MIG만 노출됐다면 PyTorch에서 cuda:0으로 보일 수 있으며 배정 번호 1/2/3/4를 CUDA device 번호로 쓰지 않는다. 관리자 GPU 노출 변수를 덮어쓰지 않는다.

## 파일 배치

다음은 권장 경로이며 실제 공유 마운트 여부를 먼저 확인한다.

```text
/workspace/shared/
  datasets/                     # 이미 옮긴 SCOUT, 모두 읽기 접근
  scout-mtmc/                   # 이관한 원본 자료 및 공통 코드 기준본
    adapters/ scripts/ tests/ vendor/ docs/ sources.lock.json
    repos/ docker/ weights/     # 공유 기준본; 실행용 editable 소스는 개인 복사
    outputs/                    # 옮기기로 선택한 기존 실험/입력 캐시
  results/
    osc/ posetrack/ uwipl/ adapter/ baseline/

각 컨테이너의 개인 영속 경로(예: /workspace/local/):
  envs/                        # 컨테이너별 설치; 공유 환경으로 만들지 않음
  work/                        # 담당 모델의 수정/빌드 가능한 소스
  cache/ tmp/                   # pip, torch extension, 모델 cache, 임시 출력
```

같은 /workspace/local 경로명이어도 실제로 개인 영역인지 확인해야 한다. 결과 폴더는 담당자만 쓰고 다른 담당자는 읽는 규칙을 사용한다. 실행별 새 디렉터리와 완료 표시/manifest를 두어 작성 중인 CSV·NPY를 읽지 않게 한다. UID/GID와 공유 그룹 권한을 맞춘다. 원본 SCOUT를 각 컨테이너에 중복 복사할 필요는 없다.

## 이동할 파일

아래 경로는 현재 scout-mtmc/ 기준이다. 구체적 존재 여부와 선택 등급은 migration-transfer-inventory.tsv에 기록했다.

| 구분 | 이동 대상 | 이유 |
|---|---|---|
| 공통 필수 | adapters/ 전체, scripts/ 전체, tests/, vendor/TrackEval/ | 데이터 로더·기하·입출력 변환·실험 runner·공통 평가. scripts는 약 1.1 MiB라 파일 몇 개만 골라 의존성을 빠뜨릴 필요가 없다 |
| 공통 필수 | sources.lock.json, README.md, docs/ | 커밋·프로토콜·수정 사항·제약과 기존 보고서 |
| 설치 명세 | docker/ 전체 | Docker를 실행하기 위해서가 아니라 requirements·설치 순서·smoke 및 vendored 소스를 보존하기 위해 필요 |
| OSC | repos/yachiyo/, docker/yachiyo/vendor/ | 원본 repo만으로는 BoT-SORT·torchreid·MMPose 추가 소스가 빠짐 |
| PoseTrack | repos/posetrack/ | detection, fast-reid, mmpose, track/aic_cpp 포함 |
| UWIPL | repos/uwipl/ | BoT-SORT, deep-person-reid, mmpose, mmyolo, STCRA 포함 |
| 가중치 | weights/ 전체 또는 담당 모델 하위 폴더 + 필요한 shared 파일 | 현재 전체 약 6.7 GiB의 실제 디스크 사용량. manifest/SHA256SUMS도 포함 |
| 검토용 결과 | deliverables/에서 각 모델 최신 ZIP과 sha256 | 결과 비교와 인계용. 이것만으로 전체 추론을 재현할 수는 없음 |

가중치에는 하드링크가 있다. 같은 체크포인트를 여러 경로에서 재사용하며 단순 복사하면 크기가 커질 수 있다. 전체 weights를 한 번에 rsync -aH로 옮기면 하드링크 관계를 보존할 수 있다. 하드링크를 지원하지 않는 저장소에서는 파일을 일반 복사해도 되지만 중복 공간을 계산한다. 기존 .download.lock, .part, 실행 PID/status는 옮길 필요 없다.

### 각 모델의 가중치

| 배정 | 기본 경로 | 사용 목적 |
|---|---|---|
| 1 OSC | weights/yachiyo/ | bytetrack_x_mot17 검출, osnet_ms_m_c Re-ID, HRNet-W48 포즈 |
| 2 PoseTrack | weights/posetrack/ | bytetrack_x_mot17, aic24.pkl, HRNet-W32. lup_moco_r101은 별도 실험/사전학습용이며 현재 권장 추적의 필수 추론 체크포인트로 단정하지 않음 |
| 3 UWIPL | weights/uwipl/ | YOLOX 검출, OSNet 5종, HRNet-W48 |
| 3 UWIPL 선택 | weights/shared/epoch_*.pth | YOLOv7 5종. 실영상 YOLOX 경로만 쓰면 초기 이관에서 생략 가능 |
| 4 baseline | 미정 | 모델을 선정한 뒤 필요한 가중치만 추가. 공통 파일은 기존 shared 사본 재사용 |

기존 UWIPL 실제 실행 포즈는 top-down HRNet-W48/MMPose 0.29 경로다. 이름이 비슷하다는 이유로 HigherHRNet 환경·가중치와 동일하게 취급하지 않는다. 합성 영상용 Re-ID 일부는 기존 조사에서 미확보로 기록돼 있으므로 weights manifest를 기준으로 한다.

### 기존 실험을 이어갈 때 추가로 이동

| 담당 | 캐시/실험 디렉터리 | 범위 |
|---|---|---|
| OSC | outputs/osc-inference-uncapped-20261101/, osc-pilot-uncapped-20261101/, osc-analysis-20261101/ | 인원 제한 해제 입력, 5초·30초 파라미터 실험과 결과. runner가 이전 경로를 참조하므로 상대 구조 유지 |
| PoseTrack | outputs/posetrack-time-study-20261106/aligned_input/ | 시간 정렬된 실제 neural 입력 약 388 MiB. 재검출 없이 추적을 다시 돌릴 때 필요 |
| PoseTrack | 같은 폴더의 frame_mapping.csv, frame_mapping_all_sources.csv, timestamp_policy.json, input_sha256.json | 입력 시점과 원본 프레임 대응·입력 계보 |
| PoseTrack 비교 | outputs/posetrack-mesh-20261104/full_input30/ | 수정 전 입력 약 392 MiB. 기존 시간 매핑과 비교할 때 필요 |
| PoseTrack 평가 | outputs/posetrack-time-study-20261106/handoff/03_evaluation/shared_gt.json 또는 최신 ZIP | 명시적 --gt로 사용 가능. 기본 경로를 쓸 경우 옛 osc-analysis/.../gt_selected.json 의존성도 확인 |
| UWIPL | outputs/uwipl-study-20261101/ | 약 2.2 GiB. 5초·30초 입력/추적/포즈/STCRA 및 비교 결과 |
| UWIPL | outputs/uwipl-anchor-sweep-20261102/ | 앵커 조건 비교 설정·실험 결과 |
| 4 공통 평가 | 최신 PoseTrack ZIP의 frame_mapping.csv, shared_gt.json, 공통 CSV, 평가 코드 | 현재 30초 공통 프로토콜을 그대로 재사용 가능. 기존 다른 모델 결과를 같은 시간 정책에 맞춰야 공정 비교 가능 |

이 목록은 대표 이어하기 입력이다. 모든 과거 분석 스크립트가 이 파일들만으로 실행된다는 뜻은 아니다. 이전 결과 재현 시 해당 runner가 참조하는 원본 관측·GT·카메라 보정·이미지 경로를 함께 확인한다. 신규 데이터에서 다시 추론할 경우 과거 신경망 캐시 이동은 선택 사항이다.

기존 데이터셋 압축파일, 추출 중간파일, 전체 Docker 이미지, Conda/venv 디렉터리, __pycache__, build/, 예전 컴파일 .so, 대용량 전체 Affinity 로그·실패 실행은 기본 이동 대상에서 제외한다. 기존 서버에서 삭제하지 않는다. 결과 검토용 ZIP에는 원본 로그의 경로·해시가 있다.

## 라이브러리: 기존 검증 버전과 새 서버 검증을 구분

아래 표는 **TITAN RTX 서버에서 검증한 설치 조합**이다. H200에서 동일 조합 전체가 검증되었다는 의미가 아니다. 최신 버전으로 일괄 업그레이드하지 않는다.

| 환경 | 기존 Python / Torch / CUDA | 주요 패키지 |
|---|---|---|
| osc-det | 3.8.12 / 1.11.0 / 11.3 | torchvision 0.12.0, NumPy 1.23.5, SciPy 1.10.1, YOLOX/BoT-SORT 소스, cython_bbox 0.1.5, lap 0.5.12, faiss-cpu 1.7.4 |
| osc-reid | 3.8.12 / 1.11.0 / 11.3 | torchreid 소스, NumPy 1.23.5, TensorBoard 2.13.0, Pillow 9.5.0 |
| osc-pose | 3.8.12 / 1.11.0 / 11.3 | MMPose 0.29.0, mmcv-full 1.7.0, MMDetection 2.28.2, xtcocotools 1.13 |
| osc-track | 3.10.14 / 불필요 / 불필요 | NumPy 1.26.4, SciPy 1.13.0, scikit-learn 1.4.2, OpenCV-headless 4.9.0.80, Matplotlib 3.8.4 |
| posetrack | 3.9.23 / 2.0.1+cu118 / 11.8 | torchvision 0.15.2+cu118, NumPy 1.23.5, SciPy 1.10.1, MMCV 2.0.1, MMEngine 0.10.4, MMDetection 3.1.0, 동봉 MMPose 1.3.1, FAISS-GPU 1.7.2, cvxopt 1.3.2, pybind11 2.12.0, lap 0.5.12 |
| uwipl-botsort | 3.8.12 / 1.11.0 / 11.3 | torchvision 0.12.0, NumPy 1.23.5, SciPy 1.9.3, scikit-learn 1.1.3, pandas 1.5.3, lap 0.4.0, cython-bbox 0.1.3, motmetrics 1.4.0, filterpy 1.4.5 |
| uwipl-reid | 3.8.12 / 1.11.0 / 11.3 | torchreid 1.4.0 소스, OSNet, TensorBoard 2.13.0 |
| uwipl-pose | 3.8.12 / 1.11.0 / 11.3 | MMPose 0.29.0, mmcv-full 1.7.1, MMDetection 2.28.2, xtcocotools 1.13 |
| uwipl-mmyolo 선택 | 3.8.12 / 1.11.0 / 11.3 | MMYOLO 0.5.0 소스, MMCV 2.0.0, MMEngine 0.7.4, MMDetection 3.0.0 |
| scout-adapter | 3.9 기준 / 불필요 / 불필요 | NumPy 1.23.5, OpenCV 4.8.1.78, Pillow 9.5.0, ffmpeg/ffprobe. 메타데이터 기능만은 표준 라이브러리 |
| scout-eval | 3.9 기준 / 불필요 / 불필요 | NumPy 1.23.5, SciPy 1.10.1, vendor/TrackEval, Matplotlib 3.7.5, Pillow. 기존 전체 UWIPL 분석까지 하면 추가 패키지 필요 |
| baseline | 아직 미정 | 알고리즘·repo·검출/Re-ID 선정 후 별도 환경 구성. 어댑터 Python 환경에 혼합 설치하지 않음 |

전 패키지는 다음 파일을 기준으로 한다.

- OSC: docker/yachiyo/requirements-base.txt + 각 stage requirements, repos/yachiyo/tracking/requirements.txt.
- PoseTrack: docker/posetrack/requirements.txt. torch/torchvision/MMCV 및 소스 설치는 Dockerfile에 별도로 있어 requirements만으로 끝나지 않는다.
- UWIPL: docker/uwipl/requirements-common.txt + 각 stage requirements. mmyolo와 pose는 MMCV 2.x/1.x 충돌로 서로 다른 환경이 필요하다.
- 설치 순서, vendored 코드 복사, tb-nightly 치환, 커스텀 demo 파일 복사는 각 Dockerfile의 RUN/COPY를 컨테이너 내부 설치 절차로 옮긴다. Dockerfile 자체를 실행하지 않는다.
- 기존 freeze의 file:// 및 /opt editable 경로를 그대로 pip install 입력으로 사용하지 않는다.

## H200에서 중요한 차이

H200은 compute capability 9.0이다. CUDA 11.8은 Hopper용 native 코드를 만들 수 있다. 이전 CUDA 바이너리도 적절한 PTX가 있으면 실행될 수 있지만, 없으면 재빌드가 필요하다. 따라서 기존 OSC/UWIPL의 torch 1.11/cu113을 무조건 실행 가능 또는 불가능으로 단정하지 않는다.

실무 순서:
1. 새 컨테이너의 Python·OS·드라이버·노출 GPU·CPU/RAM·설치 권한·nvcc 유무 확인.
2. PoseTrack은 기존 torch 2.0.1/cu118 조합을 첫 검증 대상으로 삼되 MMCV와 실제 모델 CUDA 연산까지 검사.
3. OSC/UWIPL GPU 환경은 Hopper 지원 PyTorch/CUDA 조합과 해당 MMCV 버전의 호환성을 따로 검토한다. 변경 필요 시 소스 빌드·구 API 수정과 출력 회귀 검증을 함께 수행한다. 아직 H200용 최종 lock은 확정하지 않았다.
4. CUDA 확장을 새로 빌드할 때는 지원 toolkit에서 TORCH_CUDA_ARCH_LIST=9.0 또는 9.0+PTX를 검토한다. 기존 Dockerfile의 7.5 고정을 복사하지 않는다. 환경변수만 바꾼다고 이미 설치된 wheel이 재컴파일되지는 않는다.
5. torch/torchvision/MMCV는 ABI가 맞는 조합이어야 한다. CUDA matmul 성공만으로 MMCV NMS/RoIAlign 등 확장 정상 동작을 판단하지 않는다. FAISS-GPU도 실제 사용하는 GPU 경로를 별도로 검사한다.

출처: [NVIDIA GPU compute capability](https://developer.nvidia.com/cuda/gpus), [Hopper 호환성 안내](https://docs.nvidia.com/cuda/hopper-compatibility-guide/).

## 컨테이너 안에서 필요한 시스템 도구·확장

- 공통: git, ffmpeg/ffprobe, build-essential(GCC/G++/make), 해당 Python 개발 헤더, ca-certificates, libglib2.0-0, libgl1, libgomp1. 기존 UWIPL 환경은 libsm6/libxext6도 사용.
- 소스 빌드 환경: setuptools, wheel, Cython, ninja, pybind11(해당 단계). CUDA 확장 소스 빌드에는 nvcc와 맞는 개발 헤더/toolkit 필요; GPU가 보이거나 torch.cuda가 동작해도 nvcc가 설치돼 있다는 의미는 아니다.
- OSC: YOLOX C++, cython_bbox/lap, FastReID rank/roc Cython, torchreid rank Cython, MMCV _ext.
- PoseTrack: track/aic_cpp, FastReID rank Cython, MMCV _ext, cython_bbox/lap. 이 실험의 HiGHS는 SciPy 경로; 설치된 cvxopt 원본 경로와 구분.
- UWIPL: YOLOX C++, cython_bbox/lap, torchreid rank Cython, 각 환경의 MMCV _ext. 사용하지 않은 선택적 CUDA GNN 재순위 확장을 필수로 추가하지 않는다.

설치 권한이 없다면 사용자 경로의 micromamba/Conda로 Python 버전을 분리하고 필요한 시스템 패키지·toolkit만 관리자에게 요청한다. venv는 Python 버전 자체를 바꾸지 못한다. 시스템 Python, 호스트 드라이버, MIG 설정을 변경하지 않는다.

## 기존 실행 스크립트 이관 시 수정 사항

라이브러리 설치만으로 현재 runner가 바로 실행되는 것은 아니다.

- scripts/run.sh, smoke.sh, replay-posetrack-*.sh 등 docker run 래퍼를 새 서버에서 그대로 실행하지 않는다. 내부 python 명령을 지정 환경의 python/conda run으로 실행하도록 새 launcher가 필요하다.
- /project, /opt/posetrack, /opt/yachiyo, /opt/uwipl, /data, /out, /outputs, /inference 경로가 실제 코드에 있다. PROJECT_ROOT/REPO_ROOT/DATA_ROOT/INPUT_ROOT/OUTPUT_ROOT 인자로 바꾸거나, 권한과 충돌을 확인한 컨테이너 내부 경로 대응이 필요하다. 현재 이 문서는 이식 완료를 의미하지 않는다.
- editable 설치·Cython 빌드는 각 컨테이너 개인 work 소스에서 수행한다. 공유 원본 폴더에 서로 다른 Python/ABI의 .so와 egg-info를 생성하지 않는다.
- 공유 폴더에 venv/Conda env, pip cache, torch extension cache를 하나로 합치지 않는다. 특히 CUDA_VISIBLE_DEVICES="" 또는 NVIDIA_VISIBLE_DEVICES=void 같은 기존 이미지의 GPU 비노출 설정을 baseline 환경으로 복사하지 않는다.
- OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, MAX_JOBS와 DataLoader workers는 실제 할당 CPU/RAM에 맞춘다. 설치된 host 전체 CPU 개수를 자원 한도로 가정하지 않는다.

## 이관 완료 판정

1. 파일 전송 후 manifest 체크섬, 모델 커밋, SCOUT 및 공유 GT 읽기 권한을 확인한다.
2. 단계별 별도 환경에서 pip check, 핵심 import, 확장 로딩을 검사한다.
3. 배정된 MIG에서 작은 CUDA 연산과 실제 사용하는 torchvision/MMCV 연산, 가중치 로딩을 확인한다.
4. 기존과 같은 1~2프레임 신경망 추론 후 shape/NaN/BBox-Keypoint 대응 검사. GPU·라이브러리 변경이 수치 결과를 바꿀 수 있으므로 이전 hash와 무조건 동일하다고 요구하지 않는다.
5. 기존 캐시로 짧은 추적을 실행해 공통 CSV와 평가를 확인한다. 새 신경망 결과 비교와 캐시 기반 association 비교를 분리한다.

새 서버 접속 주소·MIG profile·컨테이너 OS·설치 권한은 아직 확인하지 않았다. 이번 완료 범위는 실제 파일과 기존 설치 명세에 근거한 이동 목록·환경 분리 계획 작성이다.
