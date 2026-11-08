# UWIPL·ETRI 환경

2026-11-07 V3 순차 실험: V2(넓이1,000·거리80)를 재현하고 E1~E6를 한 조건씩 변경했다. 5초·30초 평균 view-aware IDF1 기준으로 앵커11시점·score>0.95·거리80 유지, 다수결15개 관측·new=0.7·high=0.4, STCRA 4/3/2가 선정됐다. STCRA는 공유 IDF1을 높였지만 HOTA와 IDSW를 악화했으므로 STCRA off 결과도 함께 검토해야 한다. GT 진단 4개와 timestamp 정렬된 별도 구간 6개를 포함해 총 54행 지표를 생성했다. 별도 30초에서 V2→선정 최종 설정은 공유 IDF1 36.18→36.94, IDSW 422→900이며, E5까지만 적용하면 IDSW 133이다. [7쪽 PDF](handoff/UWIPL_SCOUT_V3_단계별검증_20261107.pdf), [전체 비교표](../outputs/uwipl-v3-20261107/experiment_comparison.csv), [전달 ZIP](../deliverables/UWIPL_SCOUT_V3_20261107.zip), [결과 안내](../outputs/uwipl-v3-20261107/report/START_HERE.md)를 참조한다. GT 사용 진단은 정상 추론과 분리했고 기존 V2 및 원본 저장소는 보존했다.

2026-11-02 앵커 단일 변수 실험: 10 FPS의 기존 5초·30초 입력에서 넓이 10,000/5,000/2,500/1,000(거리80 고정), 거리80/60/40/20(넓이10,000 고정), 선정값 조합1,000/40을 비교했다. STCRA는 실행하지 않았다. 평균 view-aware IDF1로 각각1,000과40이 선택됐으나 조합은 넓이만 바꾼1,000/80보다 낮았다. [5쪽 보고서](handoff/UWIPL_SCOUT_앵커_단일변수실험_20261102.pdf), [16행 결과표](../outputs/uwipl-anchor-sweep-20261102/experiment_comparison.csv), [전달 ZIP](../deliverables/UWIPL_SCOUT_anchor_sweep_20261102.zip)을 참조한다. 기준 설정의 예측·GT 분포·평가 지표는 이전 결과와 정확히 일치한다. 같은 개발 구간 GT로 고른 설정이며 독립 테스트 성능이 아니다.

2026-11-01 추가 검증: SCOUT sequence_1의 cam_0~7, 원본 프레임 277~326(5초) 및 277~576(30초)에서 앵커·독립 STCRA·i-STCRA·보간을 실행했다. 두 시간 설정과 두 거리 설정의 32개 결과 행을 기록했다. 아래 초기 환경 구축 기록의 ‘미실행’은 2026-10-25 시점이다. 최신 결과는 [알고리즘 검증 PDF](handoff/UWIPL_SCOUT_알고리즘검증_20261101.pdf), [결과표](../outputs/uwipl-study-20261101/experiment_comparison.csv), [전달 ZIP](../deliverables/UWIPL_SCOUT_handoff_20261101.zip)을 참조한다. 2.5/2/1.5는 정확한 AIC 지도→미터 환산이 아닌 거리 후보이며, 30초 구간에는 최대 2.051초의 카메라 시차가 있다. 실행 완료를 공식 SCOUT 평가나 성능 개선으로 해석하지 않는다.

공식 저장소: <https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI>, `main`, 커밋 `8e23a69de57d2547b774399ae45792f83a2d17ef` (2026-10-25 확인).

README, `scripts/3_inference_det_real.sh`, `4_inference_det_syn.sh`, `5_inference_emb.sh`, 실제 패키지 초기화·설치 코드와 추적 코드를 검사했다. 이 저장소의 README는 4개 환경을 요구한다. SCOUT를 바로 입력할 수 있는 상태라는 의미가 아니다.

| Docker target / 이미지 | 역할 | 주요 버전 / 확장 |
|---|---|---|
| `botsort` / `scout-mtmc/uwipl-botsort:8e23a69` | YOLOX 실사 검출, BoT-SORT, clustering, STCRA | torch 1.11.0/cu113, torchvision 0.12.0, NumPy 1.23.5, `yolox._C` C++, `cython_bbox`, `lap` |
| `mmyolo` / `scout-mtmc/uwipl-mmyolo:8e23a69` | synthetic YOLOv7 검출 | vendored MMYOLO 0.5.0, MMCV 2.0.0, MMDetection 3.0.0, MMEngine 0.7.4 |
| `reid` / `scout-mtmc/uwipl-reid:8e23a69` | OSNet 임베딩 추출 | vendored torchreid 1.4.0, `rank_cy` Cython 확장 |
| `pose` / `scout-mtmc/uwipl-pose:8e23a69` | 기존 track bbox 기반 HRNet 발 키포인트 | vendored MMPose 0.29.0, mmcv-full 1.7.1, MMDetection 2.28.2, xtcocotools 1.13 |

MMYOLO의 `mmyolo/__init__.py`는 MMCV `>=2.0.0rc4,<2.1.0`, MMDetection `>=3.0.0rc6,<3.1.0`을 강제한다. MMPose는 MMCV `>=1.3.8,<=1.8.0`을 강제한다. 따라서 두 패키지를 하나의 Python 환경에 합칠 수 없다. 공식 단계 경계를 유지해 4개 이미지로 구성했다.

BoT-SORT README가 검증했던 Python 3.7 / torch 1.11.0+cu113 / torchvision 0.12.0 중 torch·CUDA·torchvision은 유지한다. Python은 공식 PyTorch 베이스 이미지에 포함된 3.8.12를 사용한다. torch 1.11 및 MMCV CUDA wheel을 같은 ABI로 설치하기 위한 선택이다. 컨테이너 OS는 이 공식 이미지의 Ubuntu 18.04이고 호스트 Ubuntu 20.04와 독립적이다. vendored MMPose의 예제 Dockerfile(torch 1.6/CUDA 10.1) 및 MMYOLO 예제(torch 1.9/CUDA 11.1)는 이 코드의 유일한 지원 버전이 아니다. 패키지 범위와 배포된 MMCV CUDA wheel을 기준으로 torch 1.11/cu113에 통일했다. 드라이버 580.126.09는 CUDA 11.3 런타임과 호환 가능하며 TITAN RTX의 아키텍처는 7.5이다. `nvidia-smi`의 CUDA 13.0은 드라이버 지원 상한이며 컨테이너 CUDA를 13으로 맞출 이유가 없다.

변경한 의존성은 `docker/uwipl/requirements-*.txt`에 고정했다. NumPy 1.23.5는 실제 코드의 `np.float` 사용을 지원한다. Pillow 9.5.0은 구형 torchvision/ReID 변환 API 호환성을 유지하며 scikit-learn 1.1.3은 기존 clustering API를 유지한다. torchreid의 `tb-nightly`는 컨테이너 사본에서 안정판 tensorboard 2.13.0으로 치환하며 호스트 저장소 원본은 수정하지 않는다. BoT-SORT의 ONNX 1.8.1/onnxruntime 1.8.0/onnx-simplifier 0.3.5는 선택적 export 도구로 추적·검출 Python 실행에 필요하지 않아 설치하지 않았다. absent `fast_reid/`를 참조하는 `mc_bot_sort.py`는 공식 AIC 실행 경로가 아니다. 실제 `tools/aic_track.py`는 `tracker/bot_sort.py`와 사전 추출 임베딩을 사용한다.

기존 track bbox를 사용하는 포즈 demo도 main에서 `has_mmdet`를 assert하므로 pose 이미지에는 호환 MMDetection 2.28.2를 설치한다. MMYOLO 이미지의 3.0.0과 분리한다.

필수 native 확장은 YOLOX COCOeval C++, `cython_bbox`, `lap`, torchreid rank Cython, MMCV CUDA 연산, pycocotools/xtcocotools이다. `deep-person-reid/torchreid/utils/GPU-Re-Ranking/extension/`의 선택적 CUDA GNN 재순위 확장은 공식 `aic_extract*.py`에서 import하지 않으며 이번 단계에는 빌드하지 않는다. 학습·export·선택적 GPU 재순위 전체 환경이 검증되었다고 해석하면 안 된다.

프로젝트 루트에서 각각 다음과 같이 빌드한다.

```bash
docker build --progress=plain --target botsort -f docker/uwipl/Dockerfile -t scout-mtmc/uwipl-botsort:8e23a69 .
docker build --progress=plain --target mmyolo -f docker/uwipl/Dockerfile -t scout-mtmc/uwipl-mmyolo:8e23a69 .
docker build --progress=plain --target reid -f docker/uwipl/Dockerfile -t scout-mtmc/uwipl-reid:8e23a69 .
docker build --progress=plain --target pose -f docker/uwipl/Dockerfile -t scout-mtmc/uwipl-pose:8e23a69 .
```

CPU 검증은 `docker run --rm --runtime=runc --network=none <이미지> python /opt/scout/smoke.py --device cpu`, GPU 검증은 `docker run --rm --runtime=nvidia --gpus "device=${GPU}" -e NVIDIA_VISIBLE_DEVICES="${GPU}" --network=none <이미지> python /opt/scout/smoke.py --device cuda`로 실행한다. `GPU`는 사용자가 실행 시 배정받은 번호로 지정한다. 기본 CMD는 CPU smoke이며 자동 추론은 실행하지 않는다. 공통 실행 스크립트와 데이터·결과 마운트는 루트 README를 따른다.

다음 단계 입력·가중치:

- YOLOX_x / ByteTrack MOT17 검출 checkpoint, AI City synthetic 환경별 YOLOv7 checkpoint. SCOUT에서 어떤 검출 경로를 비교에 사용할지 먼저 정의해야 한다.
- synthetic ReID checkpoint와 S001용 OSNet 5종(`osnet_ms_m_c`, `osnet_ibn_ms_m_c`, `osnet_ain_ms_m_c`, Market/MSMT17 모델). 경로는 공식 추출 스크립트에 하드코딩되어 있어 어댑터와 설정 수정 필요.
- HRNet W48 256x192 COCO checkpoint. README 예제 URL은 실행 시 자동 다운로드할 수 있으므로 이번에는 실행하지 않았다.
- 카메라별 동영상·프레임, scene/camera ID, 검출 TXT, 임베딩 NPY와 행 대응, track TXT, pose JSON, 카메라 보정/바닥평면·시간 동기 정보.
- `STCRA/cam_utils.py` 등의 AI City 카메라 정보, scene별 인원수·클러스터 수·경계/좌표 규칙과 `S001/S003/...` 가정은 SCOUT로 전용 시 검토·수정 대상이다. `data/test_det` 등의 원 저장소 샘플은 Docker context에서 제외했으며 SCOUT와 혼용하지 않는다.

실제 코드 검사에서 `torchreid/aic_extract.py`의 `make_parser()`가 이미 Namespace를 반환한 뒤 main에서 다시 `.parse_args()`를 호출하는 오류, `deep_person_reid`와 실제 폴더명 `deep-person-reid` 차이, 공식 embedding 셸 스크립트의 필수 `root_path` 인자 누락을 발견했다. 환경 import 검증과 별개인 실행 코드 문제로 기록하며 SCOUT 어댑터/추론 단계에서 수정해야 한다. `mmyolo/tools/aic_get_detection_syn.py`는 입력 `/data/test`도 하드코딩한다.

MMCV wheel 배포 확인: <https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html>.

`torchreid/metrics/rank.py`는 존재하지 않는 `evaluate_c`를 import하도록 원본에서 수정되어 있어 Cython 평가를 쓰지 않는다는 경고를 낸다. 실제 컴파일한 `rank_cy` 모듈과 `evaluate_cy` 함수는 smoke에서 직접 로드·호출해 검증한다. 이 원본의 평가 Python fallback은 후속 성능 평가에서 검토 대상이며 임베딩 추출 환경 실패로 보지 않는다.

## 실제 검증 결과 (2026-10-25, KST)

4개 이미지 모두 최종 Dockerfile로 빌드하고 `pip check`를 통과했다. 공통 `scripts/smoke.sh uwipl <stage> --cpu` 및 `--gpu 0`로 일반 사용자 UID/GID 1000:1000, 네트워크 차단, CPU 2개/메모리 8GiB 제한에서 컨테이너를 시작해 검증했다. 사용자가 허용한 GPU 0의 TITAN RTX 1장만 컨테이너에 노출했다.

| 단계 | 이미지 빌드 / pip check | CPU 시작·import·연산 | GPU 0 CUDA 연산 | 핵심 native 검증 |
|---|---|---|---|---|
| botsort | PASS | PASS | PASS | YOLOX C++ 로딩, Cython bbox IoU, LAP 할당, torchvision NMS |
| mmyolo | PASS | PASS | PASS | MMCV 2.0 CUDA 확장 로딩, CPU/GPU NMS |
| reid | PASS | PASS | PASS | rank_cy.evaluate_cy 합성 소입력 호출, torchvision NMS |
| pose | PASS | PASS | PASS | MMCV 1.7 CUDA 확장/xtcocotools 로딩, CPU/GPU NMS |

공통 실제 버전: Python 3.8.12, torch 1.11.0, torchvision 0.12.0, CUDA runtime 11.3, NumPy 1.23.5, OpenCV 4.8.1. 실제 실행 진입점 모듈도 모두 import했다. GPU 테스트는 4×4 행렬 연산과 bbox 2개 NMS뿐이며 모델 객체/가중치를 생성하거나 읽지 않았다.

로그는 `logs/uwipl-<stage>-build-20261025*.log`, `logs/uwipl-<stage>-cpu-20261025*.log`, `logs/uwipl-<stage>-gpu-20261025*.log`에 있다. 의존성 인벤토리는 `logs/uwipl-<stage>-pip-freeze.txt`이다. freeze에 보이는 베이스 Conda의 `file://` 경로는 원 이미지 빌드 경로로 재설치용 requirements가 아니며 재현은 고정 base digest 및 Dockerfile/requirements를 사용한다.

포즈의 MMCV 버전변경 안내, CPU 모드에서 CUDA 미노출 안내, Matplotlib이 쓰기 가능한 임시 cache 디렉터리로 전환한 안내는 검증 실패가 아니다. torchreid 평가 fallback 경고의 원본 코드 원인은 위에 설명했다.

실제 체크포인트 기반 모델 추론, SCOUT 연결·추적·성능 평가·학습은 **미실행**이다. 가중치를 별도로 다운로드하지 않았다. 원본 `repos/uwipl`은 `git status --short` 기준 변경이 없다.

생성 파일: `docker/uwipl/Dockerfile`, `requirements-common.txt`, `requirements-botsort.txt`, `requirements-mmyolo.txt`, `requirements-reid.txt`, `requirements-pose.txt`, `smoke.py`, 이 문서와 `logs/uwipl-*` 검증 로그.
