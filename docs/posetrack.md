# PoseTrack 실행 환경

2026-11-06 후속 검증: 실제 timestamp 기반으로 8카메라 30초를 정렬해 Ground 제외 / Mesh / 높이 판정 유보 후보를 재실행했다. 최대 시각 차이는 2.051초에서 35ms로 감소했다. 권장 Mesh 결과는 29,226행·88 ID, HOTA 44.05, 공유 Global IDF1 47.18, IDSW 308이다. 추가 높이 후보는 성능 저하로 채택하지 않았다. 기존 결과와는 동일 유효 GT 범위로 비교하며 시간 정렬 자체를 정확도 개선으로 해석하지 않는다. [결과와 한계](../outputs/posetrack-time-study-20261106/report/START_HERE.md), [PDF](handoff/PoseTrack_SCOUT_30초_추가검증_20261106.pdf), [결과·코드·평가 GT 전달 ZIP](../deliverables/PoseTrack_SCOUT_30s_followup_20261106.zip)을 참조한다.

최신 전체 영역 검증(2026-11-04): ROI를 제거한 8카메라 5초 관측 5,569건과 30초 관측 37,023건을 처리했다. Mesh 기반 C 결과는 각각 4,265행·52 ID, 29,835행·87 ID이다. World 좌표가 미확정인 행은 각각 146/744행이며, 30초의 카메라 간 시간 차이 최대 2.051초를 한계로 기록했다. [결과·수정·재현 안내](posetrack-mesh-validation.md), [PDF](handoff/PoseTrack_SCOUT_전체영역_Mesh_검증_20261104.pdf), [전달 ZIP](../deliverables/PoseTrack_SCOUT_full_mesh_20261104.zip)을 참조한다. 단일 평면 H를 전체 영역에서 그대로 사용한 결과가 아니라 Mesh 기하 확장 모델의 결과다. 아래 평면 ROI 설명은 이전 단계의 기록이다.

최신 평면 ROI 검증: SCOUT 메시의 국소 평면(X 32~40, Y 71~79)과 왜곡 보정 입력으로 기존 5초 구간의 추적을 실행했다. ROI 관측이 있는 cam_3~7의 입력 444개에서 출력 416행·Global ID 6개를 얻었다. Geometry·Hungarian·상태·Re-ID·Feature Bank·Global ID의 실제 값을 기록했고, 계측을 끈 원본 실행과 출력 및 프레임별 상태·Bank 개수가 같았다. [내부 계산 보고서](handoff/PoseTrack_SCOUT_평면ROI_내부계산검증.pdf), [전달 ZIP](../deliverables/PoseTrack_SCOUT_flat_ROI_trace.zip), [원시 결과](../outputs/posetrack-flat-20261101/START_HERE.md)를 참조한다. 아래 기존 전체 영역의 보정 BLOCKED 기록을 대체하는 전체 보정 검증은 아니며, 한정된 평면 ROI의 알고리즘 동작 확인이다. 실제 ID 정확도, 다른 ID 교환 및 Missing 재활성화 성공 사례는 미검증이다.

분석 기준은 [공식 저장소 master](https://github.com/ZhenyuX1E/PoseTrack/tree/ba727e6bc6e45e44cd6e006b73e00864892173c2), 커밋 `ba727e6bc6e45e44cd6e006b73e00864892173c2`이다. `README.md`, `requirements.txt`, `script/`, `detection/get_detection.py`, `mmpose/demo/save_pose_with_det_multiscene.py`, `fast-reid/tools/infer.py`, `track/run_tracking_batch.py`, `track/aic_cpp/setup.py`와 실제 import를 확인했다.

## 이미지와 버전

이미지: `scout-mtmc/posetrack-all:ba727e6` (검출·포즈·ReID·추적 공통).

| 구성 | 공식 지정 | 이번 고정값 / 이유 |
|---|---|---|
| Python | 3.9 | 3.9.23 유지 |
| CUDA | pytorch-cuda=11.8 | PyTorch cu118 wheel 런타임 유지 |
| PyTorch / torchvision | 버전 미지정 | 2.0.1+cu118 / 0.15.2+cu118: MMCV 2.0.1 공식 CUDA wheel 조합 |
| NumPy | 1.23.5 | 1.23.5 유지; 원본에 `np.bool` 등 구 API 사용 |
| MMCV | 2.0.1 | 2.0.1 CUDA wheel 유지 |
| MMEngine | >=0.6.0 | 0.10.4 고정 |
| MMPose | >=1.1.0 | 저장소 동봉 코드 1.3.1 editable 설치; 동봉 커스텀 포즈 스크립트 사용 |
| MMDetection | 누락 | 3.1.0 추가: 포즈 API/모델 import에 필요하며 MMCV 2.0.1 호환 |
| FAISS | faiss-gpu, 버전 미지정 | 1.7.2; Python 3.9 GPU wheel 제공 버전 |
| LAP | lap, 버전 미지정 | 0.5.12 고정: Python 3.9 wheel 사용 |
| Cython / pybind11 | 버전 미지정 | 0.29.37 / 2.12.0: 기존 확장 소스 빌드 지원 |
| OS | 미지정 | Debian bookworm Python 공식 이미지; 호스트 OS와 격리 |

나머지 직접 의존성은 `docker/posetrack/requirements.txt`, 빌드 후 모든 설치 버전은 이미지 안 `/opt/scout/installed-requirements.txt`에 기록한다. Python 공식 base 이미지는 `sha256:7bffea15bcc3d7fb87cf10a027986203e4281e078fa2f5b234c30fca291f0834`로 고정했다. 첫 bullseye 이미지의 보안 패키지 다운로드가 HTTP/HTTPS 모두 404로 실패하여 같은 Python 3.9.23의 bookworm 이미지로 변경했다(`logs/posetrack-build*.log`). Python/CUDA 버전을 호스트에 설치하거나 드라이버를 수정하지 않는다. PyTorch wheel이 CUDA 라이브러리를 제공하고 MMCV CUDA 연산은 공식 사전 빌드 wheel로 설치하므로 nvcc를 설치하지 않는다. 다른 CUDA 확장을 추가로 개발하면 devel 이미지가 별도로 필요하다.

TITAN RTX는 Turing/compute capability 7.5이며 CUDA 11.8 실행 대상이다. 확인한 호스트 드라이버 580.126.09는 CUDA 11.8 사용자 라이브러리보다 최신이다. `nvidia-smi`의 CUDA 13.0 표시는 드라이버 지원 상한이며 이 이미지의 CUDA 버전은 11.8이다.

## 단계와 확장

- 검출: 저장소 내 YOLOX, PyTorch, torchvision NMS. 기본 PyTorch 경로를 준비한다. 선택 사항인 torch2trt/TensorRT는 설치하지 않는다.
- 포즈: 동봉 MMPose 1.3.1, MMCV 2.0.1, MMEngine 0.10.4, MMDetection 3.1.0. MMCV의 `_ext` CUDA/C++ wheel 로딩과 NMS를 확인한다.
- ReID: 동봉 FastReID, torchvision ROIAlign, FAISS. `rank_cy`/`roc_cy`를 현재 Python ABI에 맞춰 새로 컴파일한다.
- 추적: `aic_cpp` pybind11 C++ 확장, `cython_bbox`, `lap`, CVXOPT의 GLPK 정수계획 솔버, scipy. `aic_cpp`는 CUDA 확장이 아니다.

단계 사이에 상충하는 OpenMMLab 세대가 없으므로 단일 이미지를 사용한다. 원본 Git 체크아웃은 수정하지 않는다. 저장소에 포함된 다른 Python ABI의 `.so`와 `.pyc`는 Docker 문맥에서 제외하고 소스에서 빌드한다. 가중치와 데이터를 사용하지 않고 네 단계의 실제 진입 파일 import, 확장 작은 연산, PyTorch 행렬 연산, NMS/ROIAlign을 검증한다.

## 빌드와 검증 명령

프로젝트 루트 `scout-mtmc`에서 공통 스크립트를 권장한다. 실행 시 현재 UID/GID, CPU 2개·RAM 8GiB 제한, 네트워크 차단을 적용한다.

```bash
scripts/build.sh posetrack all
scripts/smoke.sh posetrack all --cpu
scripts/smoke.sh posetrack all --gpu 1
```

동일 이미지의 직접 Docker 명령:

```bash
docker build --progress plain -f docker/posetrack/Dockerfile \
  -t scout-mtmc/posetrack-all:ba727e6 .

docker run --rm --runtime=runc --network=none \
  -e NVIDIA_VISIBLE_DEVICES=void --cpus=2 --memory=8g \
  scout-mtmc/posetrack-all:ba727e6 \
  python /opt/scout/smoke.py --device cpu

# 배정받은 GPU의 번호 또는 UUID를 실행 시 전달한다.
GPU_ID=1
docker run --rm --runtime=nvidia --gpus "device=${GPU_ID}" --network=none \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" --cpus=2 --memory=8g \
  scout-mtmc/posetrack-all:ba727e6 \
  python /opt/scout/smoke.py --device cuda
```

GPU 번호는 이미지나 모델 설정에 고정하지 않으며 위 `GPU_ID=1`은 이번에 사용자 배정이 확인된 검증 예시다. 공통 실행 스크립트의 데이터 입력 `/data/scout`는 읽기 전용, 결과 `/outputs`는 쓰기 가능한 별도 마운트를 사용한다. 이 환경변수는 향후 어댑터용 계약이며 기존 PoseTrack 코드가 자동으로 읽지는 않는다.

## 다음 단계의 입력·가중치·주의점

이번 환경 검증은 모델 추론 성공을 의미하지 않는다. 다음 항목을 준비한 후 SCOUT 어댑터와 모델별 입력 변환이 필요하다.

- 검출 `ckpt_weight/bytetrack_x_mot17.pth.tar`, HRNet `td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth`, ReID `aic24.pkl`을 공식 링크에서 팀원이 확인해 준비해야 한다. LUPerson pretrained ResNet은 ReID 학습/가중치 변환 관련 별도 자산이다.
- 기본 입력은 `dataset/test/scene_XXX/camera*/` 형태의 영상과 카메라별 `calibration.json`이다. 단순 영상 경로 연결만으로 SCOUT에 적용되지 않는다. 실제 `track/util/camera.py`는 JSON의 `camera projection matrix`(3×4)와 `homography matrix`(3×3)를 읽고 카메라 폴더명 끝 4자리를 정수로 해석한다. 이를 만들 수 있는 카메라 보정 정보·좌표계, 시간 동기화, 장면·카메라 명세가 필요하다.
- 단계별 출력은 `result/detection`, `result/pose`, `result/reid`, `result/track`; 최종 AIC 제출 형식은 `result/track.txt`다. SCOUT 공통 평가 형식 변환은 아직 구현하지 않았다.
- 원본 shell은 GPU 4~5장, 장면 30개 병렬, CPU affinity와 특정 장면 구간을 가정한다. 2-GPU 서버에서 그대로 실행하지 말고 후속 어댑터 단계에서 명시적인 자원 배정으로 변경해야 한다. 이번 작업에서 실행하지 않았다.
- README의 `script/standard_detection.sh`는 실제 파일명 `script/standrad_detection.sh`와 다르다.
- ReID 코드에는 1920×1080 프레임 가정 및 `.cuda()` 직접 호출이 있다. SCOUT 해상도·촬영 조건과 가중치 직렬화 호환성 검토가 필요하다.
- CPU smoke는 CPU 추론 지원을 보증하지 않는다. 실제 원본 추론 경로는 CUDA 사용을 가정한다.
- 선택 사항인 TensorRT 가속, 학습, 전체 추론, 정확도·속도 평가와 SCOUT 데이터 연결은 수행하지 않는다.

## 검증 결과

2026-10-25 KST에 아래 항목을 실제 실행해 확인했다.

| 항목 | 결과 | 증거 |
|---|---|---|
| 이미지 빌드 | 성공 | `logs/posetrack-build-bookworm.log` |
| 의존성 정합성 | `pip check` 성공 | 위 빌드 로그 |
| 컨테이너 시작·CPU smoke | 성공, 비root UID 1000, 네트워크 차단 | `logs/posetrack-all-cpu-20261025T170848940016.log` |
| GPU 1 CUDA smoke | 성공, TITAN RTX, CUDA 인식 true | `logs/posetrack-all-gpu-20261025T170851427647.log` |
| 실제 모델 추론·학습·성능 평가 | 수행하지 않음 | 가중치·데이터 미연결 |

빌드와 smoke 모두 Python 3.9.23, PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118, MMCV 2.0.1, MMEngine 0.10.4, MMPose 1.3.1, MMDetection 3.1.0, NumPy 1.23.5를 출력했다.

CPU에서는 실제 네 단계 진입 파일과 라이브러리 import, FastReID Cython 확장 로딩, `aic_cpp`·`cython_bbox` 상자 겹침, LAP 할당, GLPK 정수계획, FAISS 최근접 검색, torchvision NMS/ROIAlign, MMCV NMS, 작은 행렬 연산을 통과했다. GPU 1에서는 FAISS GPU index 검색(임시 메모리 16MiB 제한), PyTorch CUDA 행렬 연산, torchvision CUDA NMS/ROIAlign, MMCV CUDA NMS를 통과했다. 가중치 다운로드나 모델 forward는 실행하지 않았다.

완성 이미지 ID는 `sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d`, 크기는 6,607,848,915 bytes(약 6.61GB)다. 설치된 전체 Python 패키지 버전은 `logs/posetrack-all-pip-freeze.txt`에 저장했다. 모든 검증 컨테이너는 실행 후 자동 제거되며 기존 컨테이너를 수정하지 않았다. 원본 `repos/posetrack`의 Git 상태는 깨끗하다.

현재 환경 검증의 미해결 오류는 없다. SCOUT 적응, 실제 가중치 로딩·추론 및 선택적인 TensorRT 경로는 검증 대상 밖이며 아직 미완료다.
# 2026-11-01 담당자 요청서 v3 실행·인계 결과

최신 결과는 [7쪽 PDF 인계서](handoff/PoseTrack_SCOUT_실행검증_인계서_20261101.pdf)와 [전달 ZIP](../deliverables/PoseTrack_SCOUT_handoff_20261101.zip)에 정리했다. 상세 원본은 `outputs/posetrack-transfer-20261101/`이다.

- sequence_1, cam_0~7, source frame277~326(400장), 인원 상한 없이 PoseTrack 고유 검출·HRNet-W32·MGN ReID 추론을 완료했다. 5,569행의 Detection7/Pose57/ReID2048 대응과 유한값 검사를 통과했다.
- 기존 이미지의 GPU1 CUDA·확장·GLPK 및 pip check를 다시 통과했다. 공식 LUP 학습 초기값도 별도로 확보하여 실제 R101_reprod 설정의520개 키 로드를 확인했다. LUP는 aic24.pkl 추론의 대체 파일이 아니다.
- native Tracking과 실제 ID 결정 trace는 **BLOCKED**다. 실제 검출 하단665개 중597개가 SCOUT 지면 메시와 교차했으며, 같은 ray의 공통 PCA 평면 교점과 비교한 카메라별 p95 거리는2.568~11.811 dataset unit이었다. raw 왜곡 좌표·비평면 지면·물리 단위의 문제가 남아 native calibration.json을 생성하지 않았다.
- `05_analysis/code_flow.csv`는 정적 소스 분석이고 `input_gate_counts.csv`는 입력 품질 조건 개수다. 실제 추적 분기·bank 저장·Global ID 결정 로그가 아니다. HOTA 등 성능 평가는 수행하지 않았다.

