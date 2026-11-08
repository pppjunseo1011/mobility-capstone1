# SCOUT 제공 정보와 세 모델 입력 비교·적용 검토

2026-10-26 KST. 오늘 작업의 두 항목은 **제공 정보와 입력 요구사항 대조**, **공통 로더·모델별 변환 구현 및 추가 정보의 생성·적용 가능성 검토**다. 이 문서는 실제 모델 추론이나 성능 평가 완료 보고가 아니다.

## 1. 확인 범위와 근거

| 대상 | 실제로 확인한 범위 | 아직 확인하지 않은 범위 |
|---|---|---|
| SCOUT | 공식 로더·기하 코드, 배포 README, 실제 보정/타임스탬프, COCO 제한된 앞부분 | 전체 이미지 디코딩, 전체 GT, 실제 3D mesh, 세계 단위·유효 지면 |
| 모델 | 고정 커밋의 실제 입력 코드·중간 파일 계약·추적 규칙 | 가중치 로딩, 모델 forward, 실제 추적 결과 |
| 어댑터 | 공통 메타데이터/프레임 로더, 3개 프로파일, 영상/결과/GT 변환, 합성 검증 | 실데이터 전체 적용, 시간 재동기화, 왜곡 및 결과 좌표 역변환, 모델 실행 연결 |

기준 소스는 [sources.lock.json](../sources.lock.json)의 YACHIYO `main@f881fe0`, UWIPL `main@8e23a69`, PoseTrack `master@ba727e6`이다. SCOUT toolkit은 [scout-format.lock.json](../adapters/scout-format.lock.json)의 `1acd79513bfcf16aecef1150283af58c8273542e`에 고정했다. 새로운 upstream 변경을 자동 반영한 분석이 아니다.

공식 [로더](https://github.com/cvlab-epfl/scout_toolkit/blob/1acd79513bfcf16aecef1150283af58c8273542e/src/loading.py)와 [기하 코드](https://github.com/cvlab-epfl/scout_toolkit/blob/1acd79513bfcf16aecef1150283af58c8273542e/src/geometry.py)를 배포 파일과 대조했다. 문서의 YAML 설명과 달리 실제 보정은 JSON이며, 로더에 나온 카메라 키를 timestamp 파일에 그대로 적용할 수도 없다. 따라서 공식 설명, 실제 관찰, 아직 검증하지 않은 가정을 구분한다.

## 2. SCOUT가 제공하는 정보

| 정보 | 형식·의미 | 확인 수준과 변환 시 주의점 |
|---|---|---|
| 촬영 영상 | 사전 추출 JPEG: `images/sequence_N/cam_N/image_F.jpg` | 배포 README는 1920×1080·10FPS로 설명. 실제 전체 픽셀 검증은 미실행. 원본 MP4 제공을 가정하지 않음 |
| 카메라 보정 | `calibrations/cam_N.json`의 K, R, T, dist, view_id | 실제 25개 확인. R,T는 world→camera. 모든 확인 카메라에 비영 왜곡 계수 존재 |
| 촬영 시각 | `timestamps/timestamps_sequence_N.json` | 각각 11,960개 frame 키. sequence 1은 장치명, sequence 2는 camN 키. 중복 timestamp와 카메라별 시각 차이 존재 |
| 카메라 가용성 | 보정 25개, timestamp는 sequence 1에서 25개·sequence 2에서 22개 | sequence 2의 cam_1/3/4에 timestamp가 없음. 파일이 없다는 사실과 실제 영상이 없다는 사실은 구분 |
| 사람 정답 | COCO/MOT의 bbox, 사람 ID, world 좌표 | COCO category_id는 사람 ID이며 0도 유효. GT는 모델 검출값이 아님. 수동 GT/pseudo-label 범위·품질, sequence 간 ID 동일성은 미확인 |
| 3D 장면 | 배포 README에 GLB/PLY/PNG 및 ground-only mesh 제공 명시 | mesh 파일 자체·좌표 정합성은 미검증. 제공된다고 해서 단일 평면 H가 바로 성립하는 것은 아님 |

`COCO date_captured`를 임의로 초 단위 촬영 시각으로 해석하지 않고 timestamp JSON을 사용한다. 누락 timestamp를 FPS로 추정해 채우지 않는다.

## 3. 모델별 입력 요구사항 비교

| 항목 | YACHIYO | UWIPL | PoseTrack |
|---|---|---|---|
| 장면·카메라 폴더 | `Original/scene_001/camera_0004/` | `data/test/S001/c004/` | `dataset/test/scene_001/camera_0004/` |
| 실제 이미지/영상 입력 | 검출·ReID는 Frame의 JPEG, 포즈는 video.mp4 | img/frame JPEG, 포즈는 video.mp4 | 검출·포즈·ReID 모두 video.mp4; frames/는 어댑터 중간 배치 |
| 선택 구간 첫 frame | 1 | 0 | 0 |
| 카메라 대응 | SCOUT cam_N → camera_(N+1) | cam_N → c(N+1); S001은 경로 별칭 | cam_N → camera_(N+1) |
| 검출 중간 결과 | camera,frame,class,xyxy,score CSV 등 | 8열 검출 TXT 및 원본 행 순서 | frame,class,xyxy,score 7열 TXT |
| 포즈·ReID | HRNet 포즈 JSON, OSNet 특징 | 포즈/발 위치, S001 경로의 여러 ReID NPY | 57열 포즈 TXT, 검출 행별 ReID NPY |
| 공간 정보 | P/H 및 세계 XY를 사용하는 추적 | SCT에는 검출·특징, STCRA에는 하드코딩 H·세계 거리 규칙 | P/H, 지면 및 0.15 높이의 발 평면 |
| 시간 가정 | frame 동일성, 기본 fps=30와 고정 frame 임계값 | 강제 fps=30·S001 기준 frame/장면 규칙 | frame 동일성·frame 단위 상태 유지 |
| 최종 결과 파서 | yachiyo-json / aic9 | mot10 / mot16 / aic9 | posetrack10 / aic9 |

검출·포즈·ReID 파일은 **SCOUT에서 누락된 원천 정보를 보충하는 파일이 아니라 모델이 예측해야 하는 중간 결과**다. 실제 가중치와 모델 실행이 필요하다. GT를 그 자리에 넣으면 실험 조건이 달라진다.

세 모델의 1920×1080 하드코딩은 SCOUT의 선언 해상도와 일치할 수 있다. 실제 선택 영상이 이 크기이고 원본 test size를 유지하는지 확인해야 하며, 항상 수정이 필요하다고 단정하지 않는다. 크기·crop·왜곡 처리·test size가 바뀌면 좌표 변환도 함께 검토한다.

상세 코드 근거: [YACHIYO](adapters/yachiyo.md), [UWIPL](adapters/uwipl.md), [PoseTrack](adapters/posetrack.md).

## 4. 공통 로더와 모델별 변환 구현

```text
SCOUT 이미지·보정·타임스탬프
  └─ dataset.py: 공통 메타데이터/프레임 로드·검사
       ├─ inspect: 데이터 구성 확인
       ├─ assess: 실제 가용 정보·시간 차이·모델 요구사항 비교
       └─ export.py + profiles/<model>.py
            └─ 모델별 JPEG/MP4 + manifest + frame_map + report
                  └─ 모델별 실행 [연결·추론은 후속 단계]
                       └─ results.py: 원본 번호 복원 → predictions.csv

SCOUT GT ─ annotations.py + ground-truth → ground_truth.csv
```

- **공통 로더**: SCOUT root 탐색, JSON 보정 검사, 카메라·timestamp 별칭 대응, JPEG frame 숫자 정렬·선택, 해상도 헤더 확인, 원본 시각·프레임 보존. 학습용 PyTorch DataLoader나 전체 이미지 텐서 캐시는 이번 범위가 아니다.
- **모델별 변환**: 3개 프로파일이 폴더·카메라·프레임 시작 번호를 정한다. `manifest.json`과 `frame_map.csv`에 원본 대응을 남긴다. `export --videos --fps`는 MP4를 만들고 디코딩 프레임 수·크기를 검사한다.
- **결과 변환**: 모델별 결과를 원본 camera/frame/timestamp에 대응시키고 공통 CSV로 저장한다. camera-local ID와 sequence 범위의 cross-camera ID를 구분한다. 어댑터가 사람의 ID를 새로 연결하지는 않는다.
- **정답 분리**: COCO/MOT를 별도 GT로 변환한다. COCO는 큰 파일을 객체 단위로 읽고, 선택 이미지가 주석 인덱스에 없으면 미라벨 구간을 빈 정답으로 해석하지 않도록 실패한다.
- **이번 보완**: `assess`에서 누락 카메라를 숨기지 않고 비교 가능한 카메라·frame 범위를 기록한다. 1개 카메라만 검사한 경우 다중 카메라 동기화 통과로 표시하지 않는다. 공통 로더는 sparse frame 관측 사이의 시간 역행도 검사하며, `01` 같은 비정규 frame 키를 거부한다.

모델용 환경은 기존 9개를 유지한다. 어댑터 v0.2는 별도 `scout-mtmc/adapter:0.2`를 사용하고 v0.1 이미지는 보존한다. 코드·모델 의존성을 호스트에 추가 설치하지 않는다.

## 5. 추가 정보의 생성 및 적용 가능성

| 분류 | 항목 | 현재 구현·적용 상태 |
|---|---|---|
| 기계적 생성 가능 | 모델 입력 폴더·파일 이름, camera/frame 대응표, 원본 timestamp | 구현됨. 실제 JPEG 확보 후 해당 범위 export 가능 |
| 기계적 생성 가능 | 연속 JPEG → MP4, 모델 결과 → 공통 CSV, 별도 GT CSV | 구현됨. MP4의 지정 FPS는 재동기화가 아님. 결과 변환에는 실제 예측 파일이 필요 |
| 참고 계산 가능 | `P=K[R|T]` | 이미 계산. pinhole 참고값이며 raw 픽셀에서 렌즈 왜곡까지 표현하지 않음 |
| 조건부 생성 | 지면 H, 왜곡 보정 영상/좌표, 시간 정렬된 프레임 집합 | 지면·단위·좌표계·시간 정책 검증 후 구현/적용. 현재 자동 생성하지 않음 |
| 조건부 설정 | ROI, 카메라 연결, FPS·거리·track 유지 임계값 | 실제 SCOUT 장면과 공통 실험 조건에서 결정. AI City 전용 값을 그대로 복사하지 않음 |
| 모델 실행 필요 | 검출 bbox/score, keypoint, ReID embedding, SCT·global ID | 검출·포즈·ReID·추적 실행으로 생성. 정답이나 임의 값으로 보충하지 않음 |
| 외부 준비 필요 | 정확한 학습 가중치·모델 설정 | 팀원이 체크포인트 종류·파일·해시를 확정. ImageNet fallback 등을 동일 모델로 간주하지 않음 |

### 공간 보정의 적용 조건

실제 지면이 world 좌표계의 `Z=z0`인 평면이고, 왜곡 없는 영상에 대응하는 내부행렬을 `K'`라고 할 때 `H=K'[r1 r2 T+z0*r3]`를 구성할 수 있다. 이는 **조건을 만족할 때의 수학적 구성 방법**이며, 현재 SCOUT 지면을 `Z=0`, 단위를 m로 확인했다는 뜻이 아니다.

원본 영상에는 비영 왜곡이 있으므로 raw pixel에 단일 H만 적용하면 안 된다. 다음 두 방향을 검토할 수 있다.

1. 영상을 undistort하고 K'/P/H, bbox·keypoint, GT 및 최종 결과 좌표를 모두 같은 기준으로 처리한다. 현재 결과 변환기는 frame 번호만 역매핑하므로 원본 좌표로 돌아가는 변환이 추가로 필요하다.
2. raw 영상·모델 예측을 유지하고, 위치 계산에 사용할 점만 undistort한 뒤 검증된 평면 또는 mesh와 교차시킨다. 공식 toolkit에 ray–mesh 함수가 있으나 실제 mesh와 각 모델의 세계좌표 입력에 연결하는 작업은 미구현이다.

비평면 장면을 mesh로 처리하면 단일 H를 전제로 한 추적기의 변경이 필요할 수 있다. 이는 파일 포맷 변환보다 넓은 모델 적용 변경이므로 비교 연구에서 변경 내용을 기록해야 한다. PoseTrack은 `H[:,2] + 0.15*P[:,2]`를 사용하므로 P/H를 서로 독립적으로 정규화해서도 안 된다.

### 시간 정렬의 적용 조건

먼저 선택 카메라와 평가 구간, 기준 시간축/FPS, 허용 오차, 중복·누락 정책을 정한다. 그 후 timestamp 기반 매칭/리샘플링과 새로운 frame map을 만들고 이미지·GT·모델 결과가 같은 대응을 사용하도록 해야 한다. 현재 `assess`의 기본 0.05초는 **진단 설정값**이며 최종 연구 기준이 아니다. 전체 카메라의 최대 오차가 크다고 해서 모든 두 카메라 구간이 부적합한 것도 아니므로 실제 비교할 카메라·구간을 다시 검사한다.

## 6. 모델별 중요한 후속 연결 항목

### YACHIYO

- 검출 CSV는 1기준이고 포즈 video enumerate는 0기준이다. 원본을 유지하는 대안은 **포즈 전용 CSV의 frame−1 → 포즈 추론 → 포즈 JSON key+1** 변환이다. JSON 번호만 사후 변경하면 잘못된 영상에서 추론한 포즈는 고칠 수 없다. 이 bridge는 아직 미구현이다.
- SCT와 MCPT를 분리해 외부 단계에서 검증된 세계 XY를 채우는 보정 대안도 있다. 기본 MCPT는 WorldCoordinate를 사용하므로 보정 없이 전체 파이프라인이 성공한다고 볼 수 없다.
- 정확한 OSNet checkpoint가 없으면 vendor가 ImageNet pretrained 경로를 선택할 수 있다. 실행 전 체크포인트 존재·종류·해시를 확인해야 한다.

근거: [포즈 영상 frame](../repos/yachiyo/poser/top_down_video_demo_with_track_file.py), [포즈 매칭](../repos/yachiyo/tracking/src/pose.py), [세계좌표 처리](../repos/yachiyo/tracking/src/utils.py), [MCPT](../repos/yachiyo/tracking/src/mcpt.py). 세부 위치는 모델별 문서와 원본 커밋에 고정한다.

### UWIPL

- `S001/cNNN`은 입력 별칭일 뿐 S001 알고리즘이 SCOUT에 적용된다는 뜻이 아니다. c001..c007에만 있는 threshold와 c005 ROI, 특정 인물·ID에 대한 규칙을 검토해야 한다.
- S001 clustering은 anchor `[0,5000,...]`에서 `frame-1`을 조회한다. 0부터 시작하는 32프레임 clip에는 그 조회점들이 없어 원본 clustering을 그대로 실행할 수 없다. 짧은 clip은 우선 데이터 연결과 단계별 추론을 검증하는 용도이며 전체 MTMC 성공 기준을 적용하려면 추적 규칙의 일반화가 필요하다.
- STCRA는 같은 frame/track ID의 카메라 관측을 묶는다. camera-local SCT 결과를 바로 넣지 않고 global ID가 부여된 결과를 사용해야 한다.
- H는 외부 JSON에서 자동 로드되지 않고 `STCRA/cam_utils.py`에 하드코딩되어 있다. 보정 생성과 실행 연결은 별도 작업이다. 원본 ReID frame 0 초기화·카메라 전환 캐시 오류도 남는다.

근거: [SCT·카메라 규칙](../repos/uwipl/BoT-SORT/tools/aic_track.py), [S001 clustering](../repos/uwipl/BoT-SORT/tools/aic_hungarian_cluster_S001.py), [STCRA](../repos/uwipl/STCRA/run_stcra.py), [H](../repos/uwipl/STCRA/cam_utils.py).

### PoseTrack

- JPEG staging만으로 실행되지 않으며 video.mp4가 필요하다. 장면 선택 및 저장소 상대 경로를 실행 workspace에 연결해야 한다.
- P/H·0.15 발 높이 평면·세계 단위·왜곡 처리의 일관성을 검증해야 한다.
- 검출·포즈·ReID는 동일 행 순서로 결합된다. 중간 단계에서 임의로 정렬·삭제하면 다른 사람의 특징을 결합할 수 있다. 빈 결과·한 행 입력 처리도 실제 연결 때 검증한다.

근거: [카메라 보정](../repos/posetrack/track/util/camera.py), [중간 결과 결합](../repos/posetrack/track/run_tracking_batch.py), [추적기](../repos/posetrack/track/Tracker/PoseTracker.py).

## 7. 실행 및 산출물

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 scripts/adapter.py build
python3 scripts/adapter.py test

# 작은 공식 메타데이터만으로도 3개 모델 요구사항 대조 가능
python3 scripts/adapter.py assess \
  --data datasets/scout/reference-metadata \
  --output outputs/scout-compatibility-sequence1-new \
  -- --sequence 1 --sync-tolerance 0.05

# 실제 이미지를 준비한 뒤 선택 구간 입력 생성
python3 scripts/adapter.py export \
  --data /실제/SCOUT/경로 --output outputs/scout-posetrack-clip-new \
  -- --model posetrack --sequence 1 --cameras cam_3 cam_4 \
     --max-frames 32 --videos --fps 10
```

`assess`는 `compatibility.json`과 읽기 쉬운 `comparison.md`를 새 출력 폴더에 만든다. 원본 데이터는 읽기 전용이며 기존 출력 폴더를 덮어쓰지 않는다. 영상·모델 보정·추론·가중치 다운로드는 이 명령의 부수 효과로 실행되지 않는다.

완료 및 미완료를 보고할 때는 **“공통 로더·모델별 1차 변환 구현과 생성 가능성 검토 완료, 실제 모델 적용은 조건 확인 및 후속 연결 필요”**라고 구분한다. 실제 검증 수치와 산출물 경로는 [어댑터 문서](adapter.md)의 2026-10-26 기록에 남긴다.
