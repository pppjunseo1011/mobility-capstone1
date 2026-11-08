# SCOUT 1차 어댑터

이 구현은 **추출된 SCOUT → 모델별 입력 폴더/번호 대응표**, **모델 결과 → SCOUT 원본 좌표계의 공통 CSV**를 제공한다. 모델 가중치 추론이나 성능 평가를 실행하지 않는다. 입력 배치 성공과 전체 추적 파이프라인 실행 가능 여부는 별개이며, 생성하는 `report.json`의 `model_inference_ready`는 아직 `false`다.

**v0.3(2026-10-27):** `select-window`, OpenCV 기준 영상 생성, `yachiyo-pose-input` / `yachiyo-pose-output`을 추가했다. 기존 어댑터 기능을 포함한 컨테이너 검사 95개를 통과했다. [연속 20초 구간 검증과 새 명령](scout-next-stage.md).

**v0.4(2026-10-28):** `geometry-probe`와 `project-ground`를 추가했다. 원본 픽셀을 왜곡 보정한 뒤 실제 지면 메시와 교차시킨다. 전체 검사 119개를 통과했다. 모델용 H나 실제 추론은 포함하지 않는다. [기하 기능·CSV 형식·검증 결과](scout-ground-projection.md).

**현재 v0.5(2026-10-31 검증):** `model-ground-input`과 `model-ground`로 세 모델의 중간 파일을 원본 프레임·픽셀 및 지면 좌표 후보에 연결한다. 전체 자동 검사 149개와 실제 기하 + 합성 모델 자료 24건 검사를 통과했다. [모델 출력 연결 계약·사용법](scout-model-ground.md).

## 실제로 확인한 SCOUT 1.0 형식

공식 toolkit [loading.py](https://github.com/cvlab-epfl/scout_toolkit/blob/1acd79513bfcf16aecef1150283af58c8273542e/src/loading.py), [geometry.py](https://github.com/cvlab-epfl/scout_toolkit/blob/1acd79513bfcf16aecef1150283af58c8273542e/src/geometry.py)와 실제 배포 메타데이터를 대조했다. 기준 커밋과 파일 해시는 [scout-format.lock.json](../adapters/scout-format.lock.json)에 기록한다. 최초 구현 시에는 작은 메타데이터 묶음과 압축 파일 앞부분만 확인했다. **2026-10-27 전체 압축 해제 후에는 실제 JPEG 목록·보정/시간 파일과 선택한 실제 이미지·MOT/COCO 주석까지 추가 검증했다.** [최신 범위와 결과](scout-real-validation.md).

```text
scout/
  images/sequence_1/cam_0/image_0.jpg
  calibrations/cam_0.json
  timestamps/timestamps_sequence_1.json
  annotations/coco_format/annotations_coco_sequence_1.json
  annotations/mot_format/sequence_1/cam_0.txt
```

이미지·MOT 경로는 공식 로더를 기준으로 구현했고 실제 표본에서 읽기를 확인했다. 전체 JPEG 파일명 562,427개를 조사했지만 전체 픽셀과 전체 GT 정확도를 검증한 것은 아니다. 입력으로 `scout/` 자체 또는 이를 포함한 부모 폴더를 지정할 수 있다. 현재 실제 경로는 `datasets/scout/extracted/scout`이다.

실제 파일에서 확인한 차이와 주의점:

- 배포 README는 보정을 YAML이라고 소개하지만 실제 보정은 **JSON의 `K`, `R`, `T`, `dist`, `view_id`**다. `R,T`는 world→camera이며 P는 `K[R|T]`다. 모든 확인 카메라의 왜곡 계수는 0이 아니다. 계산한 P는 pinhole 참고값으로만 저장하고 모델용 H를 만들지 않는다.
- sequence 1 타임스탬프는 `frame 문자열 → cvlabrpi 장치명 → 초`다. 예를 들어 `cam_0`의 `view_id=cvlabrpi10_0`를 `cvlabrpi10`에 대응시킨다. sequence 2는 `cam0`처럼 밑줄 없는 키를 사용한다. 이름 대응 규칙과 각 카메라의 선택된 `timestamp_key`를 기록하며 필요하면 `--timestamp-map`으로 명시한다.
- 보정 파일은 25개다. 두 timestamp 파일은 각각 11,960개 frame 키를 가지며, sequence 2의 `cam_1`, `cam_3`, `cam_4`는 timestamp가 없다. `inspect`는 이를 보고하고, `export`에 그 카메라를 선택하면 실패한다.
- timestamp가 같은 프레임들이 실제로 존재한다. 중복 시간은 삭제하지 않고 표시하며 시간 역행은 거절한다. 같은 frame 번호라도 카메라 시간이 일치하지 않을 수 있어, `sync.max_skew_seconds`와 허용값 초과 여부를 기록한다. 시간 보간·리샘플링은 하지 않는다.
- 배포 README는 1920×1080, 10FPS라고 설명하며 COCO 이미지 메타데이터에서도 해당 크기를 확인했다. 어댑터는 실제 선택 JPEG의 헤더 크기를 읽고 각 카메라의 시간 차이에서 FPS를 추정한다. 헤더 검사만으로 전체 이미지 픽셀의 정상성을 보증하지 않는다.
- COCO의 `category_id`는 일반적인 클래스 ID와 달리 **사람의 ID**다. ID `0`도 유효하다. 실제 첫 annotation은 `image_id="10300000"`, `category_id=0`, xywh bbox, 3원소 world 좌표였다. `file_name`에는 sequence가 생략되어 있어 이 값을 경로로 그대로 신뢰하지 않는다.
- COCO `date_captured`는 timestamp JSON과 단위·표현이 다르다. 어댑터는 이 값을 초로 추정하지 않고 timestamp JSON을 사용한다. COCO 파일 하나가 약 902MB여서 GT 변환은 객체 단위 스트리밍으로 읽는다.
- 사람 ID의 sequence 간 동일성은 미확인이다. 공식 논문과 UMPN 코드에서 수동 정리 범위를 **sequence_1의 cam_0~cam_7**로 확인했고 사용자도 이 범위를 지정했다. 수동 정리는 1FPS, 이후 선형 보간은 10FPS다. 일반 변환기의 개별 행 `label_quality=unverified`는 그대로 유지하고, 조사한 범위·출처는 별도 연구 기록에 남긴다. 미라벨 카메라를 빈 GT로 해석해 평가하면 안 된다.
- 실제 COCO는 주석 본문에 행이 있어도 `images` 목록에 해당 이미지가 없는 경우가 있다. 현재 선택 구간은 64장 중 48장이 누락되어 엄격 변환이 거부된다. 인덱스를 추측해 보충하지 않으며, 검증 가능한 MOT 형식으로 1,401행을 내보냈다.

## 구성과 실행 환경

`adapters/scout_adapter/`의 메타데이터·번호 변환기는 Python 표준 라이브러리만 사용한다. 현재 CPU 전용 이미지 `scout-mtmc/adapter:0.5`는 이미 구축·검증한 PoseTrack 이미지의 Python 3.9, OpenCV 4.8.1, ffmpeg, 테스트용 Pillow 레이어를 재사용한다. v0.1/v0.2/v0.3/v0.4와 기존 9개 모델 이미지는 보존한다. 새 모델 의존성을 설치하지 않으며, 이미지 재빌드 전에 기존 기반 이미지 ID가 기록과 같은지 확인한다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 scripts/adapter.py build
python3 scripts/adapter.py test
```

실행 래퍼는 네트워크 차단, GPU 미노출, 현재 UID/GID, CPU 2개·메모리 4GiB를 사용한다. SCOUT와 기존 export/결과 파일은 읽기 전용으로 연결하고 새 출력만 별도 쓰기 공간에 만든다. 출력은 기존 폴더를 덮어쓰지 않으며, 실패 시 불완전한 export를 최종 폴더로 남기지 않는다. `--dry-run`은 파일을 생성하지 않고 실행 명령만 보여준다.

## 1. 포맷 검사

이미 확인한 작은 메타데이터만으로도 검사할 수 있다. 이미지가 아직 없다는 상태가 명시된다.

```bash
python3 scripts/adapter.py inspect \
  --data datasets/scout/reference-metadata \
  --output outputs/metadata-inspection-new \
  -- --sequence 1
```

전체 파일 다운로드·무결성 확인 후 압축을 별도로 풀고 `--data`를 실제 SCOUT 루트로 지정한다. 첫 검사는 일부 카메라로 시작한다. sequence 2는 예를 들어 `--cameras cam_0 cam_2`처럼 timestamp가 있는 카메라를 선택한다.

### 요구사항 비교·생성 가능성 검사 (`assess`, v0.2)

```bash
python3 scripts/adapter.py assess \
  --data datasets/scout/reference-metadata \
  --output outputs/scout-compatibility-new \
  -- --sequence 1 --sync-tolerance 0.05
```

`compatibility.json`과 `comparison.md`에 실제 가용 카메라·JPEG 존재 여부·보정·촬영 시각 진단과 세 모델 입력 요구사항을 함께 기록한다. 모델용 P/H나 가중치·추론 결과를 생성하지 않는다. `--cameras`, `--timestamp-map`도 기존 검사와 동일하게 사용할 수 있다. [종합 비교와 적용 가능성 검토](scout-model-compatibility.md).

같은 frame의 시각 차이는 **비교 가능한 카메라들의 timestamp가 모두 있는 frame**에서 계산한다. 누락 카메라와 frame 수를 별도로 보존하며, 카메라가 하나뿐이거나 선택한 카메라가 누락되면 전체 동기화 통과를 선언하지 않는다. `--sync-tolerance`는 진단 임계값으로, 최종 연구 평가 기준은 아니다. 보간·리샘플링·동기화는 수행하지 않는다.

공통 로더는 v0.2에서 `01`, `1e0` 같은 비정규 frame 키를 거부한다. 시간 역행과 중복은 sparse frame 사이의 인접 관측까지 검사하고, FPS 추정에는 frame 번호가 연속인 양의 시간 차이만 사용한다.

## 2. 모델 입력 배치

기본은 카메라당 최대 32프레임이다. 파일명은 숫자 순서로 처리하며 선택한 첫 카메라의 frame 집합을 다른 카메라도 모두 가져야 한다. 누락 프레임을 조용히 삭제하거나 GT로 검출 결과를 채우지 않는다. `--start-frame`은 원본 frame ID의 하한이고 `--max-frames`는 개수다.

```bash
SCOUT_DIR=/실제/SCOUT/경로

python3 scripts/adapter.py export \
  --data "$SCOUT_DIR" --output outputs/scout-yachiyo-preview \
  -- --model yachiyo --sequence 1 --cameras cam_3 cam_4 --max-frames 32

python3 scripts/adapter.py export \
  --data "$SCOUT_DIR" --output outputs/scout-uwipl-preview \
  -- --model uwipl --sequence 1 --cameras cam_3 cam_4 --max-frames 32

python3 scripts/adapter.py export \
  --data "$SCOUT_DIR" --output outputs/scout-posetrack-preview \
  -- --model posetrack --sequence 1 --cameras cam_3 cam_4 --max-frames 32
```

| 모델 | 생성하는 프레임 경로 예시 | 모델 프레임 시작 |
|---|---|---|
| YACHIYO | `Original/scene_001/camera_0004/Frame/000001.jpg` | 1 |
| UWIPL | `data/test/S001/c004/img/00000.jpg` 및 `frame/00000.jpg` | 0 |
| PoseTrack | `dataset/test/scene_001/camera_0004/frames/000000.jpg` | 0 |

카메라의 모델 ID는 `원본 cam_N의 N + 1`이다. 원본 번호가 0이라고 모델 ID 0으로 바꾸지 않는다. PoseTrack의 `frames/`는 어댑터 중간 산출물이며 **원본 모델은 video.mp4를 읽는다**.

프레임을 복사하지 않고 manifest/폴더 계획만 확인하려면 `--materialize manifest`를 추가한다. 기본 복사는 원본 픽셀과 해상도를 유지하고, symlink/hardlink로 원본에 쓰기 경로를 만들지 않는다. UWIPL은 두 프레임 디렉터리에 같은 파일을 복사하므로 추가 공간을 사용한다.

영상까지 만들려면 명시적으로 `--videos --fps 10`을 추가한다. v0.3의 기본 `--video-decoder opencv`는 OpenCV로 읽은 BGR24 픽셀을 FFmpeg lossless RGB H.264 MP4로 인코딩한다. 프레임 수·해상도는 ffprobe로 확인하고 픽셀 일치는 별도 실제 검증으로 확인한다. 이전 FFmpeg JPEG 디코더는 `--video-decoder ffmpeg`로 재현한다. 원본 프레임 ID가 연속되지 않으면 영상 생성을 거부한다. 지정 FPS는 영상의 재생 시간축이며 카메라 timestamp를 보정하거나 반복 프레임을 제거하지 않는다.

각 export에 다음 파일이 생긴다.

- `manifest.json`: 원본 root·sequence·카메라·frame·timestamp·크기·보정·동기 오차
- `frame_map.csv`: 모델 frame과 원본 frame의 양방향 대응
- `profile.json`, 모델별 config: 입력 경로와 원본 코드 요구사항. 일부 config는 설명용이며 모델이 자동으로 읽는 파일이 아니다.
- `metadata/cam_N.json`: 원본 보정과 pinhole 참고 P. 모델용 calibration.json/H는 생성하지 않는다.
- `report.json`: 입력 배치 결과, 영상 검사 결과, 실제 추론을 막는 항목

## 3. 추적 결과 정규화

모델이 **해당 export의 정확한 카메라·frame 번호로 생성한 결과**만 넣는다. 전체 결과와 일부 clip의 frame map을 섞으면 안 된다. 모델에 resizing/undistortion을 별도로 적용했다면 그 좌표 역변환을 추가한 뒤 사용해야 한다.

```bash
# YACHIYO: Frame 1기준 JSON → 원본 SCOUT frame
python3 scripts/adapter.py normalize \
  --workspace outputs/scout-yachiyo-preview \
  --input /실제/fixed_whole_tracking_results.json \
  --output outputs/common-yachiyo \
  -- --format yachiyo-json --id-scope sequence

# UWIPL SCT: 카메라별 ID이며 다중카메라 global ID로 취급하지 않음
python3 scripts/adapter.py normalize \
  --workspace outputs/scout-uwipl-preview \
  --input /실제/S001_c004.txt --output outputs/common-uwipl-sct \
  -- --format mot10 --camera cam_3 --id-scope camera

# PoseTrack: 마지막 열 position_source는 confidence가 아님
python3 scripts/adapter.py normalize \
  --workspace outputs/scout-posetrack-preview \
  --input /실제/scene_001.txt --output outputs/common-posetrack \
  -- --format posetrack10 --id-scope sequence
```

지원 결과: YACHIYO `yachiyo-json`/`aic9`, UWIPL `mot10`/`mot16`/`aic9`, PoseTrack `posetrack10`/`aic9`. `--id-scope sequence`는 실제로 cross-camera ID가 부여된 결과에만 사용한다. 각 sequence를 별도 파일로 변환하며, 여러 scene을 합친 파일에서 scene 정보를 추측하지 않는다.

공통 `predictions.csv`는 원본 카메라·frame·초 단위 timestamp·xywh 픽셀 좌표·이름 공간을 포함한 ID를 저장한다. 예: `posetrack:sequence_1:7`; camera-local이면 `uwipl:sequence_1:cam_3:7`. 원본 숫자 ID도 `native_track_id`에 보존한다. 알 수 없는 camera/frame, 비유한 값, 잘못된 bbox, 같은 camera/frame/ID 중복은 거부한다. 없는 confidence나 world z는 빈 값이며 발 위치 방식 값을 confidence로 사용하지 않는다. 이것은 비교 연구용 중간 규격이며 평가 지표 계산은 별도 작업이다.

## 4. GT는 별도 출력

```bash
python3 scripts/adapter.py ground-truth \
  --data "$SCOUT_DIR" --workspace outputs/scout-posetrack-preview \
  --output outputs/scout-gt-preview \
  -- --format coco
```

`--format mot`도 지원한다. COCO에서는 선택한 이미지가 주석의 image index에 없으면 실패하여 미라벨 구간을 빈 GT로 오인하지 않는다. MOT의 빈 프레임은 해당 형식만으로 미라벨/무인 구간을 구별할 수 없어 coverage를 미확인으로 유지한다. GT는 별도 `ground_truth.csv`로만 만들고 모델 입력 디렉터리의 Detection/feature/pose를 채우지 않는다. ID·label quality 확인 전 실제 성능 평가에 사용하면 안 된다.

## 검증 결과 — 2026-10-25 KST

- CPU Docker 이미지 빌드 성공: `scout-mtmc/adapter:0.1`, 이미지 ID `sha256:074946d59cd207ce915039bce44e187ae0328d3e04dc5abbedf8774f9c8b7dae`.
- 합성 입력의 통합·GT 스트리밍 테스트 **42개 통과, 실패/skip 0개**. 세 프로파일 입력 배치, 0/1기준 프레임 역변환, sequence/camera ID 범위, COCO person ID 0, 잘못된 결과 거부, 원본 보호와 실패 정리, 실제 소형 MP4 인코딩·디코딩을 포함한다.
- 공식 보정·타임스탬프를 읽기 전용으로 연결한 Docker 검사 성공. sequence 1의 25개, sequence 2의 22개 timestamp 카메라를 확인했으며 결과 파일 소유자는 현재 UID 1000이다.
- 기존 환경 파일 25개의 해시와 모델 저장소 3개의 clean 상태를 확인했다. 기존 Docker 환경 검사를 다시 모델 추론으로 해석하지 않는다.
- 실제 SCOUT 이미지 전체 변환, 원본 모델 가중치 추론, 평가 지표 검증은 **미실행**이다.

증거: [빌드 로그](../logs/adapter-build.log), [테스트 로그](../logs/adapter-tests.log), [검증 기록 및 구현 파일 해시](../logs/adapter-validation.json). 실제 메타데이터 검사 결과는 `outputs/adapter-metadata-docker-sequence1/inspection.json`, `outputs/adapter-metadata-docker-sequence2/inspection.json`에 있다.

## v0.2 추가 검증 — 2026-10-26 KST

- 새 CPU 이미지: `scout-mtmc/adapter:0.2`, ID `sha256:d7d832323f7ce8889bbf1ef921df0598daf360a5a603b448d11bcd811c742a21`.
- Docker 합성 테스트 **54개 통과, 실패/skip 0개**: 기존 42개와 새 비교·누락·시간 진단·출력 보호 테스트 12개.
- 공식 보정·타임스탬프를 읽기 전용으로 연결한 `assess`를 sequence 1/2에 실행했다. 입력은 metadata-only 경로이므로 JPEG/GT 부재를 SCOUT 전체 배포의 부재로 해석하지 않는다.

| 진단 항목 | sequence 1 | sequence 2 |
|---|---:|---:|
| 선택 보정 카메라 | 25 | 25 |
| timestamp 비교 가능 카메라 | 25 | 22 |
| 비교 frame 키 | 11,960 | 11,960 |
| 같은 frame의 최대 시각 차이 | 약 2.950초 | 약 2.507초 |
| 같은 frame의 중앙 시각 차이 | 약 0.034초 | 약 0.033초 |
| 0.05초 초과 frame 수 | 4,610 | 4,252 |

sequence 2의 cam_1/3/4는 timestamp가 없어 별도 미가용 항목으로 남긴다. 위 통계는 해당 sequence의 비교 가능한 카메라 전체에 대한 값이며, 특정 카메라 쌍이나 짧은 구간의 결과로 일반화하지 않는다. 허용값 0.05초는 검사 설정이고 재동기화·모델 적용 성공 여부와 별개다.

실행 증거: [v0.2 빌드](../logs/adapter-v02-build.log), [54개 테스트](../logs/adapter-v02-tests.log), [v0.2 검증 및 해시](../logs/adapter-v02-validation.json). 실제 비교 보고서는 [sequence 1](../outputs/scout-compatibility-sequence1-20261026/comparison.md), [sequence 2](../outputs/scout-compatibility-sequence2-20261026/comparison.md)에 있다. outputs는 원본 데이터와 마찬가지로 Git/Docker 빌드 문맥에서 제외한다.

기존 9개 모델 이미지와 소스는 유지했으며, 실제 SCOUT 이미지 전체 검증·mesh 검사·모델 가중치 추론·평가 지표 계산은 추가 실행하지 않았다.

## 남은 작업

모델별 구체적인 코드 근거와 막힌 항목은 [YACHIYO](adapters/yachiyo.md), [UWIPL](adapters/uwipl.md), [PoseTrack](adapters/posetrack.md)에 있다. 다음 단계는 전체 SCOUT 압축 해제 후 이미지/주석 일치 검사, 비교 구간·동기화 정책 결정, 왜곡과 실제 지면/세계 단위 처리, 원본 모델의 FPS·해상도·경로·행 순서 오류 수정, 가중치 확인, 작은 실제 clip 추론이다. 현재 어댑터는 원본 저장소나 모델 실행 코드를 수정하지 않는다.
