# SCOUT 어댑터 v0.3 — 연속 구간·디코딩·실행 연결

**후속 작업(2026-10-28):** [어댑터 v0.4](scout-ground-projection.md)에서 왜곡 역변환과 실제 광선–지면 메시 교차를 구현했다. 아래는 2026-10-27의 보존 기록이다.

2026-10-27, 사용자 요청에 따라 **sequence_1 / cam_0~cam_7**에서 가중치 없는 다음 단계를 진행했다. [이전 8프레임 검증](scout-real-validation.md)에서 발견한 문제를 대상으로 한다. 모델 추론·학습·성능 평가는 실행하지 않는다.

## 시간축과 테스트 구간

`select-window`는 원본 frame 번호를 그대로 유지하면서 조건을 만족하는 가장 이른 연속 구간을 찾는다. 영상이나 timestamp를 보간·재배열하지 않는다. 기본 조건은 카메라 간 동일 frame의 시간 차이 ≤50ms, 카메라별 인접 시간 차이 `0 < dt ≤150ms`, 모든 선택 JPEG 존재다. 중복 timestamp·시간 역행·이미지 누락·frame 번호 간격이 있으면 구간을 끊는다.

실제 timestamp 11,960개 시점을 조사했다. 8개 카메라의 최대 시간 차이가 50ms를 넘은 시점은 1,626개, 같은 카메라에서 인접 timestamp가 반복된 경우는 1,537개다. 두 수는 서로 다른 집계이며 더해서 프레임 수로 해석하면 안 된다.

선정 결과는 **원본 frame 277~476, 카메라당 200프레임**이다. 10FPS 영상으로 약 20초이며, 카메라별 첫 관측~마지막 관측 시간은 약 19.889~19.918초다. 최대 카메라 간 시간 차이는 **35ms**, 카메라별 인접 간격은 약 **63~137ms**다. [선택 보고서](../outputs/scout-first8-window-v03-20261027/selection.json), [전체 시간 진단 CSV](../outputs/scout-first8-window-v03-20261027/timeline.csv).

이 구간은 입력 연결 점검용이다. GT나 추적 성능으로 쉬운 구간을 고르지 않았으며 최종 학습/평가 분할도 아니다. 시간차가 큰 구간까지 적용하려면 별도의 동기화 정책이 필요하다. `select-window`의 종료 코드 0은 보고서 생성 성공을 뜻하므로 자동화에서는 `selection.json`의 `selected`가 null이 아닌지도 확인해야 한다.

## JPEG와 영상의 디코딩 일치

v0.2에서는 FFmpeg가 JPEG를 읽었지만 모델의 JPEG 입력 경로는 OpenCV를 사용해 색상 값이 일부 달랐다. v0.3은 기본적으로 **OpenCV `imread(IMREAD_COLOR)` → BGR24 → lossless RGB H.264 MP4** 순서로 만든다. 원본 JPEG 복사본의 바이트는 바꾸지 않는다.

기존 경로는 `--video-decoder ffmpeg`로 재현할 수 있다. 새 기본값은 `--video-decoder opencv`다. 프레임 한 장씩 처리하고 디코더 버전과 입력 픽셀 해시를 저장한다. 인코딩·probe 실패 시 MP4를 최종 경로에 게시하지 않고, 이미 있는 파일도 덮어쓰지 않는다. 이것은 디코딩 색차 수정이며 왜곡 보정이나 영상 좌표계 변환은 아니다.

## YACHIYO 포즈 프레임 연결

새 명령 `yachiyo-pose-input`은 detector/ReID의 1기준 CSV를 **포즈 전용 0기준 CSV**로 변환한다. `yachiyo-pose-output`은 실제 포즈 JSON의 frame을 다시 **1기준**으로 복원한다. 원래 검출 CSV를 바꾸지 않으며, 준비 CSV·export 해시와 모든 frame/bbox 대응을 검사한다.

다음 명령은 실제 검출·포즈 결과 파일이 준비된 이후의 사용 예다. 현재 가중치나 실제 중간 결과는 없으므로 여기의 모델 결과 파일을 만들어내거나 아래 변환을 실제 추론 완료로 표시하지 않았다.

```bash
python3 scripts/adapter.py yachiyo-pose-input \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-yachiyo \
  --input /실제/검출/camera_0001.txt --output outputs/pose-input-cam0-new \
  -- --camera cam_0 --input-kind model_output

# 원본 포즈 CLI에는 위 pose_input.csv와 대응 video.mp4를 전달한다.
# 포즈 추론 후 같은 export와 준비 CSV로 frame 기준을 복원한다.
python3 scripts/adapter.py yachiyo-pose-output \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-yachiyo \
  --input /실제/포즈/camera_0001_frame0.json \
  --pose-input outputs/pose-input-cam0-new --output outputs/pose-output-cam0-new \
  -- --camera cam_0 --input-kind model_output
```

`model_output`은 호출자의 출처 선언이며 모델 실행의 증명이 아니다. 합성 테스트 입력에는 `synthetic_fixture`를 유지한다. 지원하는 native CSV·COCO-17·bbox score 계약과 제한은 [YACHIYO 어댑터 문서](adapters/yachiyo.md)에 있다.

## 환경과 검증

새 CPU 이미지: `scout-mtmc/adapter:0.3`, ID `sha256:5ebc11a7ae782c8e4b6dbef0ea32f19237f05f16531e49b14c441261eb423900`. 기반 모델 이미지를 재사용하며 새 의존성을 설치하지 않았다. 기존 v0.1/v0.2와 모델 이미지 9개는 보존한다.

**컨테이너 회귀 검사 95개 PASS**: 기존 54개 + 시간 구간 7개 + 영상 픽셀/실패 처리 10개 + YACHIYO 프레임 연결 24개. [빌드 로그](../logs/adapter-v03-build.log), [검사 로그](../logs/adapter-v03-tests.log). 합성 검사는 실제 모델 추론 성공과 구분한다.

실제 20초 변환 검사는 **31개 모두 PASS**다. 원본 JPEG 1,600장을 읽었고 세 모델용 MP4 24개(각 200프레임)를 생성했다. 모든 MP4의 프레임 순서·픽셀이 OpenCV로 직접 읽은 원본 JPEG와 정확히 일치했다(최대 픽셀 차이 0). 선택 원본 JPEG의 해시·수정 시각도 유지됐다. MOT 공통 GT **33,550행**을 생성했다. [validation.json](../outputs/scout-first8-20s-v03-20261027/validation.json), [실행 로그](../logs/scout-first8-20s-v03-20261027.log).

기존 모델 컨테이너에서도 **24/24개 영상 PASS**다. 각 컨테이너의 OpenCV·MMCV·FFmpeg 순차 디코딩과 해당 export JPEG의 OpenCV 픽셀이 모두 일치했다. 어댑터에서 확인한 영상 해시와 모델 reader가 확인한 영상 해시도 24개 모두 같다.

| 검사 이미지 | 영상 | 결과 |
|---|---:|---|
| YACHIYO pose | 8개 × 200프레임 | [PASS](../logs/model-video-20s-v03-yachiyo-20261027/validation.json) |
| UWIPL pose | 8개 × 200프레임 | [PASS](../logs/model-video-20s-v03-uwipl-20261027/validation.json) |
| PoseTrack all | 8개 × 200프레임 | [PASS](../logs/model-video-20s-v03-posetrack-20261027/validation.json) |

정확한 이미지 ID·코드 해시·영상 해시 연결·원본 보존·미실행 범위는 [종합 검증 기록](../logs/adapter-v03-validation.json)에 저장했다. 기존 환경 파일 25개와 모델 저장소 3개는 변경되지 않았다. 영상 reader 검사는 신경망 forward나 원본 모델의 전체 실행 경로 검증을 포함하지 않는다.

이번에는 `--gt-formats mot`를 명시했다. 이전에 확인한 COCO 이미지 목록 누락은 수정하거나 해결된 것으로 표시하지 않는다. 모델별 프레임 복사본과 lossless 영상을 모두 만든 이번 표본은 약 **12.73GiB**를 사용한다. 전체 길이로 확장하기 전에는 원본 JPEG 직접 읽기 또는 공통 영상 캐시를 연결해 중복 저장을 줄여야 한다.

## 카메라 보정 진단

8개 회전행렬의 수치 조건은 정상이며, 동일 P를 사용하는 합성 투영 왕복도 수치적으로 일치했다. 이는 실제 촬영 보정의 정확도를 입증하는 검사가 아니다. 273개 합성 화면 격자의 ideal↔distorted 차이는 최대 약 **106.47px**였고 기본 undistort 역변환 왕복에도 외곽에서 오차가 남았다.

`mesh_ground_only.ply`의 vertex 497,662개를 읽어 전역 평면을 적합했다. 잔차 RMS 약 **0.394**, 최대 약 **3.827**이며 단위는 데이터의 world 단위로 기록했다. 실제 카메라별 보행 영역·축·단위 검증 없이 `Z=0`이나 하나의 전역 H를 모델에 넘기지 않았다. GT를 평면 추정에 사용하지 않았다. 수치·진단 한계·후속 연결은 [기하 진단 문서](scout-geometry.md)에 있다.

## 재실행

모든 출력은 새 폴더로 지정한다. 원본 데이터는 읽기 전용이며 CPU 2개·RAM 4GiB·네트워크 없음으로 실행한다.

```bash
python3 scripts/adapter.py select-window \
  --data datasets/scout/extracted/scout --output outputs/window-new \
  -- --sequence 1 --cameras cam_0 cam_1 cam_2 cam_3 cam_4 cam_5 cam_6 cam_7 \
  --frames 200 --sync-tolerance 0.05 --max-step 0.15

# 위 selection.json에서 실제 선정된 시작 frame을 확인한다.
python3 scripts/validate-scout-sample.py run \
  --data datasets/scout/extracted/scout --output outputs/first8-20s-new \
  --sequence 1 --cameras cam_0 cam_1 cam_2 cam_3 cam_4 cam_5 cam_6 cam_7 \
  --start-frame 277 --frames 200 --gt-formats mot

# 각 모델 이미지에서 OpenCV/MMCV/FFmpeg와 복사 JPEG의 픽셀을 함께 확인
python3 scripts/validate-model-video-inputs.py run \
  --exports-root outputs/first8-20s-new --output logs/model-readers-20s-new \
  --sequences 1 --camera-count 8 --frames 200 --compare-jpeg
```

## 실제 추론 전에 남은 연결

| 대상 | 아직 필요한 작업 |
|---|---|
| 공통 | 실제 검출·포즈·ReID 가중치와 provenance 확인, 평가 구간 확정, 긴 시간축 처리 정책 |
| 보정 | 왜곡·지면·세계 좌표 단위 검증 후 모델의 P/H 또는 외부 세계좌표 변환 연결 |
| YACHIYO | 구현한 frame 변환기 양쪽에 실제 포즈 CLI 연결, 경로와 30FPS/프레임 기반 임계값 검토 |
| UWIPL | 7카메라 고정 설정을 8카메라에 일반화, 장면 anchor frame·특정 인물 처리·STCRA 보정 연결 |
| PoseTrack | P/H·발 높이 0.15의 의미 확인, 경로와 검출/포즈/ReID 행 대응·빈 결과 처리 |

어댑터 입력 배치와 프레임 변환기가 준비됐다는 이유로 위 항목을 해결된 것으로 표시하지 않는다. 원본 모델 저장소를 수정하지 않았으며 `model_inference_ready=false`를 유지한다.
