# 어댑터 v0.5 — 모델 출력과 지면 좌표 변환 연결

2026-10-31. v0.4의 `project-ground` 앞에 **모델별 중간 결과를 읽어 원본 픽셀과 프레임으로 대응시키는 기능**을 추가했다. 실제 SCOUT 보정·mesh를 사용한 연결 검사까지 완료했다. 가중치가 없으므로 모델 파일은 명시적인 합성 자료로 검사했으며 실제 모델 추론·추적·성능 평가는 수행하지 않았다.

**후속 검증 (2026-10-31):** 실제 PoseTrack 검출·포즈 141행을 기존 v0.5 입력 연결기로 읽어 8개 카메라의 프레임 대응을 확인했다. 발목 조건 충족 78행, 낮은 신뢰도 63행이며, 이번 후속 검증에서는 지면 투영·추적은 수행하지 않았다. [실제 추론과 출력 검사](scout-small-inference.md). 아래 합성 자료 검증 기록은 당시 상태로 보존한다.

## 지원하는 연결

| 입력 형식 | 명시적으로 선택하는 정책 | 프레임과 식별번호 |
|---|---|---|
| `yachiyo-sct`: 카메라별 SCT JSON | `yachiyo-bbox-bottom`: `(x1+x2)/2, y2` | 모델 frame 1부터, `OfflineID`는 camera 범위. -1은 미할당 그대로 보존 |
| `uwipl-mot16`: 추적 10열 + 양쪽 발목 6열 | `uwipl-feet`: 양쪽 confidence >0.5이면 발목 평균, 아니면 `(x+0.5w, y+0.95h)` | 모델 frame 0부터. camera/sequence ID 범위를 호출자가 명시 |
| `posetrack-pose57`: 포즈 57열 + 대응 검출 7열 | `ankles-only`: 양쪽 발목 confidence >0.7인 경우만 평균 | 모델 frame 0부터. 검출 행에 사람 ID가 없으므로 `id-scope none` |

YACHIYO와 UWIPL의 **2D 점 선택 규칙**은 고정한 원본 소스에 맞췄다. 세계좌표 변환을 기존 H에서 SCOUT 메시로 바꾸는 것은 별도 적용 변경이며 전체 모델과 동등하다는 뜻이 아니다. PoseTrack 정책은 진단용이다. 원본의 다중 카메라 삼각측량, 관절 갱신 시각, 우선순위, 높이 0.15 발 평면과 bbox 대체 규칙을 재현하지 않는다. 낮은 발목 신뢰도를 bbox로 임의 대체하지 않고 실패 상태로 보존한다.

원본 코드 근거: `repos/yachiyo/tracking/src/utils.py`의 `add_object_from_image_path`, `repos/uwipl/STCRA/file_utils.py`의 `load_tracking`, `repos/posetrack/mmpose/demo/save_pose_with_det_multiscene.py`, `repos/posetrack/track/run_tracking_batch.py`, `repos/posetrack/track/Tracker/PoseTracker.py`의 `get_output`.

## 명령과 산출물

- `model-ground-input`: native 파일 → `points.csv`, `point_map.csv`, `bridge.json`. 이미지 좌표와 원본 프레임 대응만 준비한다.
- `model-ground`: 위 준비 → v0.4 지면 교차 → `ground_candidates.csv`, `model-ground.json`. `prepared/`, `projection/`에 각 단계의 입력 대응과 기하 기록을 보관한다.
- `--emit-yachiyo-candidate`: YACHIYO SCT에만 명시적으로 사용할 수 있다. 모든 행이 유효한 교차점을 얻은 경우에만 별도 `native_candidate/cameraNNN_tracking_results.json`에 WorldCoordinate의 XY를 채운다. 원본 파일, ID, frame, bbox, NpyPath 및 기타 필드는 유지하고 XYZ 전체는 CSV에 보관한다. 실패 행이 하나라도 있으면 후보 파일을 만들지 않고 명령을 실패 처리한다.

`ground_candidates.csv`는 실제 추적 결과 `predictions.csv`가 아니다. 입력 행별 상태와 좌표 후보를 기록한다. 낮은 발목 신뢰도, 영상 밖 좌표, 지면 교차점 없음은 행을 삭제하거나 0으로 채우지 않는다. 카메라·프레임·native serial/ID·원래 행번호·선택 정책을 보존하며 다른 카메라의 ID를 새로 연결하지 않는다.

PoseTrack은 검출과 포즈의 행 수, 순서, frame, bbox 및 score를 비교한다. 원본 포즈 저장기의 float32 변환을 반영하여 비교하며, 동일 frame/bbox의 중복은 대응이 모호하므로 거부한다. ReID NPY의 행 대응은 이번 연결 검사 대상이 아니다. 32MiB 이하 파일, 최대 4,096행의 작은 구간을 대상으로 한다.

입력 좌표가 원본 왜곡 영상의 좌표라는 점과 카메라 식별은 호출자가 올바르게 지정해야 한다. 파일 형식만으로 실제 추론 출처, 좌표 전처리 여부를 증명할 수 없다. `input_provenance_verified=false`, `model_inference_ready=false`를 유지한다.

## 실행 예시

다음은 **실제 모델 파일이 준비된 이후**의 명령이다. 현재 합성 자료를 실제 예측으로 표시하지 않는다. 모든 output은 새 경로여야 한다. 데이터·export·입력 파일은 읽기 전용, 출력만 쓰기 가능하며 CPU·네트워크/GPU 차단 컨테이너에서 처리한다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc

# YACHIYO SCT → 지면 좌표 후보
python3 scripts/adapter.py model-ground \
  --data datasets/scout/extracted/scout \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-yachiyo \
  --input /실제/camera001_tracking_results.json \
  --output outputs/yachiyo-ground-new \
  -- --camera cam_0 --format yachiyo-sct --policy yachiyo-bbox-bottom \
     --id-scope camera --input-kind model_output

# UWIPL: clustering 이전의 camera-local SCT를 사용하는 예
python3 scripts/adapter.py model-ground \
  --data datasets/scout/extracted/scout \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-uwipl \
  --input /실제/S001_c001_feet.txt \
  --output outputs/uwipl-ground-new \
  -- --camera cam_0 --format uwipl-mot16 --policy uwipl-feet \
     --id-scope camera --input-kind model_output

# PoseTrack: 포즈 TXT와 동일 검출 행의 CSV를 모두 전달
python3 scripts/adapter.py model-ground \
  --data datasets/scout/extracted/scout \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \
  --input /실제/pose/camera_0001.txt \
  --detections /실제/detection/camera_0001.txt \
  --output outputs/posetrack-ground-new \
  -- --camera cam_0 --format posetrack-pose57 --policy ankles-only \
     --id-scope none --input-kind model_output
```

YACHIYO에서 native 형식의 별도 후보 파일도 필요하면 첫 명령의 끝에 `--emit-yachiyo-candidate`를 추가한다. 이는 MCPT 실행 명령이 아니며 공간 단위·유효 지면·가림·경로·시간 설정을 검증한 뒤 사용해야 한다. UWIPL의 `id-scope sequence`는 실제 clustering에서 카메라 간 global ID를 부여한 경우에만 사용한다. 옵션 변경으로 global ID가 생성되지 않는다.

## 완료한 검증

이미지 `scout-mtmc/adapter:0.5`의 ID는 `sha256:bfd6a2d84792f82e1a23cfc0758306150b4dd3bea9a6bf096e3c188706bce061`이다. 새 의존성은 설치하지 않았으며 이전 이미지와 모델 코드도 보존한다. `logs/adapter-v05-build.log`는 초기 샌드박스 접근 실패 기록이다. 이후 2026-10-29 빌드에 성공했고, 아래 실행기는 성공한 이미지 ID와 현재 Docker 이미지를 대조한 뒤 실행했다.

- 전체 자동 검사 **149개 통과**: 기존 119개 + 모델 연결 30개. 프레임 0/1 차이, ID 0/-1/큰 정수, 잘못된 행순서·열 수·중복, confidence 경계, 빈 결과, 원본 보존, 실패 시 출력 정리를 포함한다.
- 실제 SCOUT 보정·메시 + 합성 native 자료: **3모델 × 8카메라, 24건 통과**. 총 48행에서 40행이 지면 좌표 후보를 얻었고 PoseTrack의 낮은 confidence 8행은 좌표 없이 보존했다. 이 비율은 성능 지표가 아니다.
- UWIPL 원본 `file_utils.load_tracking(enable_feet=True)`을 실행하여 8개 카메라 각각의 높은 신뢰도·bbox 대체 사례에서 2D 점 선택이 일치함을 확인했다. 실제 STCRA는 실행하지 않았다.
- YACHIYO 후보 파일 8개에서 WorldCoordinate 이외의 모든 입력 필드가 유지됐다. 실제 SCPT/MCPT는 실행하지 않았다.
- 선택 원본 JPEG 16개, 보정 8개, 시간 파일, 메시, 모델별 manifest/profile 등 **32개 파일의 해시와 수정 시각이 유지**됐다.

합성 bbox/keypoint/ID는 `outputs/scout-model-ground-v05-20261031/synthetic_fixtures/`에 보관하고 카메라마다 provenance를 기록했다. 실제 사람의 검출·포즈·추적 결과와 섞으면 안 된다. 실제 이미지에서 사람을 찾아 만든 점이 아니라 기하적으로 교차하는 위치를 선택한 테스트다.

재현 명령:

```bash
python3 scripts/adapter.py test
python3 scripts/validate-model-ground.py run \
  --data datasets/scout/extracted/scout \
  --exports outputs/scout-first8-20s-v03-20261027 \
  --output outputs/model-ground-validation-new
```

검증 기록: [자동 검사](../logs/adapter-v05-tests-20261031.log), [24건 실행 로그](../logs/scout-model-ground-v05-20261031.log), [24건 결과](../outputs/scout-model-ground-v05-20261031/validation.json), [종합 기록](../logs/adapter-v05-validation.json).

## 남은 연결

가중치가 준비되면 실제 검출·포즈·SCT 자료로 같은 입력 계약을 확인해야 한다. YACHIYO 후보를 원본 MCPT에 투입하는 실행 계층, UWIPL의 하드코딩 H/7카메라·장면 규칙, PoseTrack의 P/H·다중뷰·발 높이 처리, FPS 기반 임계값은 아직 연결하거나 일반화하지 않았다. 지면 좌표의 물리적 단위·실제 정확도·보행 영역·가림도 미검증이다. 모델용 native H 생성, 원본 추적기 실행, 전체 추론·학습·성능 평가는 하지 않았다.

구현: [model_ground.py](../adapters/scout_adapter/model_ground.py), [단위 검사](../tests/test_model_ground.py), [실제 기하 연결 검사](../scripts/validate-model-ground.py).
