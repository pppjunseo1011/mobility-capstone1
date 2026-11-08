# SCOUT → YACHIYO 1차 프로파일

구현은 [yachiyo.py](../../adapters/scout_adapter/profiles/yachiyo.py)이다. 기준 원본은 `repos/yachiyo`, `main`, 커밋 `f881fe0edf60fe697efb2bf4c935b80a80861230`이다. Python 표준 라이브러리만 사용한다.

이 프로파일은 **입력 배치 계획·카메라 설정·결과 파싱**을 제공한다. 프레임 복사/번호 변경, 영상 생성, 원본 프레임·timestamp 역매핑은 공통 어댑터가 담당한다. 모델 추론을 실행하거나 검출·ReID·포즈 산출물을 생성하지 않는다. `inference_ready`는 항상 `false`이며, 해결할 항목은 `blockers`에 기록한다.

## 입력 배치와 번호 계약

`build_profile(manifest)`의 모든 경로는 어댑터 **출력 루트 기준 상대경로**이다. 원본 SCOUT 또는 모델 checkout을 수정하지 않는다.

| 항목 | 매핑 |
|---|---|
| 장면 | `sequence_number=1` → `scene_001` |
| 카메라 | 원본 `cam_N` → 모델 ID `N+1`, 폴더 `camera_{N+1:04d}` |
| 모델 프레임 | 선택한 프레임 순서대로 **1부터 연속 번호** |
| 이미지 | `Original/scene_001/camera_0001/Frame/000001.jpg` |
| 포즈용 영상의 필수 목적지 | `Original/scene_001/camera_0001/video.mp4` |
| 추후 검증할 보정의 목적지 | `Original/scene_001/camera_0001/calibration.json` |
| 중간 결과 | `Detection/scene_001`, `EmbedFeature/scene_001/camera_0001`, `Pose/scene_001/camera_0001` |
| 추적·제출 결과 | `Tracking/scene_001`, `Submission/scene_001` |

예를 들어 원본 frame `[81, 83]`을 선택했다면 모델 frame `[1, 2]`에 대응한다. 파일명만 보고 원본 frame을 복원하면 안 되며, 공통 manifest의 역매핑을 사용한다. 카메라를 일부만 선택해도 `cam_2`는 모델 ID 3을 유지한다.

실제 원본의 영상 이름은 `video.mp4`이다(`tools/extract_frame.py:33`, `scripts/pose.sh:14`). `video_path`는 필요한 **목적지 명세**이며, 영상이 이미 존재하거나 생성됐다는 뜻이 아니다. 공통 내보내기가 영상을 만들 경우 선택 이미지와 순서·개수·프레임 번호·시간 간격을 일치시켜야 한다.

제공하는 native 설정은 `tracking/config/scene_2_camera_id_file.json`이며 형식은 다음과 같다.

```json
[
  {"scene_name": "scene_001", "camera_ids": [1, 3]}
]
```

이 파일 역시 출력 루트에 생성한다. 기존 모델 저장소의 동일 파일을 덮어쓰지 않으므로 실행기가 출력 설정을 명시적으로 사용해야 한다. 보정 파일, 가중치 경로, 추론 결과, Python 설정 파일은 자동 생성하지 않는다.

## 실제 원본 코드에서 확인한 요구사항

| 단계 | 계약·주의점 | 근거 |
|---|---|---|
| 프레임 | 추출은 1기준 6자리 JPG. 검출은 정렬된 디렉터리 항목을 1부터 열거하며 파일명 숫자를 읽지 않음 | `tools/extract_frame.py:34`, `detector/aic24_get_detection.py:161`, `:170` |
| 검출 | CSV 8열: `camera,frame,class=1,x1,y1,x2,y2,score`. 세 번째 열은 person ID가 아님. JSON serial은 0기준 8자리 | `detector/aic24_get_detection.py:192`, `:201`, `:209` |
| ReID | 같은 camera의 CSV·JSON 행순서 일치 필요. `Frame/{frame:06d}.jpg`를 crop | `embedder/aic24_extract.py:68`, `:76`, `:90`, `:91` |
| 특징 | `feature_{frame}_{serial}_{x1}_{x2}_{y1}_{y2}_{conf점제거}.npy`. 좌표 순서는 `x1,x2,y1,y2` | `embedder/aic24_extract.py:89`, `tracking/src/utils.py:84` |
| 추적 | Detection JSON 대신 `EmbedFeature` 아래 NPY를 재귀 탐색하고 파일명에서 검출 정보를 파싱 | `tracking/src/utils.py:27`, `:81` |
| 포즈 | CSV frame을 그대로 사용하고, 영상은 0기준으로 열거 | `poser/load_tracking_result.py:10`, `poser/top_down_video_demo_with_track_file.py:149`, `:156` |
| 추적 설정 | 기본 `fps=30`, `image_size=(1920,1080)`, 장면별 camera ID 설정 필요 | `tracking/src/tracking.py:27`, `:31`, `tracking/infer.py:46` |

`Frame/`에 메타데이터나 관련 없는 파일을 함께 놓지 않는다. 원본 검출기는 디렉터리 항목 전체를 정렬하므로 추가 파일도 프레임 순번에 영향을 줄 수 있다.

## 실행 전 해결해야 할 항목

1. **해상도:** 검출기는 `scale=min(800/1080,1440/1920)`와 1920×1080 clipping을 사용한다(`detector/aic24_get_detection.py:159`, `:180`, `:190`). 다른 해상도를 그대로 연결하면 좌표가 잘못될 수 있다. 실제 이미지 크기와 전처리 비율을 사용하도록 실행 코드를 수정해야 한다. 프로파일은 이를 해결한 것으로 처리하거나 임의 resize하지 않는다.
2. **포즈 프레임 불일치:** 원본 검출은 1기준인데 포즈 영상 순회는 0기준이다. 단순 연결 시 첫 영상 프레임을 건너뛰고 bbox와 영상이 한 프레임 어긋난다. 영상 인덱스·CSV Frame·포즈 JSON 키를 함께 맞춰야 하며, 포즈 결과의 키만 사후 변경해서는 해결되지 않는다.
3. **보정과 영상 공간:** 실제 왜곡이 있는 raw 영상에 임의 P/H를 만들지 않는다. `calibration.json`은 `camera projection matrix`, `homography matrix` 키를 요구한다(`tracking/src/utils.py:162`). H는 **세계 지면→영상** 방향이며 코드가 `inv(H)@[u,v,1]` 후 homogeneous 정규화를 수행한다(`:168`). 지면 정의, 좌표 단위, 왜곡 처리, 이미지 크기와 실제 대응점을 검증한 뒤 공급해야 한다. 원본 보정 dict가 있어도 native JSON을 자동 생성하지 않는다.
4. **동기화와 시간:** `cvlabrpi…` timestamp 키 해석은 공통 어댑터 책임이다. timestamp skew가 없으면 미검증, 설정 tolerance를 넘으면 차단 사유를 기록한다. 통과하더라도 sparse sampling을 모델의 연속 30fps로 해석하면 안 된다. 실제 sampling rate와 frame-count 임계값을 조정해야 한다.
5. **산출물·가중치:** YOLOX/OSNet/HRNet 가중치, 실제 검출, crop feature, pose JSON이 필요하다. GT를 검출이나 추적 결과로 대체하지 않는다. 비어 있는 출력 디렉터리는 모델 입력 완료를 뜻하지 않는다.
6. **실행 경로:** 원본 shell은 Conda 활성화와 코드 복사, 공통 CWD의 상대경로를 사용한다. 데이터의 읽기 전용 마운트와 별도 출력 마운트를 연결하는 실행기가 필요하다.

추가로 검출 결과를 append 모드로 쓰므로 재실행 시 CSV 중복과 JSON 연결 손상이 발생할 수 있다(`detector/aic24_get_detection.py:207`). fresh output을 사용해야 한다. ReID의 `np.genfromtxt`는 빈 파일·한 행 파일을 2차원 배열로 정규화하지 않으므로 별도 처리도 필요하다(`embedder/aic24_extract.py:68`).

보정 미존재를 일부 함수가 경고만 하더라도 전체 MCPT가 이를 지원하는 것은 아니다. 거리 계산과 후처리는 `WorldCoordinate.values()`를 직접 사용한다(`tracking/src/mcpt.py:607`, `:831`). 이 프로파일은 이러한 후처리를 임의로 꺼서 성공으로 표시하지 않는다.

## 결과 파싱

`read_results(path, *, format=...)`는 다음 두 형식을 지원한다. 원본을 읽기만 하며 scene별 ID namespacing과 SCOUT frame 복원은 공통 결과 변환기가 수행한다.

| format | native 형식 | 반환 frame |
|---|---|---|
| `yachiyo-json` | `camera ID → serial → {Frame,Coordinate,GlobalOfflineID,WorldCoordinate}` | native `Frame`의 **1기준 그대로** |
| `aic9` | `camera,id,frame0,x,y,w,h,world_x,world_y` 9열 | native `frame0 + 1` |

권장 입력은 `Tracking/scene_NNN/fixed_whole_tracking_results.json`이다(`tracking/src/run.py:83`). `aic9` 원본은 `Submission/scene_NNN/track1.txt`이며 공식 변환기가 `Frame-1`을 적용한다(`tools/generate_submission.py:39`). 공식 구분자는 공백이며 reader는 쉼표 구분 9열도 허용한다.

반환 행 예:

```json
{
  "camera_id": 1,
  "track_id": 0,
  "frame": 1,
  "bbox_xywh": [10.0, 20.0, 20.0, 30.0],
  "score": null,
  "world_xyz": null,
  "world_xy": [-2.5, 3.5]
}
```

`GlobalOfflineID=0`은 유효하다. 같은 장면의 카메라 간에는 같은 ID 의미를 유지하고, 장면이 바뀌면 별도 namespace를 적용해야 한다. 카메라별 SCT `OfflineID`를 대신 사용하지 않는다.

최종 native 형식에는 confidence와 측정된 world z가 없으므로 **score나 z를 만들지 않는다**. `world_xyz=None`, `world_xy=[x,y]`로 원래 XY만 보존한다. JSON의 WorldCoordinate가 없거나 null이면 `world_xy=None`이다. native world 좌표 단위·보정 신뢰도가 검증됐다는 의미는 아니다.

Reader는 누락/음수 global ID, 잘못된 카메라/프레임 정수, 비유한 수, 0 이하 bbox 크기, 음수 이미지 좌표, 필수 JSON 구조 오류, 중복 JSON 키, 같은 camera/track/frame 중복, 열 수 오류를 `ValueError`로 거부한다. 미할당 결과를 조용히 버리지 않는다. CSV 빈 줄은 무시한다. 양 끝 좌표는 native 방식대로 `w=x2-x1`, `h=y2-y1`로 변환하며 `+1`을 적용하지 않는다.

## 검증 범위

합성 manifest로 상대경로·카메라 `N+1` 고정·native camera config·미완료 상태를 확인했다. 같은 합성 검출을 JSON과 공백/쉼표 AIC9으로 읽어 결과 일치, frame 0→1, 유효 ID 0, score/z 미생성을 검증했다. 잘못된 열 수·숫자·bbox·ID·중복 결과 등 12개 오류 입력이 거부되는 것도 확인했다.

이 검증은 프로파일과 결과 reader에 대한 것이다. SCOUT 전체 데이터 변환, 네트워크 모델 추론, 추적 성능 검증을 통과했다는 뜻이 아니다.

## 2026-10-26 추가 적용 검토

아래는 고정한 원본 코드를 다시 읽어 확인한 **적용 가능성**이다. 제안한 추가 변환기·실행기는 아직 구현하거나 모델 추론으로 검증하지 않았다. 원본 저장소를 수정하지 않는 조건으로 검토했다.

- **포즈 프레임 정렬:** detector/ReID용 1기준 CSV를 보존하고, 별도 pose 입력 CSV에만 `frame - 1`을 적용한 뒤 원본 포즈 CLI를 실행하고, 출력 JSON의 frame key에 `+1`을 적용하는 방안이 가능하다. CSV frame을 그대로 읽고 영상과 결과를 0부터 열거하기 때문이다(`poser/load_tracking_result.py:10`, `poser/top_down_video_demo_with_track_file.py:98`, `:149`, `:176`). tracker는 frame과 bbox의 정확한 조합으로 keypoint를 연결한다(`tracking/src/pose.py:107`, `:116`). 따라서 추론 전 CSV 변환과 추론 후 JSON 변환이 모두 필요하며, 잘못된 영상에서 추론한 결과의 key만 바꾸어서는 해결되지 않는다.
- **P/H와 영상 왜곡:** SCOUT의 `P=K[R|T]`는 계산할 수 있지만 raw 픽셀의 비영 왜곡까지 표현하지 않는다. 검증된 지면이 `z=z0`이고 영상을 `K'` 기준으로 undistort했다면 `H=K'[r1 r2 T+z0*r3]`를 계산할 수 있다. 세계 단위·지면·영상 공간 확인이 먼저이며, 임의의 `z0=0`이나 단위를 선언하지 않는다. 원본은 `H^-1`로 bbox 하단중앙을 세계 XY로 변환한다(`tracking/src/utils.py:91`, `:162`, `:168`). 전체 영상을 undistort하는 경우 공통 결과의 raw 좌표 복원을 위한 별도 역변환도 필요하다.
- **세계좌표의 외부 연결 대안:** raw 영상을 유지하면서 SCT 결과의 bbox 하단중앙을 외부에서 undistort하고, 검증된 지면과 광선의 교점으로 세계 XY를 계산하여 SCT JSON의 `WorldCoordinate`를 채운 뒤 MCPT를 실행하는 방안도 검토할 수 있다. SCPT는 세계좌표를 사용하지 않고 loader는 보정이 없으면 null을 저장하지만(`tracking/src/utils.py:149`, `:164`), 기본 실행은 세계좌표 기반 유사도 조정을 켜며 MCPT·후처리는 실제 XY를 요구한다(`tracking/infer.py:148`, `tracking/src/mcpt.py:607`, `:831`). 이 대안은 **미구현**이며, 지면·단위·좌표 정확도 검증과 단계별 실행 연결이 필요하다.
- **OSNet checkpoint의 대체 로딩:** 원본은 `deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar`를 지정한다(`embedder/aic24_extract.py:36`). 이 파일이 없으면 vendor `FeatureExtractor`가 `pretrained=True`로 분기하고, OSNet은 ImageNet 사전학습 가중치를 받을 수 있다(`docker/yachiyo/vendor/deep-person-reid/torchreid/utils/feature_extractor.py:74`, `docker/yachiyo/vendor/deep-person-reid/torchreid/models/osnet.py:480`, `:508`; vendor 경로는 프로젝트 루트 기준). 실행 전에 의도한 ReID checkpoint의 존재·출처·해시를 검증해야 하며, 자동 다운로드 성공을 동일한 ReID 모델 준비 완료로 간주하지 않는다.
- **FPS와 고정 임계값:** 기본 `fps=30`과 3초 구간은 SCOUT의 선택 시간축에 맞게 설정해야 한다(`tracking/src/tracking.py:21`, `:30`, `:104`). `short_track_th=120`, 보간 간격 15뿐 아니라 코드 내부의 `len(common_frames)>120`도 있어 FPS 하나를 바꾸면 모든 시간 조건이 조정되는 구조가 아니다(`tracking/src/tracking.py:71`, `:84`, `tracking/src/mcpt.py:637`). MCPT는 timestamp 대신 동일 Frame의 교집합을 사용한다(`tracking/src/mcpt.py:623`). 외부 runner에서 `tracking_params`를 전달할 수 있지만(`tracking/src/run.py:9`, `:45`), 카메라 동기화·반복 timestamp·선택 구간의 길이와 고정 조건의 영향은 별도로 검증해야 한다.

## 2026-10-27 포즈 프레임 연결 구현

앞 절의 **CSV frame−1 → 원본 포즈 추론 → JSON key+1** 중 두 파일 변환기를 [bridges/yachiyo.py](../../adapters/scout_adapter/bridges/yachiyo.py)에 구현했다. 가운데 추론 단계는 실행하지 않았다. 기존 detector/ReID CSV, 원본 모델 저장소, SCOUT 데이터는 수정하지 않는다. 가중치·보정·시간축과 전체 실행 연결은 여전히 별도 준비가 필요하며 `model_inference_ready`는 `false`다.

표준 Python만 사용하는 API는 다음과 같다. `export`는 YACHIYO export 디렉터리, `source`는 변환할 원본 파일, `output`은 **아직 없는 새 디렉터리**, `camera`는 `cam_0` 같은 원본 카메라명이다.

```python
from scout_adapter.bridges.yachiyo import prepare_pose_csv, restore_pose_json

prepare_pose_csv(export, source, output, camera="cam_0", input_kind="unverified")
restore_pose_json(export, source, output, camera="cam_0",
                  pose_input=prepared_directory, input_kind="unverified")
```

| 단계 | 입력과 출력 | 검증 |
|---|---|---|
| `prepare_pose_csv` | native 1기준 검출 CSV → `pose_input.csv` 0기준, `frame_map.csv`, `bridge.json` | 정확한 8열·카메라·`class=1`·export frame 범위·행순서·정수 xyxy·이미지 경계·confidence·중복 frame/bbox |
| `restore_pose_json` | 원본 포즈 CLI의 0기준 JSON → `camera_NNNN_out_keypoint.json` 1기준, `bridge.json` | prepare 디렉터리의 CSV 해시와 동일 export manifest/profile 해시·카메라 확인 후 모든 검출 frame/bbox에 정확히 대응하는 COCO-17 결과 확인 |

CSV는 `camera_0001,frame,class=1,x1,y1,x2,y2,score` 형식이며 세 번째 열을 track ID로 해석하지 않는다(`detector/aic24_get_detection.py:192`, `:209`; `embedder/aic24_extract.py:76`). 출력은 행순서와 frame 이외 필드를 유지한다. 포즈 입력을 원래 Detection CSV와 바꿔치기하면 ReID의 1기준 이미지 경로가 깨지므로 **별도 파일을 원본 포즈 CLI의 첫 번째 인자로 전달**해야 한다(`embedder/aic24_extract.py:91`, `poser/top_down_video_demo_with_track_file.py:31`).

복원은 원본 HRNet COCO 설정과 기본 pose 경로를 대상으로 한다. 원본 loader는 detector confidence를 버리고 bbox의 다섯 번째 값에 `1.0`을 넣는다(`poser/load_tracking_result.py:12`). 따라서 결과 bbox는 `xyxy,1.0`, keypoints는 17×3이어야 하며, 준비 CSV의 모든 검출과 일치해야 한다. 기본 `--bbox-thr=0.3`에서는 이 bbox들이 모두 유지된다(`poser/top_down_video_demo_with_track_file.py:53`; vendor `mmpose/apis/inference.py:372`, `:406`). 검출을 걸러내는 별도 실행 옵션이나 다른 keypoint 체계는 이 계약에서 지원하지 않는다. keypoint의 영상 밖 좌표와 heatmap score 1 초과는 임의 clipping하지 않고 유한성만 확인한다.

빈 검출 CSV와 빈 포즈 JSON은 빈 결과로 유지하며 가짜 bbox나 빈 frame을 삽입하지 않는다. 같은 frame/bbox 중복, 누락된 포즈, 잘못된 frame 기준, 중복 JSON 키, 비유한 숫자, 변경된 prepare CSV·export metadata는 거부한다. 기존 출력 덮어쓰기와 입력·출력 경로 중첩도 거부하고, 쓰기 실패 시 부분 디렉터리를 최종 출력으로 남기지 않는다.

`input_kind`는 `unverified`(기본), `model_output`, `synthetic_fixture` 중 하나다. 이것은 호출자가 선언한 출처이며 모델 실행의 증명이 아니다. prepare 입력이 `synthetic_fixture`이면 복원도 같은 표시를 요구한다. `bridge.json`에는 입력·출력 해시와 `input_provenance_verified=false`, `video_alignment_verified=false`, `model_inference_run=false`를 기록한다. 실제 영상의 픽셀·순서 일치는 기존 영상 검증을 따로 통과해야 한다. 번호만 옮기는 변환기로 잘못된 영상에서 추론한 keypoint를 교정할 수는 없다.

[test_yachiyo_bridge.py](../../tests/test_yachiyo_bridge.py)에 합성 테스트 24개를 작성했다. 원본 frame 81–83과 모델 1–3·포즈 0–2의 대응, 빈 입력·한 행 입력, 원본 bytes/mtime 보존, 잘못된 형식·중복·해시 변경 거부, 실패 시 부분 출력 정리를 포함한다. 합성 keypoint는 실제 모델 결과로 표시하지 않는다. 이 절 작성 시 구문 검사를 통과했으며 Docker 통합 테스트 결과는 별도 검증 기록에 남긴다.

이후 `scout-mtmc/adapter:0.3`에서 이 24개를 포함한 전체 **95개 검사 PASS**를 확인했다. [검사 로그](../../logs/adapter-v03-tests.log), [CPU 컨테이너 CLI 사용법](../scout-next-stage.md).


## 2026-10-31 지면 좌표 후보 연결

[v0.5 모델 출력 연결](../scout-model-ground.md)에 지원 형식, 명시적 점 선택 정책, 원본 프레임 대응과 실패 처리를 정리했다. 실제 SCOUT 기하와 합성 native 자료로 검증했으며 실제 모델 추론·추적은 수행하지 않았다. 원본 보정 H/추적 알고리즘 적용은 여전히 별도 작업이다.
