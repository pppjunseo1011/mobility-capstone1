# SCOUT → UWIPL 1차 프로파일

대상 원본은 `repos/uwipl`, `main`, commit `8e23a69de57d2547b774399ae45792f83a2d17ef`이다. 구현은 `adapters/scout_adapter/profiles/uwipl.py`이며 Python 표준 라이브러리만 사용한다. `build_profile(manifest)`는 배치 계약만 반환하고 파일·영상·검출·가중치·모델 실행을 생성하지 않는다. 부모 exporter가 선택 이미지 복사, 모델 frame 번호와 원본 frame 번호의 대응, 명시적인 영상 생성 및 manifest 기록을 담당한다.

**`inference_ready=false`이다.** 폴더 배치 성공과 SCOUT 모델 추론 성공은 다르다. S001/c001 별칭은 AI City의 보정값·카메라 관계·임계값이 SCOUT에 적합하다는 의미가 아니다.

출력 루트를 기준으로 다음 배치를 선언한다. 원본 영상명은 `video.mp4`이다(`scripts/preprocessing/extract_frame.py:67–74`, 원본 README:59). 영상은 선언만 하며 생성·복사를 자동 수행하지 않는다.

```text
configs/uwipl_adapter.json
data/test/S001/c001/img/00000.jpg
data/test/S001/c001/frame/00000.jpg
data/test/S001/c001/video.mp4        # 명시적 영상 생성 시에만 존재
...선택된 카메라별 cNNN...
data/test_det/                     # 빈 디렉터리, GT를 검출로 변환하지 않음
data/test_emb/                     # 빈 디렉터리, 임베딩 미생성
data/SCT/
data/tracklet/
data/hungarian_cluster/
data/hungarian_cluster_S001/
pose/tracking_result/
pose/feet/
reproduce/
```

모델 camera ID는 manifest의 양의 정수 `model_camera_id`, camera 이름은 `c{:03d}`이다. SCOUT `cam_0` 등의 원래 ID는 `source_camera`로 보존한다. `img/`와 `frame/`은 동일한 선택 프레임을 모델 frame **0부터**, 최소 5자리 JPG 이름으로 노출한다. 원본 `source_frame=81`을 모델 `0`으로 내보냈다면 결과 `frame=0`은 부모 frame map을 통해 81로 복원해야 한다. 빈 구간을 임의로 채우거나 GT를 이용해 검출·추적 결과를 만들지 않는다. 단일 scene export는 SCOUT sequence를 별도 필드로 보존하며 서로 다른 sequence의 ID namespace는 부모가 분리한다.

| 단계 | 실제 원본 입력·출력 계약 | 소스 근거 |
|---|---|---|
| YOLOX 실사 검출 | `PROJECT/data/test/S001/CAMERA/img/`; 파일 정렬 순서를 0부터 enumerate; `PROJECT/data/test_det/S001.txt` append | `BoT-SORT/tools/aic_get_detection_S001.py:151–192` |
| Synthetic 검출 | `/data/test/SCENE/CAMERA/img/`; JPG basename 정수 사용; CWD 기준 `data/test_det/SCENE.txt` append (`--out-dir`와 다름) | `mmyolo/tools/aic_get_detection_syn.py:14,106,131–146` |
| 검출 CSV | 8열 `camera,frame,class=1,x1,y1,x2,y2,score`, 0기준 pixel 좌표 | 위 두 검출 코드 |
| ReID | `PROJECT/data/test/SCENE/CAMERA/frame/{frame:05d}.jpg`; 검출 행순서 그대로 feature NPY 생성 | `deep-person-reid/torchreid/aic_extract.py:39–92`, `aic_extract_S001.py:30–93` |
| S001 임베딩 | OSNet 5종 `S001_osnet_x1_0.npy`, `S001_osnet_ibn_x1_0.npy`, `S001_osnet_ain_x1_0.npy`, `S001_osnet_x1_0_market.npy`, `S001_osnet_x1_0_msmt17.npy`; 같은 행을 concat | `BoT-SORT/tools/aic_track.py:229–247` |
| SCT | `data-root`를 직접 받음; `SCT/S001_CAMERA.txt`와 `tracklet/S001_CAMERA.pkl`; frame 0부터 마지막 검출 frame까지 | `BoT-SORT/tools/aic_track.py:139–192,195–275` |
| 포즈 | 영상 enumerate 0과 SCT frame 매칭; JSON `{frame:[{bbox,keypoints},...]}`에 track ID 없음 | `mmpose/demo/load_tracking_result.py:5–16`, `top_down_video_demo_with_track_file.py:149–209` |
| 발 좌표 | MOT 10열 뒤 COCO ankle 15·16의 `(x,y,confidence)` 각각 3열 추가 | `mmpose/tools/convert.py:5–24` |
| 보정 | `HOMO`는 world→image H; image→world는 역행렬 곱 후 homogeneous 정규화 | `STCRA/cam_utils.py:37–51` |
| 최종 제출 | `camera,scene_numeric*1000+global_id,frame,x,y,w,h,-1,-1` 9열 | `BoT-SORT/tools/generate_submission.py:55–61,87–102` |

단계별 root 의미도 다르다. `run_tracking.py:12`는 project-root 뒤에 `data`를 붙이지만 `aic_track.py:199`는 data-root를 직접 받는다. ReID는 project-root를 받는다. `configs/uwipl_adapter.json`은 이 정보를 명시하는 어댑터 계약이며 원본 코드가 자동으로 읽는 설정 파일은 아니다.

## 실제 추론 전에 해결해야 하는 원본 제약

- 실사 검출은 1920×1080과 `min(800/1080,1440/1920)`을 고정하고 bbox를 1920×1080으로 자른다. 실제 이미지·모델의 resize ratio로 수정해야 한다. `aic_get_detection_S001.py:157,173,181–185`.
- ReID synthetic 스크립트는 `make_parser()`가 이미 반환한 Namespace에 `.parse_args()`를 다시 호출하며 checkpoint 경로에 실제 `deep-person-reid`와 다른 `deep_person_reid`를 쓴다. 두 ReID 스크립트는 `cur_frame=0`으로 시작해 첫 검출 frame 0에서 `img`가 초기화되지 않고, camera가 바뀌어도 같은 frame이면 이전 이미지를 재사용한다. `aic_extract.py:17–28,49,68–85`; `aic_extract_S001.py:69–86`.
- `FeatureExtractor`는 지정 checkpoint가 없으면 pretrained 초기화를 선택한다. 필요한 checkpoint 존재를 검사하지 않은 실행은 다운로드를 유발할 수 있다. `torchreid/utils/feature_extractor.py:70–88`.
- SCT의 FPS는 30 고정이다. c001..c007 threshold, c005 ROI 제거를 일반화해야 한다. 카메라에 검출이 없거나 CSV가 1행인 경우에도 원본의 `np.genfromtxt` shape와 마지막 행 접근을 보완해야 한다. `aic_track.py:145,198,208–214,229,251–269`.
- Clustering은 정해진 scene별 수천~5만 frame anchor 선택과 feature 거리 임계값을 사용한다. S001에는 c001/c002 특별 진입 인물과 ID 8 처리도 있다. 짧은 clip에서 원본이 정상 완주한다고 가정할 수 없다. `aic_hungarian_cluster.py:90–134,259–265`; `aic_hungarian_cluster_S001.py:274–345,449–480`.
- 포즈 JSON과 SCT는 frame 내부 행순서가 정확히 같아야 한다. converter는 S001/c001..c007 고정이며 `S001_c001_feet.txt`를 출력하지만 STCRA는 filename 마지막 `_` 토큰을 camera로 사용하므로 입력명 정리가 필요하다. `mmpose/tools/convert.py:5–24`; `STCRA/run_stcra.py:285–287`.
- STCRA는 S001의 homography와 카메라별 가중치, 같은 frame=같은 시각을 가정한다. 거리 250/200/150의 세계 좌표 단위도 SCOUT에 맞춰 검증해야 한다. CLI output 옵션을 무시하고 `reproduce/S001_CAMERA.txt`에 쓴다. `run_stcra.py:92–104,262–301`.
- Boundary filtering은 1920×1080 고정이고, 제출 생성은 특정 7개 scene과 43개 카메라를 강제한다. interpolation은 호스트 절대 경로를 고정한다. `boundaryrect_removal.py:17–33`; `generate_submission.py:21–27,102`; `aic_interpolation.py:9,117–140`.

`calibration_path=None`이며 H를 자동 계산하지 않는다. 원본 calibration이 있어도 raw 영상의 렌즈 왜곡, 지면 평면, extrinsic convention, 공통 좌표계와 세계 단위, 이미지 resize/crop을 검증해야 한다. 파일을 쓰는 것만으로 이 조건을 충족할 수 없다. 원본은 ankle confidence 둘 다 >0.5면 평균을 쓰고, 그렇지 않으면 bbox의 `(x+0.5w,y+0.95h)`를 쓴다(`STCRA/file_utils.py:23–46`). timestamp skew가 없거나 허용 오차를 넘으면 별도 blocker를 기록한다.

## 결과 읽기 계약

`read_results(Path(...), format="aic9"|"mot10"|"mot16")`는 모델의 0기준 frame을 유지하고 다음 레코드를 반환한다.

```python
{"camera_id": 1, "track_id": 1007, "frame": 0,
 "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
 "score": None, "world_xyz": None}
```

- `aic9`: 정확히 9열. camera ID는 양수이며 `scene*1000+id` 값은 그대로 보존한다. 최종 2열은 원본의 `-1,-1`만 허용한다. 2D 세계 좌표를 임의의 Z와 합쳐 3D로 만들지 않는다.
- `mot10`: 정확히 10열 `frame,id,x,y,w,h,score,world_x,world_y,world_z`. camera는 `None`이며 부모 CLI에서 **`--camera` 필수**다. world가 모두 `-1`이면 `None`; 그 외 유한 좌표는 보존한다. score `-1`은 `None`으로 표현한다.
- `mot16`: MOT10 뒤 ankle 6열을 추가로 검증한다. ankle 정보는 공통 추적 레코드에 포함하지 않는다.
- frame와 track ID는 음수가 아닌 정수, bbox는 유한 수와 양의 폭·높이, confidence는 `[0,1]`이어야 한다. 정수값을 나타내는 `7.0` 등은 허용한다. NaN/Inf, 누락 열, 과잉 열, 빈 필드·빈 행, 비정수 ID, 음수 frame, 잘못된 CSV는 파일·행번호를 포함한 오류로 거절한다. 이미지 경계 밖 bbox는 clipping하지 않는다. header는 허용하지 않는다.
- 파일명에서 camera와 ID scope를 추측하지 않는다. SCT의 ID는 **camera-local**이고, clustering 이후는 **scene-global**이다. 사용자/부모 CLI가 scope를 명시해야 한다. 원본 frame 복원과 sequence·camera namespace는 부모 변환기의 책임이다.

합성 소입력으로 프로파일 JSON 직렬화·경로·frame 0·영상명·보정 미생성을 검증했다. aic9/mot10/mot16 및 2^53보다 큰 ID 보존의 4개 유효 사례와 11개 잘못된 입력 거절 사례를 모두 통과했다. 실제 SCOUT 추론·학습·성능 평가는 실행하지 않았으며 원본 UWIPL 저장소와 데이터·가중치는 수정하지 않았다.

## 2026-10-26 추가 적용 검토

원본 코드를 다시 읽어 아래 제약을 확인했다. 입력 배치와 결과 변환은 구현되어 있지만, 아래 실행 오류 수정·추적 규칙 일반화·보정 연결은 **미구현**이다. 검출 TXT, ReID NPY, SCT, global ID와 포즈는 모델 실행으로 생성할 중간 산출물이며 SCOUT GT로 대체하지 않는다. 전체 비교는 [SCOUT–모델 호환성 검토](../scout-model-compatibility.md)를 참고한다.

- **32프레임 clip의 anchor 부재:** S001 clustering의 anchor 목록은 `[0,5000,...,50000]`이고 실제 검출 조회는 `frame-1`이다. 모델 frame 0..31만 있는 export에서는 `-1,4999,...`에 검출이 없어 `all_embs=None`으로 clustering을 호출하게 된다. 짧은 구간의 anchor 선택은 별도 설계가 필요하다. 근거: `BoT-SORT/tools/aic_hungarian_cluster_S001.py:110–152,290`.
- **7개 카메라와 특정 인물 규칙:** SCT threshold는 c001..c007만 정의되어 c008 이상에서 조회 오류가 발생하고 c005에는 별도 ROI 제거가 적용된다. S001 clustering은 7개 길이의 설정을 camera 정렬 순서로 사용하며, c001/c002의 특정 인물 진입을 찾아 ID 8을 부여한다. 카메라 이름을 바꾸거나 누락 데이터를 채우는 것으로 해결할 조건이 아니다. 근거: `BoT-SORT/tools/aic_track.py:208–214,251–269`; `aic_hungarian_cluster_S001.py:274,302–345,358–360,449–454`.
- **STCRA에는 global ID가 먼저 필요:** STCRA는 같은 ID와 frame의 카메라 관측을 합친다. 따라서 clustering 이후 scene-global ID를 가진 결과에 포즈를 연결해야 하며, camera-local SCT를 그대로 넣으면 서로 다른 사람을 같은 ID로 취급할 수 있다. 포즈 추론 자체는 global ID를 만들지 않는다. 근거: `STCRA/run_stcra.py:67–96`; `mmpose/demo/load_tracking_result.py:10–15`.
- **ReID frame 0·이미지 cache 오류:** S001 추출기는 `cur_frame=0`으로 시작하므로 첫 검출이 frame 0이면 이미지가 초기화되지 않는다. cache 조건에 camera가 없어 camera가 바뀌어도 frame이 같으면 이전 이미지를 재사용한다. frame 번호를 임의로 바꾸는 대신 실행 코드에서 `(camera,frame)`을 기준으로 처리해야 하며 현재 수정하지 않았다. 근거: `deep-person-reid/torchreid/aic_extract_S001.py:69–86`.
- **보정 생성과 적용은 별개:** 원본은 외부 calibration JSON을 읽지 않고 `cam_utils.HOMO["S001"]`을 사용한다. 지면·세계 단위·왜곡 처리·동기화가 검증되면 평면 H 생성은 조건부로 가능하지만, 생성 파일을 원본에 연결하는 작업과 카메라 가중치·거리 기준 검토가 남는다. `Z=0` 평면의 `H=K[r1 r2 T]`를 왜곡된 raw pixel에 곧바로 적용해서는 안 된다. 근거: `STCRA/cam_utils.py:3–51`; `run_stcra.py:92,102–104,290–292`. 현재 프로파일의 설정 JSON은 원본 모델이 자동으로 읽는 설정이 아니다.
- **해상도는 조건부 제약:** 위의 해상도 수정 요구는 입력 크기나 전처리가 달라지는 경우에 해당한다. SCOUT raw 1920×1080을 유지하고 공식 `test_size=(800,1440)`을 쓰면 detector의 고정 scale과 일치한다. resize/crop/undistortion 또는 test size 변경 시 bbox 좌표와 보정을 함께 점검해야 한다. 근거: `BoT-SORT/tools/aic_get_detection_S001.py:134–135,157,173,181–184`; `yolox/exps/example/mot/yolox_x_mix_det.py:21–22`.

경로·프레임·열 형식 변환은 어댑터의 범위지만, anchor 선택·특정 인물 처리·카메라별 임계값을 SCOUT에 맞게 일반화하는 작업은 **추적 방법의 변경을 포함한다**. 원본 저장소를 보존하는 별도 실행 계층을 만들더라도 변경 내용과 실험 조건을 기록해야 한다. 이번 추가 검토에서는 문서만 수정했으며 실제 SCOUT 모델 추론으로 검증한 것은 아니다.


## 2026-10-31 지면 좌표 후보 연결

[v0.5 모델 출력 연결](../scout-model-ground.md)에 지원 형식, 명시적 점 선택 정책, 원본 프레임 대응과 실패 처리를 정리했다. 실제 SCOUT 기하와 합성 native 자료로 검증했으며 실제 모델 추론·추적은 수행하지 않았다. 원본 보정 H/추적 알고리즘 적용은 여전히 별도 작업이다.
