# SCOUT → PoseTrack 1차 어댑터

분석 대상은 [PoseTrack `master`, 커밋 `ba727e6`](https://github.com/ZhenyuX1E/PoseTrack/tree/ba727e6bc6e45e44cd6e006b73e00864892173c2)이다. 구현은 [posetrack.py](../../adapters/scout_adapter/profiles/posetrack.py)에 있으며 Python 표준 라이브러리만 사용한다. 모델 환경 검증 결과는 [기존 환경 문서](../posetrack.md)에 있다.

이번 구현은 입력 배치 계획과 추적 결과 파서다. 실제 가중치 로딩·추론·성능 평가를 수행하지 않으며, 프로파일의 `ready`는 `false`다. 보정행렬은 자동 생성하지 않는다. 영상 생성과 이미지 복사, 원본 프레임 역매핑은 공통 어댑터가 담당한다.

## 입력 배치와 프레임 매핑

`build_profile(manifest)`는 모든 경로를 export 출력 디렉터리에 대한 상대 경로로 반환한다. 예를 들어 `sequence_number=1`, 첫 번째 모델 카메라 ID가 1이면 다음 배치를 계획한다.

```text
config/posetrack_adapter.json
dataset/test/scene_001/camera_0001/frames/000000.jpg
dataset/test/scene_001/camera_0001/frames/000001.jpg
result/detection/scene_001/
result/pose/scene_001/
result/reid/scene_001/
result/track/
result/track_log/
```

`config/posetrack_adapter.json`은 어댑터 설명용 JSON이며 원본 모델이 읽는 설정 파일이 아니다. `frames/`도 중간 이미지 배치용이다. 실제 검출·포즈·ReID 코드는 다음 파일을 읽는다.

```text
dataset/test/scene_001/camera_0001/video.mp4
```

`video.avi` 또는 JPEG 폴더를 그대로 연결할 수 없다. 공통 어댑터에서 명시적으로 영상을 생성한 뒤 디코딩 순서와 프레임 수를 확인해야 한다. 영상·보정 파일이 존재하는 것만으로 실제 추론 성공을 뜻하지 않는다.

프레임 번호는 영상의 첫 디코딩 프레임이 0이다. 예를 들어 선택한 원본 프레임이 `[81, 82]`이면 staging 이름은 `000000.jpg`, `000001.jpg`이고 모델 결과 프레임은 0, 1이다. 공통 manifest를 이용해 다시 81, 82로 복원한다. 선택 구간이나 stride가 달라도 원본 프레임 번호를 결과에 직접 넣으면 안 된다.

카메라 이름은 `camera_0001`처럼 네 자리 접미사를 사용한다. 프로파일은 `model_camera_id`를 1–9999로 제한하고 중복을 거부한다. 원본 `cam_0` 및 timestamp 키와의 연결은 공통 manifest에 유지한다. 원본 코드는 카메라 폴더명 마지막 네 문자만 정수로 읽으므로 원래 SCOUT 이름을 그대로 쓰면 안 된다.

근거: [검출 경로·0-based 번호·CSV 저장, 175–267행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/detection/get_detection.py#L175), [포즈 video.mp4 입력, 173–211행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/mmpose/demo/save_pose_with_det_multiscene.py#L173), [ReID 입력, 131–151행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/fast-reid/tools/infer.py#L131), [카메라 ID 파싱, 22–23행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/util/camera.py#L22).

## 단계별 중간 산출물

| 단계 | 파일 | 형식 |
|---|---|---|
| 검출 | `result/detection/scene_001/camera_0001.txt` | 쉼표 구분 7열: `frame,class=1,x1,y1,x2,y2,score` |
| 포즈 | `result/pose/scene_001/camera_0001.txt` | 공백 구분 57열: `frame,x1,y1,x2,y2,score`와 COCO 순서 17개 관절의 `(x,y,score)` |
| ReID | `result/reid/scene_001/camera_0001.npy` | 검출 행 순서에 맞춘 `N×2048` 특징 |
| 추적 | `result/track/scene_001.txt` | 공백 구분 10열, 아래 결과 계약 참고 |
| AIC 제출 변환 | `result/track.txt` | 공백 구분 9열; 여러 장면을 합치며 장면 열은 없음 |

포즈와 ReID에는 검출별 고유 키가 없다. 추적기는 검출의 frame 마스크를 포즈·ReID 배열에 그대로 적용하므로 카메라 집합, 카메라 파일 정렬, 검출 행 순서를 모두 일치시켜야 한다. 개별 단계에서 임의로 검출을 정렬·삭제하면 다른 사람의 특징을 연결할 수 있다. 모델 실행 전에 빈 파일과 단일행의 `np.loadtxt` 차원 처리, 검출이 없는 경우의 `np.concatenate([])` 처리도 보완해야 한다.

근거: [포즈 레코드 생성, 31–41행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/mmpose/demo/save_pose_with_det_multiscene.py#L31), [ReID 저장, 149–193행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/fast-reid/tools/infer.py#L149), [동일 행 마스크로 특징 결합, 33–90행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/run_tracking_batch.py#L33).

## 보정과 실행 준비가 필요한 부분

각 카메라에 `dataset/test/<scene>/<camera>/calibration.json`이 필요하다. 원본 코드가 사용하는 키와 방향은 다음과 같다.

| JSON 키 | 모양 | 의미 |
|---|---|---|
| `camera projection matrix` | 3×4 | 동차 world 좌표를 image 좌표로 투영하는 `P` |
| `homography matrix` | 3×3 | 지면 world 좌표 `(X,Y,1)`를 image 좌표로 보내는 `H` |

추적기는 `H⁻¹`로 박스 하단 중심을 지면에 역투영한다. 또한 `H`의 마지막 열에 `0.15 * P[:,2]`를 더해 발 위치 평면을 구성한다. 이 계산에는 동일한 world 좌표계, 지면 `z=0`, 높이 0.15의 단위가 전제되어 있다. SCOUT의 `K/R/T/dist`가 있다는 이유만으로 이 의미가 확인되는 것은 아니다. 원본 추적기의 투영에는 렌즈 왜곡 항도 없다. **이번 단계는 P/H를 합성하거나 유효한 것처럼 보이는 calibration.json을 만들지 않는다.** 이미지의 왜곡 처리와 보정 좌표의 일관성, 실제 지면·단위를 검증한 후 준비해야 한다.

근거: [P/H 읽기 및 발 평면 생성, 12–20행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/util/camera.py#L12), [지면 역투영, 151–179행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/Tracker/PoseTracker.py#L151).

추가 실행 제약은 다음과 같다.

- **해상도:** 검출의 box 복원 배율이 `min(800/1080,1440/1920)`로 고정되어 있다. ReID는 박스를 1920×1080 범위에 자르고 추적은 `y > 1078`을 화면 하단으로 취급한다. 실제 이미지 해상도를 검증하고 다르면 해당 코드와 보정행렬을 함께 조정해야 한다. 영상 생성만으로 해결되지 않는다.
- **동기화와 FPS:** 추적기는 timestamp 파일을 읽지 않고 같은 frame 번호를 동시점으로 처리한다. 동기화 오차가 미확인 또는 허용치를 넘으면 프로파일에 차단 사유를 기록한다. 원본의 3·15프레임 상태 만료 등은 시간 단위가 아니므로 실제 SCOUT FPS에 대한 영향을 별도로 검토한다.
- **경로:** 주요 경로가 `__file__`로 계산한 저장소 root 기준이다. `/data/scout`, `/outputs` 환경변수를 지정하는 것만으로 입력이 바뀌지 않는다. 별도 실행 workspace에서 내보낸 `dataset/test`, 쓰기 가능한 `result`, 읽기 전용 가중치를 연결해야 한다.
- **장면 선택:** 검출은 `--scene 1`이면 `scene_001`을 찾는다. Pose/ReID의 `--start`와 `--end`는 정렬된 장면 목록의 slice다. 단일 장면은 `--start 0 --end 1`이며 기본값 `--end -1`은 마지막 장면을 제외한다.
- **실행 스크립트:** upstream shell은 장면 범위·GPU·CPU affinity를 고정한다. 공통 자원 배정에 맞춘 별도 실행 명령이 필요하다. ReID는 작업 디렉터리 기준 `../ckpt_weight/aic24.pkl`을 읽는다.
- **가중치:** ByteTrack YOLOX, HRNet 포즈, 학습된 ReID `aic24.pkl`이 필요하다. 이번 어댑터는 다운로드나 추론을 시작하지 않는다.

근거: [검출 고정 배율, 182·247행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/detection/get_detection.py#L182), [ReID 크기 고정, 164–175행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/fast-reid/tools/infer.py#L164), [프레임 단위 만료, 956–979행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/Tracker/PoseTracker.py#L956), [ReID 장면 slice·가중치 경로, 96–112행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/fast-reid/tools/infer.py#L96), [고정 실행 배정](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/script/standrad_detection.sh#L14).

## 결과 변환 계약

`read_results(path, format="posetrack10")`와 `format="aic9"`를 지원한다. 두 형식 모두 공백 구분이며 다음 열을 읽는다.

```text
posetrack10: camera_id track_id frame x y width height world_x world_y position_source
aic9:       camera_id track_id frame x y width height world_x world_y
```

`position_source`는 1=박스 하단의 지면 위치, 2=발 관절 평면, 3=복원한 3D 관절의 XY를 뜻한다. confidence가 아니다. 파서는 이 값을 `position_source`로 별도 보존하고 `score=None`을 반환한다. world 좌표는 XY만 있으므로 `world_xy=[X,Y]`, `world_xyz=None`으로 반환하며 Z를 임의로 채우지 않는다. AIC 9열에는 위치 추정 방식이 이미 삭제되어 있으므로 이를 복원하지 않는다.

카메라 ID 1–9999, 양의 정수 track ID, 0 이상의 정수 frame, 유한한 좌표, 양수 박스 폭·높이, 10열의 위치 방식 1–3을 검사한다. upstream `np.savetxt`의 지수 표기 정수도 읽되 소수 ID는 거부한다. 행의 열 개수 오류, NaN/Inf, 잘못된 숫자는 파일·행 번호를 포함한 `ValueError`로 처리한다. 빈 줄과 `#`로 시작하는 줄은 무시하며 빈 결과 파일은 0개 레코드로 읽는다.

파서 자체는 모델 frame 번호를 0-based 그대로 반환한다. 원본 source frame과 카메라 이름으로의 복원은 공통 manifest가 담당한다. track ID는 매 scene에서 1부터 다시 시작하므로 `(sequence, track_id)`를 함께 보존해야 한다. `result/track.txt`에는 sequence 열이 없으므로 여러 sequence를 합친 제출 파일을 장면 구분 없이 공통 결과로 변환해서는 안 된다. 장면별 원본 10열 파일 사용을 권장한다.

근거: [추적 10열 출력, 985–994행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/Tracker/PoseTracker.py#L985), [장면별 ID 생성, 815–831행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/Tracker/PoseTracker.py#L815), [제출 9열 변환·정수 box 저장, 17–26행](https://github.com/ZhenyuX1E/PoseTrack/blob/ba727e6bc6e45e44cd6e006b73e00864892173c2/track/generate_submission.py#L17).

검증은 작은 가상 manifest와 임시 결과 파일로 수행했다. 상대 경로·JSON 직렬화·0-based 프레임 패턴, 9/10열 파싱, 지수 표기 정수 ID, 빈 결과, 중복 카메라 거부와 잘못된 결과 행 10종을 확인했다. 실제 SCOUT 이미지 디코딩이나 모델 실행은 이 검증에 포함되지 않는다.

이 파서의 형식 검증과 프로파일 생성 성공은 실제 모델 추론 성공이나 SCOUT 보정의 정확성 검증을 의미하지 않는다.


## 2026-10-31 지면 좌표 후보 연결

[v0.5 모델 출력 연결](../scout-model-ground.md)에 지원 형식, 명시적 점 선택 정책, 원본 프레임 대응과 실패 처리를 정리했다. 실제 SCOUT 기하와 합성 native 자료로 검증했으며 실제 모델 추론·추적은 수행하지 않았다. 원본 보정 H/추적 알고리즘 적용은 여전히 별도 작업이다.
