# SCOUT 어댑터 v0.4 — 원본 픽셀과 지면 메시의 좌표 연결

**후속 작업(2026-10-31):** [v0.5 모델 출력 연결](scout-model-ground.md)에서 native 파일을 읽어 본 기하 기능에 연결하는 명령을 추가했다. 아래는 v0.4의 보존 기록이다.

2026-10-28. `sequence_1 / cam_0~cam_7`에서 가중치 없이 가능한 후속 작업으로 **왜곡을 고려한 원본 픽셀 → 지면 메시 좌표 변환**을 구현했다. 이전 v0.3 기능은 유지한다. 사람의 발 위치 검출, 실제 모델 추론·성능 평가는 수행하지 않았다.

## 구현한 기능

- `geometry-probe`: 기존 export의 카메라·해상도·프레임 정보를 읽고 화면 격자점으로 실제 보정과 지면 메시를 검사한다. `input_kind=synthetic_grid`이며 실제 사람 위치가 아니다.
- `project-ground`: **원본 왜곡 영상의 픽셀 좌표 CSV**를 동일한 기하 경로로 변환한다. 카메라·원본 프레임 번호를 export와 대조하고 timestamp를 연결한다.
- 계산 순서: 왜곡 역변환 → 카메라 중심에서 출발하는 세계 좌표계 광선 → 지면 메시의 가장 가까운 전방 교차점 → 원본 영상으로 재투영하여 수치 오차 확인.
- 실패 상태는 `no_ground_intersection`, `undistort_not_converged`, `reprojection_failed`로 구분한다. 실패 행의 세계좌표는 빈 값이며 0이나 평면 좌표로 대체하지 않는다.
- 실제 지면 메시의 **497,662개 정점과 984,425개 삼각형 면**을 모두 읽는다. 삼각형 경계 상자를 계층화한 BVH로 탐색 범위를 줄인다. 면 방향과 관계없이 가장 가까운 양의 교차점을 선택한다.
- OpenCV 왜곡 역변환은 최대 100회, 종료 오차 `1e-12`를 사용한다. 픽셀 왕복 오차가 `1e-4px`보다 크면 좌표를 내보내지 않는다.

데이터와 기존 export는 읽기 전용이다. 원본 보정·프레임·시각이 export와 다르면 중단한다. 입력·출력 경로 겹침, 기존 출력 덮어쓰기, 중복 point ID, 영상 범위 밖 픽셀도 거부한다. 최대 4,096개 점을 처리하며 영상 재생성이나 원본 복사는 하지 않는다.

## 실제 데이터 검사

기존 `outputs/scout-first8-20s-v03-20261027/sequence1-posetrack`의 manifest를 사용했다. 형식이 공통이므로 특정 모델의 추론 결과에 의존하지 않는다. 카메라마다 첫 원본 프레임 277에 21×13 격자점을 대응시켰다. **이미지에서 사람이나 지면을 인식하여 고른 점이 아니다.**

| 항목 | 결과 |
|---|---|
| 전체 화면 격자 | 273점 × 8카메라 = 2,184점 |
| 지면 메시와 교차 | 1,043점 |
| 지면 교차점 없음 | 1,141점, 세계좌표 빈 값 |
| 왜곡 역변환 실패 / 재투영 실패 | 0 / 0 |
| 왜곡 역변환 후 픽셀 왕복 최대 오차 | 약 `9.92×10⁻¹³px` |
| 교차점의 원본 영상 재투영 최대 오차 | 약 `2.12×10⁻¹⁰px` |
| 메시 탐색 노드 / 무시한 면적 0 삼각형 | 32,767개 / 0개 |
| 새 기하 검사 / 전체 자동 검사 | 24개 / 119개, 모두 통과 |

카메라별 교차 / 비교차 개수는 cam_0: 104/169, cam_1: 88/185, cam_2: 114/159, cam_3: 125/148, cam_4: 122/151, cam_5: 188/85, cam_6: 148/125, cam_7: 154/119다. 화면에는 지면 메시와 만나지 않는 방향이 포함되므로 **교차 비율은 검출·추적 정확도가 아니다.**

같은 격자를 `points.csv`로 다시 입력하여 공개 `project-ground` 경로도 확인했다. 두 경로의 `world_points.csv`는 바이트 단위로 동일하다. 원본 보존과 코드 해시는 [종합 검증 기록](../logs/adapter-v04-validation.json)에 기록했다.

실행 이미지: `scout-mtmc/adapter:0.4`, 빌드 ID `sha256:bf9dab699aed5769713937ac213e9925ec61d74664116dd49f4604dd6aa3a533`. 기존 기반 이미지의 NumPy 1.23.5 / OpenCV 4.8.1을 사용한다. CPU 2개·RAM 4GiB, 네트워크·GPU 차단으로 실행했다. 새 의존성·가중치를 다운로드하지 않았으며 v0.3과 기존 모델 이미지 9개는 보존한다.

## 사용 방법

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 scripts/adapter.py build
python3 scripts/adapter.py test

# 화면 격자 검사. 출력은 새 경로를 지정한다.
python3 scripts/adapter.py geometry-probe \
  --data datasets/scout/extracted/scout \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \
  --output outputs/ground-probe-new \
  -- --grid-columns 21 --grid-rows 13

# 실제 모델의 발 위치 등 원본 픽셀 좌표가 준비되면 사용한다.
python3 scripts/adapter.py project-ground \
  --data datasets/scout/extracted/scout \
  --workspace outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \
  --input /실제/원본픽셀좌표.csv \
  --output outputs/ground-points-new \
  -- --input-kind model_output
```

입력 CSV 헤더는 아래와 같아야 한다. `source_frame`은 모델의 0/1기준 번호가 아닌 **원본 SCOUT 프레임 번호**다. `point_id`는 같은 camera/frame 내 점 구별자이며 카메라 간 사람 ID를 뜻하지 않는다.

```csv
camera,source_frame,point_id,pixel_x,pixel_y
```

출처는 `model_output`, `synthetic_fixture`, `unverified` 중 선택한다(기본 unverified). 호출자의 선언으로 실제 추론 여부를 증명하지 않으므로 `input_provenance_verified=false`를 유지한다. 격자 CSV 재실행에는 `synthetic_fixture`를 사용했다. GT를 읽거나 bbox의 아래쪽 중심을 임의로 발 위치로 선택하지 않는다.

출력은 `points.csv`(입력 점), `world_points.csv`(상태·XYZ·교차 면·원본 시각·오차), `geometry.json`(집계·파일 해시·가정·한계)이다. 모델 추적 결과인 `predictions.csv`와 분리한다.

## 적용 범위와 남은 연결

[이전 기하 진단](scout-geometry.md)에서 남겨 둔 광선–메시 교차와 왜곡 역변환 정밀도를 이번에 다뤘다. 고정된 SCOUT toolkit의 world→camera `R,T`, 카메라 중심 `−RᵀT`, 왜곡을 포함한 투영 관계를 따른다. NumPy/OpenCV로 구현했으며 공식 코드를 복사하거나 trimesh 의존성을 추가하지 않았다. toolkit의 최소 깊이 1 및 높이 0.8 규칙은 단위와 적용 근거를 확인하지 않았으므로 도입하지 않았다.

재투영 오차는 같은 보정 모델을 이용한 **수치적 일관성**이다. 실제 보정 정확도, 세계 단위가 미터인지, 실제 보행 영역, 가림 여부를 검증한 것이 아니다. 지면 전용 메시만 검사하므로 건물·사람·나무 등에 가려진 지면도 교차할 수 있다. 보행 영역 및 가림 처리는 별도로 결정해야 한다.

모델용 H를 만들거나 원본 추적 알고리즘을 변경하지 않았다. YACHIYO의 WorldCoordinate, UWIPL STCRA의 하드코딩 H, PoseTrack의 P/H 및 발 높이 가정과 연결하는 작업은 남아 있다. 단일 평면 H를 전제로 하는 모델에 비평면 메시 기반 XYZ를 넣는 것은 형식 변경만으로 끝나지 않는다. 실제 포즈/발 좌표가 준비되면 원본 프레임·픽셀 기준으로 연결해야 하며, 모델의 resize/왜곡 보정 좌표는 먼저 역변환해야 한다.

구현: [geometry.py](../adapters/scout_adapter/geometry.py), [ground.py](../adapters/scout_adapter/ground.py), [기하 테스트](../tests/test_geometry.py).

로그: [빌드](../logs/adapter-v04-build.log), [전체 검사](../logs/adapter-v04-tests.log), [실제 격자](../logs/scout-first8-ground-v04-20261028.log), [CSV 입력](../logs/scout-first8-ground-csv-v04-20261028.log).
