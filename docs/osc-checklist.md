# OSC 인원 제한 해제 재검증 — 2026-11-01

프레임당 10개 상한을 해제하고 sequence_1 / cam_0~cam_7 / source frame 277~326의
400장을 검출·Pose·ReID부터 다시 처리했다. SCPT·MCPT는 2회 실행하고 평가·시각화를 갱신했다.
현재 결과: [outputs/osc-pilot-uncapped-20261101](../outputs/osc-pilot-uncapped-20261101).
이전 10개 제한 결과는 `outputs/osc-pilot-20261101`에 비교 근거로 보존한다.

## 인계 자료

- [5쪽 PDF](handoff/OSC_SCOUT_인계서_제한해제_20261101.pdf)
- [전달 ZIP](../deliverables/OSC_SCOUT_handoff_uncapped_20261101.zip)
- [읽는 순서](handoff/START_HERE.txt)

이전 PDF·HTML·ZIP은 새 전달 자료 검증 후 제거한다. 기존 상세 문서는 이 파일로 갱신하며,
실행 근거가 되는 로그·원본 결과는 보존한다. 다른 PC 재실행용 Docker·가중치는 ZIP에 포함하지 않는다.

## 11개 항목 상태

| 번호 | 항목 | 상태 | 결과 |
|---|---|---|---|
| 01 | SCOUT 프레임·GT | 완료 | 대표 GT BBox 시각화 24장 |
| 02 | 검증 구간 | 완료 | sequence_1 / cam_0~7 / 원본 frame 277~326 |
| 03 | 카메라·프레임 매핑 | 완료 | camera_0001~8 / OSC frame 1~50 |
| 04 | 검출·임베딩 | 완료 | 5,569×512 · 개수 제한 없음 · NaN/Inf 없음 |
| 05 | Pose 대응 | 완료 | 5,569개 BBox–Keypoint serial 대응 통과 |
| 06 | World·별도 GT | 부분 완료 | 위치 오차 비교 완료 / 물리 단위·보정 적합성 미확정 |
| 07 | SCPT·Local ID | 완료 | 5,329행 할당 · 보정 전후 결과 보존 |
| 08 | MCPT·Global ID | 제한 실행 | 612행 · 9 ID / 공간 거리 보정 OFF |
| 09 | 공통 CSV | 완료 | camera/frame/bbox/id 검사 · 중복 없음 |
| 10 | 프로토콜별 평가 | 파일럿 완료 | 2D·multi-view 분리 / 공식 SCOUT·HOTA 미실시 |
| 11 | 반복·기록 | 완료 | 추적 2회 · 최종 JSON 해시 동일 |

## 동일 조건 비교

| 항목 | 이전 최대 10개 | 이번 개수 상한 없음 |
|---|---:|---:|
| 후속 입력 검출 수 | 3,826 | 5,569 |
| Local ID 할당 행 | 3,758 | 5,329 |
| 최종 Global ID 행 | 566 | 612 |
| Global ID 수 | 8 | 9 |
| Local 2D IDF1 | 34.60% | 36.09% |
| Local 2D recall | 40.91% | 51.19% |
| Local 2D MOTA | 19.66% | 14.67% |
| Local ID 전환 횟수 | 960 | 1,328 |
| Global 결과의 카메라별 2D IDF1 | 12.41% | 13.39% |
| Multi-view Global IDF1 | 11.35% | 12.34% |

`comparison.json` 및 `comparison.csv`에 비교 수치를 저장했다. 원시 검출 6,412개,
400장의 이미지 SHA-256, GT, 카메라 매핑, calibration/profile, 추적 parameters가 기존과 같은지 검사했다.
신뢰도 조건을 통과한 모든 검출을 선택했고 이전 상위 10개와의 정확 일치도 확인했다.
400장 중 263장에서 10개를 초과하며 실제 선택 수는 5~24개이다.

Local recall·IDF1과 multi-view IDF1은 상승했지만 Local MOTA는 하락하고 ID 전환은 증가했다.
인원 수 상한 해제만으로 추적 단절이 해결됐다고 해석하지 않는다.

## 유지한 조건과 해석

- objectness ≥0.3, 검출기 NMS와 가중치 유지. 개수 상한만 `10 → null`로 변경.
- 약 10 FPS / 5초 / GT 7,589행·43 ID. 첫 8카메라 주석에 보간 포함.
- 원본 frame−276=OSC frame, cam_0~7→camera_0001~8. timestamp 최대 편차 약 0.032초.
- `fps=10`, `short_track_th=40` (4초), `epsilon_scpt=.10`, `epsilon_mcpt=.37`, `time_period=3` 유지.
- 공간 거리 보정·World 기반 삭제/보간·Global 재할당 OFF. 원본 OSC 전체 설정 재현은 아님.
- GT는 시각화·평가에만 사용하고 추적 컨테이너에는 연결하지 않음.
- Local 미할당 240행, Global 보정 단계 제거 4,957행.
  raw 결과와 삭제 목록 보존. 임의 ID 생성 없음.
- cam_5에는 최종 Global 출력 없음. 카메라별 Global 행: cam_0: 187 · cam_1: 50 · cam_2: 46 · cam_3: 50 · cam_4: 43 · cam_5: 0 · cam_6: 145 · cam_7: 91.

## World 비교

예측 BBox 하단 중심 → 왜곡 보정 → ground mesh 교차를 먼저 계산했다.
이후 별도 MOT 주석 XYZ와 IoU≥0.5 일대일 매칭으로 비교했다.
좌표 상태: {'outside_image': 58, 'hit': 4957, 'no_ground_intersection': 554}.
BBox 대응 3,973쌍 중 World 비교 3,951쌍.
XYZ 오차 평균 / 중앙값 / p95: **1.524 / 0.660 / 6.797 데이터셋 단위**.
대응된 부분집합의 사람 위치 오차이며 독립 측량에 의한 보정 오차가 아니다.
m/cm 단위·허용 오차·카메라 보정 적합성은 미확정이다. 제한 해제 전후는 대응 대상 수도 달라져
World 오차 변화만으로 보정 품질 변화를 판단하지 않는다.

## 평가·반복 검증

motmetrics 1.4.0으로 카메라별 2D와 multi-view Global identity를 분리했다.
2D는 같은 camera/frame의 IoU≥0.5, 전체 50프레임과 빈 예측을 포함한다.
Multi-view는 각 camera/frame에서만 BBox 대응하되 모든 view에 하나의 ID 대응을 계산한다.
서로 다른 카메라의 픽셀 좌표를 직접 매칭하지 않는다. World XY 진단은 별도 프로토콜이다.
공식 SCOUT/HOTA 미실시. 4개 프로토콜 self-test와 CSV 일치·중복 검사를 통과했다.

추적 2회 시간: 7.639초 / 7.525초.
최종 fixed_whole SHA-256: `97a66d955e9fa6fa3d9f64a65d0d039e651040e7efd2676396bd05bfec51972a` (두 번 동일).
신경망 추론을 재사용한 추적 시간이며 전체 FPS 또는 추론 2회 반복 결과가 아니다.

모델 커밋: `f881fe0edf60fe697efb2bf4c935b80a80861230`.
`provenance/evidence.json`에 가중치·데이터·스크립트·출력 해시, `prepare.log`, `track.log`,
`evaluate.log` 및 추론 단계 `run.log`에 Docker 이미지 ID와 실제 명령을 저장했다.

## 기존 서버에서 재실행할 경우

각 `*-new`는 존재하지 않는 새 경로를 사용한다. GPU와 데이터 경로는 실행 시 지정한다.

```bash
python3 scripts/infer-scout-sample.py \
  --data datasets/scout/extracted/scout \
  --adapter-export outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \
  --output outputs/osc-inference-uncapped-new --models yachiyo \
  --gpu-posetrack 1 --gpu-others 0 --frames-per-camera 50 --max-people-per-frame 0
python3 scripts/run-osc-pilot.py prepare --output outputs/osc-pilot-uncapped-new \
  --inference-run outputs/osc-inference-uncapped-new
python3 scripts/run-osc-pilot.py track --output outputs/osc-pilot-uncapped-new
python3 scripts/run-osc-pilot.py evaluate --output outputs/osc-pilot-uncapped-new
```

`--max-people-per-frame 0`이 개수 제한 해제이다. 과거 실험 호환을 위해 일반 스크립트 기본값 10은
유지했으며 이 실행은 명시적으로 0을 전달했다. resume 시 선택 정책이 다르면 중단한다.
자료 생성/감사 스크립트는 이번 고정 결과 경로를 사용한다.

다음 단계: 더 긴 연속 구간, Local 단절·Global 미할당 분석, World 단위·실패 처리 확정,
공간 조건 MCPT, 독립 검증 구간 및 공식 SCOUT/HOTA 평가 조건 확정.
