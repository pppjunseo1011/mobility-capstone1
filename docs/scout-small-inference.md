# SCOUT 소량 실제 추론 — 2026-10-31

**세 모델의 검출·포즈·ReID 9단계에서 실제 가중치 로딩과 GPU 추론을 완료했다.** 기존 환경 import 검사와 달리 SCOUT 이미지로 예측을 생성했다. 카메라 내 ID 추적(SCT), 카메라 간 ID 연결(MTMC), HOTA/IDF1 등의 성능 평가는 실행하지 않았다.

## 입력과 실행 범위

- `sequence_1`, `cam_0~cam_7`, 카메라당 원본 frame 277·278 두 장: 총 16장, 1920×1080.
- 기존 검증된 어댑터 export `outputs/scout-first8-20s-v03-20261027/sequence1-posetrack/manifest.json`의 원본 경로·프레임·카메라·timestamp 대응을 이용해 선택했다. JPEG를 소량 복사하여 컨테이너에 읽기 전용으로 연결했다. 동기화 정확도를 새로 검증하거나 이미지를 왜곡 보정하지는 않았다.
- 입력 선택·실행 전후 원본 JPEG SHA-256 일치를 확인했다. GT 주석은 입력으로 사용하거나 컨테이너에 연결하지 않았다.
- GPU 1: PoseTrack. GPU 0: YACHIYO, UWIPL. 한 번에 한 단계씩 실행했고 매 단계 시작 전 여유 GPU 메모리를 확인했다.
- 기존 이미지 사용, 컨테이너 CPU 2개·메모리 10GB·네트워크 차단, 가중치 읽기 전용, 출력 폴더만 쓰기 가능. 추가 의존성 설치·이미지 빌드·가중치 다운로드는 없었다.
- 원본의 AI City 전체 장면 실행 스크립트 대신 별도 소량 실행기를 사용했다. 원본 소스 3개 저장소는 변경 없이 보존했다.

## 결과

| 모델 | 검출 | 포즈 | ReID |
|---|---|---|---|
| PoseTrack | 16장, 250행 | 선택된 141행 × 17관절 | MGN `aic24.pkl`: 141 × 2048 |
| YACHIYO | 16장, 250행 | 선택된 141행 × 17관절 | OSNet: 141 × 512 |
| UWIPL 실영상 경로 | 16장, 250행 | 선택된 141행 × 17관절 | OSNet 5개 각각 141 × 512 |

**141은 서로 다른 사람 수가 아니라 프레임별 검출 행의 합계**다. 같은 사람이 다른 프레임·카메라에서 반복될 수 있다. 세 모델이 동일한 YOLOX 체크포인트를 쓰므로 검출 수가 같은 것은 세 MTMC 방식의 성능이 같다는 의미가 아니다.

- 9단계 모두 `PASS`. 각 단계에서 GPU 텐서 연산과 실제 예측을 수행했다.
- 검출 bbox, 포즈, ReID의 행 수·순서·원본 frame 대응을 별도 CPU 컨테이너에서 검사했다. NaN/Inf 없음, ReID 특징의 0벡터 없음.
- YOLOX와 HRNet 체크포인트는 모델 tensor key/shape가 모두 일치했다. OSNet은 추론 시 사용하지 않는 학습 ID 분류기 `classifier.weight/bias`만 제외했으며 나머지 key/shape 일치를 확인했다. 이 분류기 제외는 기존 FeatureExtractor의 특징 추출 사용 방식에 맞춘 것이다.
- PoseTrack은 공식 배포 `aic24.pkl`을 원본 코드처럼 MGN 모듈로 로딩하고 `reid_inferencer.process_frame_simplified`를 사용했다. 전후 좌우 반전 특징 평균도 원본 함수가 수행한다.
- 실제 모델 단계별 보고서의 소요 시간에는 모델 로딩·전처리·출력 저장이 포함된다. 소량 실행이며 워밍업·반복 측정이 없으므로 FPS 성능 비교 자료로 사용하지 않는다.

## 원본 실행과 다른 제한

검출은 각 환경의 YOLOX 실험 설정(800×1440, `test_conf=0.001`, NMS 0.7)과 원본 전처리를 사용했다. 원본 저장 코드의 objectness >0.1 필터를 적용한 250행을 `detections.json`에 보존했다. bbox는 실제 영상 경계로 자르고 정수 좌표로 변환했다.

포즈·ReID 계산량을 제한하기 위해 **objectness ≥0.3인 검출 중 프레임당 상위 10개**만 후속 입력으로 선택했다. confidence는 원본 코드가 저장하는 objectness 기준이며 objectness×class probability와 구분한다. 원본 전체 파이프라인의 누락 없는 재현이나 최적 임계값 튜닝 결과가 아니다. 가려진 사람·작은 사람·많은 사람이 있는 프레임에서는 이 제한이 결과에 영향을 준다.

UWIPL은 실영상용 YOLOX·OSNet 5개·HRNet 경로를 검사했다. 추가로 다운로드한 합성 영상용 YOLOv7 5개는 이번 실행 대상이 아니며, 미확보 합성 ReID 가중치도 사용하지 않았다. YACHIYO/UWIPL의 포즈는 이번 진단에서 검출 bbox를 직접 입력했다. 원본처럼 추적 결과를 먼저 만들어 넣는 전체 흐름은 아직 실행하지 않았다.

## 실제 결과와 기존 어댑터 연결

PoseTrack 결과를 원본 출력 규격에 맞는 카메라별 검출 7열·포즈 57열로 저장했다. 기존 어댑터 v0.5의 `prepare_model_points`로 8개 카메라, 141행을 읽어 검출·포즈 pairing과 source frame 277·278 대응을 확인했다.

- 양쪽 발목 신뢰도 >0.7: **78행 `ready`**.
- 낮은 발목 신뢰도: **63행 `low_ankle_confidence`**. 삭제하거나 임의 좌표로 대체하지 않았다.
- 사람 ID는 생성하지 않았고 `id_scope=none`을 유지했다.
- 이번에는 입력 연결까지만 수행했으며 지면 메시 교차, P/H 생성, 세계좌표 정확도 검증은 실행하지 않았다.

기존 어댑터의 `input_provenance_verified=false`, `model_inference_run=false`는 어댑터 자체가 추론 출처를 증명하거나 추론을 실행하지 않는다는 보수적 표시다. 필드를 임의로 바꾸지 않았다. 이번 실제 추론의 출처는 상위 실행 보고서·가중치 manifest·이미지 ID·입력 SHA-256·native 출력 파일 해시로 별도 기록한다. YACHIYO SCT/UWIPL MOT16 연결은 실제 추적 ID 결과가 필요하므로 이번 검증에서 수행하지 않았다.

## 파일과 재실행

결과 루트: `outputs/scout-inference-small-20261031/`.

- `input/manifest.json`: 표본 경로, source frame, timestamp, 이미지 SHA-256.
- `<모델>/detection/detections.json`: 원본 필터 후 모든 bbox와 후속 선택 bbox.
- `<모델>/pose/poses.json`: bbox와 17개 관절 좌표·점수.
- `<모델>/reid/*.npy`, `*-rows.json`: 특징 배열과 검출 행 대응.
- `<모델>/<단계>/report.json`, `run.log`: 로딩 검사, 출력 수·차원, GPU 사용 정보, 실행 명령·이미지 ID.
- `posetrack/pose/cam_N-det7.txt`, `cam_N-pose57.txt`: 어댑터 입력용 native 형식.
- `validation/cam_N/`: 실제 PoseTrack 출력의 원본 frame 대응과 발목 점 선택 결과.
- `summary.json`, `validation/validation.json`: 9단계 실행 및 별도 출력 검사 요약.
- `runner-history/`: 실행 당시 소량 실행기 원문을 SHA-256 이름으로 보존.

[종합 검증 기록](../logs/scout-small-inference-20261031.json), [추론 실행기](../scripts/infer-scout-sample.py), [컨테이너 내부 추론 코드](../scripts/scout-sample-stage.py), [출력 검증 코드](../scripts/validate-scout-inference.py).

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 scripts/infer-scout-sample.py \
  --data datasets/scout/extracted/scout \
  --adapter-export outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \
  --output outputs/scout-inference-small-new \
  --gpu-posetrack 1 --gpu-others 0 --frames-per-camera 2
```

GPU 번호와 출력 경로는 실행 시 지정한다. `--models posetrack`처럼 모델을 제한할 수 있다. 새 output을 기본으로 사용한다. `--resume`은 같은 표본 manifest에서 이미 PASS한 단계를 재사용하며 재추론하지 않는다. 실행기나 모델 설정을 바꾼 뒤에는 새 output에서 검증해야 한다. 초기 실행은 카메라당 2프레임이었다. 2026-11-01 OSC 확장에서는 실행기 상한을 200프레임/카메라로 늘리고 `--start-index`를 추가했다. 기존 실행 결과는 변경하지 않았다.

시각화 예: [PoseTrack 검출](../outputs/scout-inference-small-20261031/posetrack/detection/cam_0_000000.jpg), [PoseTrack 포즈](../outputs/scout-inference-small-20261031/posetrack/pose/cam_0_000000.jpg), [YACHIYO 포즈](../outputs/scout-inference-small-20261031/yachiyo/pose/cam_0_000000.jpg).

## 다음 단계

이번 결과로 실영상 입력·가중치·핵심 추론 환경이 동작함을 확인했다. 다음에는 더 긴 짧은 구간에서 카메라별 추적 ID를 생성하고, SCOUT 좌표계·왜곡·시간 동기화·지면/거리 단위와 모델별 공간 규칙을 정리하여 카메라 간 ID 연결을 검증한다. 그 이후에 공통 결과 변환과 HOTA/IDF1 평가를 수행한다. 현재 2프레임 표본만으로 추적 성능이나 모델 우열을 판단하지 않는다.
