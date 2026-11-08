# 최종 환경 검증 기록

2026-10-25 Asia/Seoul. 아래 로그는 최종 이미지의 실제 실행 결과다. **9개 이미지 빌드, CPU 9/9, CUDA 대상 GPU 8/8 성공. 모델 가중치 추론은 실행하지 않았다.**

| 환경 | 이미지 ID 앞 12자리 | CPU | GPU | Python / PyTorch / CUDA |
|---|---|---|---|---|
| yachiyo/detection | `2094117ed652` | [PASS](../logs/yachiyo-detection-cpu-20261025T170707001856.log) | [PASS](../logs/yachiyo-detection-gpu-20261025T170708853248.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| yachiyo/reid | `dbf917dca7cf` | [PASS](../logs/yachiyo-reid-cpu-20261025T170950665944.log) | [PASS](../logs/yachiyo-reid-gpu-20261025T170951657402.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| yachiyo/pose | `9497ec27ae53` | [PASS](../logs/yachiyo-pose-cpu-20261025T170716860323.log) | [PASS](../logs/yachiyo-pose-gpu-20261025T170718737020.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| yachiyo/tracking | `84bcd82ef6ce` | [PASS](../logs/yachiyo-tracking-cpu-20261025T170011878912.log) | 해당 없음(CPU 전용) | 3.10.14 / — / — |
| uwipl/botsort | `7d0723493cbd` | [PASS](../logs/uwipl-botsort-cpu-20261025T170923181846.log) | [PASS](../logs/uwipl-botsort-gpu-20261025T170924616191.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| uwipl/mmyolo | `a4c2cffa0eab` | [PASS](../logs/uwipl-mmyolo-cpu-20261025T170951620398.log) | [PASS](../logs/uwipl-mmyolo-gpu-20261025T170952950020.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| uwipl/reid | `42d6b7fa915d` | [PASS](../logs/uwipl-reid-cpu-20261025T170956914585.log) | [PASS](../logs/uwipl-reid-gpu-20261025T170958155635.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| uwipl/pose | `1693f7cca4b6` | [PASS](../logs/uwipl-pose-cpu-20261025T171001554706.log) | [PASS](../logs/uwipl-pose-gpu-20261025T171003497536.log) (호스트 0) | 3.8.12 / 1.11.0 / 11.3 |
| posetrack/all | `da99db46dd7d` | [PASS](../logs/posetrack-all-cpu-20261025T170848940016.log) | [PASS](../logs/posetrack-all-gpu-20261025T170851427647.log) (호스트 1) | 3.9.23 / 2.0.1+cu118 / 11.8 |

모든 runtime smoke는 공통 실행 스크립트로 UID/GID 1000, 네트워크 차단, CPU 2개/메모리 8GiB에서 실행했다. CUDA 컨테이너는 지정한 GPU 한 장만 노출했다. 최종 검증 뒤 스크립트의 잘못된 경로 거부 순서를 수정했고, 명령 생성과 실제 읽기 전용 마운트/결과 저장 검증을 재수행했다. 이 수정은 이미지와 모델 의존성을 변경하지 않는다.

- 각 빌드의 `pip check` 통과. 실제 단계 진입점 import, 소스 재빌드 C++/Cython 및 MMCV 연산을 확인했다.
- [읽기 전용 입력·가중치/호스트 결과 저장](../logs/mount-validation.txt), [GPU/경로 선택 및 잘못된 경로 거부](../logs/launcher-validation.txt) 통과.
- [서버 최초 기록](../logs/server-before.txt), [서버 종료 기록](../logs/server-after.txt), [이미지·환경 파일 manifest](../logs/environment-manifest.json).
- Python 설치 목록: `logs/yachiyo-*-packages.json`, `logs/yachiyo-*-freeze.txt`, `logs/uwipl-*-pip-freeze.txt`, `logs/posetrack-all-pip-freeze.txt`.
- 빌드 실패 이력도 남겼다. 구형 CUDA 이미지의 만료된 NVIDIA apt 키, PoseTrack Bullseye 보안 패키지 404, ReID chardet 경고는 Docker 환경 안에서 해결 후 재검증했다.

이 최초 모델 환경 검증의 범위 밖/미검증: 실제 가중치 로딩과 모델 forward, SCOUT 데이터 연결, 전체 추론·학습·정확도/속도 비교, 선택적인 TensorRT/ONNX 및 별도 GPU 재순위 확장. 원본 실행 스크립트의 AI City 전용 가정과 발견한 코드 오류는 모델별 문서의 후속 작업에 기록했다. 후속 요청으로 구현한 어댑터와 2026-10-26의 54개 테스트·메타데이터 비교 결과는 [별도 어댑터 검증](adapter.md)에 기록한다. 그 검증도 실제 모델 추론 성공을 뜻하지 않는다.
