# mobility-capstone1 — 모빌리티캡스톤디자인1

# SCOUT MTMC 연구 코드 및 서버 이관

이 저장소는 SCOUT 어댑터, 세 모델 실행·분석·평가 코드, 환경 명세와 작은 비교 결과표를 보관합니다. **데이터셋·가중치·원본 모델 소스·실험 캐시·PDF/ZIP·로그는 Git 업로드에 포함하지 않습니다.** 기존 서버 원본은 별도로 보존합니다.

```bash
git clone https://github.com/pppjunseo1011/mobility-capstone1.git
cd mobility-capstone1
python3 scripts/restore-upstreams.py
python3 scripts/restore-upstreams.py --check
```

외부 소스 7개는 `upstream-sources.lock.json`에 기록된 공식 저장소와 커밋으로 복원합니다. 외부 코드의 라이선스는 각 원본 저장소를 따릅니다. 위 명령은 라이브러리 설치나 모델 실행을 수행하지 않습니다.

새 H200 MIG 서버는 **이미 배정된 컨테이너 안에 환경을 설치**합니다. Docker를 중첩 실행하지 않습니다. 1번 OSC, 2번 PoseTrack, 3번 UWIPL, 4번 어댑터 담당이며, 모델 환경과 확장 모듈은 각 컨테이너에서 따로 준비합니다. [설치·이관 안내](docs/mig-container-migration.md), [별도 전송 목록](docs/migration/README.md)을 참고하세요. 기존 실행 코드의 `/project`, `/opt`, `/out` 등 경로는 새 서버에 맞게 조정해야 합니다. 구형 CUDA 환경의 H200 호환성은 별도 검증 대상입니다.

작은 결과표는 [reports/metrics](reports/metrics)에 있습니다. 실험마다 입력·동기화·평가 프로토콜이 달라 숫자를 직접 모델 순위로 해석하면 안 됩니다. 원본 경로는 `provenance.json`에 기록했습니다. 아래 과거 작업 기록에 있는 `outputs/`, `logs/`, PDF/ZIP 링크는 별도 전송 파일을 가리키며 Git clone만으로 열리지 않습니다.

2026-11-08 기준 후속 실험은 [PoseTrack 문서](docs/posetrack.md), [UWIPL 문서](docs/uwipl.md), [OSC 문서](docs/osc-checklist.md)를 확인하세요. 아래 내용은 최초 구축 이후의 작업 기록으로, 당시의 ‘아직 미실행’은 현재 상태와 다를 수 있습니다.

---

# SCOUT 다중 카메라 추적 비교 — Docker 환경

2026-10-25, `/media/aimlab/HDD00/users/junseo-capstone/scout-mtmc`에 작성했다. 최초 작업은 세 모델의 실행 환경 구축과 의존성/확장 모듈/작은 CPU·CUDA 연산 검증이다. **환경 검증은 모델 추론 성공 또는 SCOUT 적용 성공을 의미하지 않는다.** 추가 요청으로 SCOUT 다운로드·압축 해제와 1차 데이터 어댑터를 준비했고, 2026-10-27 실제 표본 검증을 진행했다. 현재 테스트 범위는 사용자 지정 **sequence_1, cam_0~cam_7**이다. 소량 이미지의 실제 검출·포즈·ReID 추론은 2026-10-31 완료했다. 전체 이미지 검증과 카메라별/다중 카메라 추적·학습·평가는 후속 단계다. [실제 표본 검증 결과와 남은 문제](docs/scout-real-validation.md).

**구축 완료: 이미지 9개 빌드, CPU 검증 9/9, GPU 대상 CUDA 검증 8/8 성공.** 데이터·가중치 읽기 전용 연결과 별도 결과 저장도 실제 컨테이너에서 검증했다. [단계별 결과와 실행 로그](docs/validation.md), [정확한 이미지 ID와 환경 파일 해시](logs/environment-manifest.json).

**이전 어댑터 v0.3:** 연속 시간 구간 선택, OpenCV 기준 JPEG/MP4 픽셀 일치, YACHIYO 검출↔포즈 프레임 변환을 추가했다. 새 CPU 이미지 빌드와 회귀 검사 95개를 통과했다. sequence_1의 cam_0~cam_7, 원본 frame 277~476의 약 20초 구간에서 실제 어댑터 검사 31개와 모델 컨테이너 영상 입력 검사 24/24개를 통과했다. JPEG 1,600장, MOT GT 33,550행을 확인했다. 보정의 수치 진단도 수행했으며 실제 모델 추론과 native H 생성은 아직 하지 않았다. [v0.3 작업·검증·실행 안내](docs/scout-next-stage.md), [종합 기록](logs/adapter-v03-validation.json).

**이전 어댑터 v0.4 (2026-10-28):** 원본 픽셀의 왜곡을 보정한 뒤 실제 지면 메시와 교차시켜 XYZ를 계산하는 기능을 추가했다. 화면 격자 2,184점 중 1,043점에서 교차점을 찾았고 나머지는 좌표를 비워 두었다. 자동 검사 119개와 실제 CSV 입력 재실행을 통과했다. 수치적 일관성 검사이며 실제 사람 위치 정확도나 모델 추론 성공을 뜻하지 않는다. [사용법·결과·한계](docs/scout-ground-projection.md), [검증 기록](logs/adapter-v04-validation.json).

**현재 어댑터 v0.5 (2026-10-31 검증 완료):** YACHIYO SCT, UWIPL MOT16, PoseTrack 포즈·검출 파일을 원본 프레임과 지면 좌표 후보에 연결한다. 자동 검사 149개와 실제 SCOUT 기하 + 합성 모델 자료의 24건 연결 검사를 통과했다. 실제 모델 추론이나 추적 성능 검증은 아니다. [입력 계약·사용법·완료 범위](docs/scout-model-ground.md), [검증 기록](logs/adapter-v05-validation.json).

**세 모델 가중치 확보 (2026-10-31):** PoseTrack 3개, YACHIYO 3개, UWIPL 실영상용 7개 및 추가 YOLOv7 5개를 `weights/<모델>/`에서 사용할 수 있다. YACHIYO/UWIPL 공통 파일은 하드링크로 공유한다. 크기·SHA-256을 확인했고 ZIP 형식은 CRC 검사도 통과했다. UWIPL 합성 ReID 1개는 공식 링크 HTTP 404로 미확보이며 창고용 YOLOv7은 배포 파일명과 원본 실행 설정이 다르다. 실영상용 체크포인트 로딩·소량 검출/포즈/ReID 추론도 후속 검증에서 통과했다. [가중치 목록·출처·검증 기록](docs/model-weights.md).

**소량 실제 추론 완료 (2026-10-31):** 8개 카메라 × 2프레임 = 16장으로 세 모델의 검출·포즈·ReID 9단계를 통과했다. 모델별 검출 250행 중 141행을 선택해 포즈와 ReID 특징을 생성하고 행 대응을 확인했다. PoseTrack 실제 결과 141행의 기존 어댑터 연결도 통과했다. 카메라 내/간 추적 ID 생성과 성능 평가는 아직 수행하지 않았다. [범위·결과·재실행 안내](docs/scout-small-inference.md), [종합 검증 기록](logs/scout-small-inference-20261031.json).

**OSC 인원 제한 해제 재검증 (2026-11-01):** sequence_1/cam_0~cam_7/source frame 277~326의 400장을 프레임당 개수 제한 없이 다시 추론했다(`--max-people-per-frame 0`, objectness ≥0.3 유지). 후속 입력 5,569개, 임베딩 5,569×512, Local 할당 5,329행, 최종 Global 612행·9 ID. SCPT/MCPT 2회 결과 해시가 동일하고 CSV·평가 검사를 통과했다. 이전 최대 10개 조건 대비 Local recall 40.91→51.19%, multi-view IDF1 11.35→12.34%이나 ID 전환은 증가하고 MOTA는 하락했다. 공간 보정 OFF인 5초 파일럿으로 공식 SCOUT/HOTA 결과는 아니다. [상세 결과](docs/osc-checklist.md), [검증 근거](outputs/osc-pilot-uncapped-20261101/provenance/evidence.json), [PDF](docs/handoff/OSC_SCOUT_인계서_제한해제_20261101.pdf), [전달 ZIP](deliverables/OSC_SCOUT_handoff_uncapped_20261101.zip). 이전 `outputs/osc-pilot-20261101`과 로그는 비교 근거로 보존한다.

**OSC 원인 분석 및 조건별 비교 (2026-11-01):** 5초 기준에서 raw SCPT 2,029 ID → NMS 후 1,826 ID였고 warp 추가 분리는 없었다. Global 미할당 4,953행과 할당 후 Pose 삭제 4행을 구분했다. `short_track_th=10/20/30/40` 단일 조건 비교 및 기존 th=40의 약 30초(277~576) 확장을 완료했다. 5초 Local 결과는 네 설정에서 바이트 단위로 동일하며, 각 실험은 2회 반복해 최종 해시가 일치한다. 30초는 최대 timestamp 편차 2.051초와 불연속이 있어 동기화된 공식 World 평가로 해석하지 않는다. HOTA·DetA·AssA는 미산출이다. [분석 PDF](docs/handoff/OSC_SCOUT_원인분석_비교실험_20261101.pdf), [결과·코드 ZIP](deliverables/OSC_SCOUT_analysis_20261101.zip), [설정별 비교 CSV](outputs/osc-analysis-20261101/experiment_comparison.csv), [분석 요약](outputs/osc-analysis-20261101/summary.json).

## 서버

Ubuntu 20.04.6 x86_64, i9-13900K (32 논리 CPU), RAM 약 63GiB, **TITAN RTX 24GiB × 2**, NVIDIA 580.126.09. Docker 28.1.1과 NVIDIA Container Toolkit 1.17.6이 설치되어 있고 GPU 컨테이너 연결을 확인했다. 최초 여유 공간은 작업 HDD 약 938GiB, Docker SSD 약 338GiB. 기존 컨테이너 48개(실행 33개), 이미지 24개를 보존한다. [서버 상세 기록](docs/server.md).

사용자 확인에 따른 배정 예시는 YACHIYO·UWIPL → GPU 0, PoseTrack → GPU 1이다. GPU는 실행할 때 지정하며, 미지정 시 CPU로 실행한다. 두 모델의 GPU 0 동시 전체 추론은 메모리 검증 전까지 실행하지 않는다. 후속 요청으로 CPU 어댑터 환경을 추가했으며 어댑터에는 GPU를 노출하지 않는다.

## 소스와 환경 구성

| 모델 | 브랜치 / 커밋 | 단계별 이미지 태그 |
|---|---|---|
| [YACHIYO](https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS) | main / `f881fe0edf60fe697efb2bf4c935b80a80861230` | `scout-mtmc/yachiyo-{detection,reid,pose,tracking}:f881fe0` |
| [UWIPL](https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI) | main / `8e23a69de57d2547b774399ae45792f83a2d17ef` | `scout-mtmc/uwipl-{botsort,mmyolo,reid,pose}:8e23a69` |
| [PoseTrack](https://github.com/ZhenyuX1E/PoseTrack/tree/master) | master / `ba727e6bc6e45e44cd6e006b73e00864892173c2` | `scout-mtmc/posetrack-all:ba727e6` |

| 환경 | 실제 Python | PyTorch / torchvision | CUDA | 검증 |
|---|---|---|---|---|
| YACHIYO 검출·ReID·포즈 | 3.8.12 | 1.11.0 / 0.12.0 | 11.3 | CPU 및 GPU 0 성공 |
| YACHIYO 추적 | 3.10.14 | 사용 안 함 | 사용 안 함 | CPU 성공 |
| UWIPL 4단계 | 3.8.12 | 1.11.0 / 0.12.0 | 11.3 | CPU 및 GPU 0 성공 |
| PoseTrack 전체 단계 | 3.9.23 | 2.0.1+cu118 / 0.15.2+cu118 | 11.8 | CPU 및 GPU 1 성공 |

YACHIYO 포즈는 MMPose 0.29.0 / mmcv-full 1.7.0 / MMDetection 2.28.2, UWIPL 포즈는 MMPose 0.29.0 / mmcv-full 1.7.1 / MMDetection 2.28.2를 사용한다. UWIPL MMYOLO는 MMCV 2.0.0 / MMDetection 3.0.0 / MMEngine 0.7.4, PoseTrack은 MMCV 2.0.1 / MMDetection 3.1.0 / MMPose 1.3.1 / MMEngine 0.10.4다. YACHIYO CPU 추적은 NumPy 1.26.4 / SciPy 1.13.0 / scikit-learn 1.4.2이며 다른 환경은 구형 NumPy API 호환을 위해 1.23.5로 고정했다.

버전 선택, 공식 설정과의 차이, 단계별 코드 검토와 필요한 가중치/입력은 [YACHIYO](docs/yachiyo.md), [UWIPL](docs/uwipl.md), [PoseTrack](docs/posetrack.md)에 기록했다. 구형 CUDA 이미지의 apt 키 문제와 PoseTrack 기반 OS의 패키지 다운로드 실패는 이미지 내부에서 해결했다. 현재 환경 검증의 실패 항목은 없다.

YACHIYO는 검출/특징/포즈와 CPU 추적을 나누고, UWIPL은 MMCV 1.x 포즈와 MMCV 2.x 검출의 충돌 때문에 환경을 분리한다. PoseTrack은 표준 PyTorch 검출·포즈·ReID·추적을 하나의 환경으로 구성하며 선택 사항인 TensorRT 가속은 설치하지 않는다.

## 빌드

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 scripts/container.py list
bash scripts/build.sh yachiyo all
bash scripts/build.sh uwipl all
bash scripts/build.sh posetrack all
# 또는 전체를 순서대로 빌드
bash scripts/build.sh all
```

각 단계만 빌드하려면 `bash scripts/build.sh uwipl pose`처럼 지정한다. 빌드는 GPU 연산 없이 수행되며 컴파일 병렬도는 2로 제한한다. `--dry-run`은 실제 Docker 명령만 출력한다. 빌드 스크립트는 소스 커밋 일치와 변경/추가 파일 유무를 확인하며, 실패한 빌드를 성공 처리하지 않는다. Docker 접근이 가능한 호스트 셸에서 실행한다. 모델 Python 의존성은 모두 컨테이너 안에 설치한다.

다른 경로/서버에서 소스를 다시 준비할 때는 `python3 scripts/fetch-sources.py`로 잠긴 커밋을 가져온다. YACHIYO 빌드는 `docker/yachiyo/fetch-vendor.sh`로 외부 의존성의 잠긴 커밋도 준비/확인한다. 기존 체크아웃을 덮어쓰지 않는다. 기반 이미지도 수 GB 규모이며, 모델 가중치와 별개다.

## 최소 동작 검증

```bash
bash scripts/smoke.sh yachiyo detection --cpu
bash scripts/smoke.sh yachiyo detection --gpu 0
bash scripts/smoke.sh yachiyo reid --gpu 0
bash scripts/smoke.sh yachiyo pose --gpu 0
bash scripts/smoke.sh yachiyo tracking --cpu
bash scripts/smoke.sh uwipl botsort --gpu 0
bash scripts/smoke.sh uwipl mmyolo --gpu 0
bash scripts/smoke.sh uwipl reid --gpu 0
bash scripts/smoke.sh uwipl pose --gpu 0
bash scripts/smoke.sh posetrack all --cpu
bash scripts/smoke.sh posetrack all --gpu 1
```

모든 GPU 단계는 `--cpu`로 CPU 검증도 할 수 있다. 각 GPU smoke는 한 장만 지정한다. YACHIYO tracking은 원래 CPU 알고리즘이므로 GPU 검증 대상이 아니다. 검증은 가중치 없이 import, 재빌드한 C++/Cython 확장, NMS 등의 작은 연산으로 제한한다. 컨테이너에서는 배정한 호스트 GPU가 `cuda:0`으로 보인다.

`logs/`에 빌드 및 검증 명령/결과를 남긴다. 새 smoke 컨테이너는 종료 후 자동 제거된다. 실행 기본값은 네트워크 차단, CPU 2개, 메모리 8GiB, 현재 사용자 UID/GID이다. 향후 실제 추론 시 `--cpus`, `--memory`를 조정할 수 있다. 네트워크가 필요한 명시적 작업에는 `--network bridge`를 지정한다.

`python3 scripts/check-mounts.py`는 임시 소형 샘플로 읽기 전용 마운트와 호스트 결과 저장/소유권을 검사한다. SCOUT 연결 검증이 아니다. `python3 scripts/capture-manifest.py`는 현재 9개 이미지 ID·크기와 환경 파일 해시를 `logs/environment-manifest.json`에 기록한다. 소스 커밋·기반 이미지 digest·주요 Python 패키지는 고정하고 실제 전체 설치 목록은 모델별 pip-freeze 로그에 남긴다. apt 및 일부 전이 의존성까지 완전한 오프라인 잠금을 만든 것은 아니므로, 향후 재빌드 결과는 이미지 ID와 설치 목록을 다시 비교해야 한다.

## SCOUT 다운로드 (추가 요청)

2026-10-25 **18:58 KST**, 사용자 등록·약관 동의 완료 진술 및 다운로드 승인에 따라 공식 SCOUT 1.0 전체 압축 파일(371.02 GB / 345.54 GiB)의 다운로드를 시작했다. 저장 위치는 `datasets/scout/archives/`이며, 완료 전에는 `.part`로 보관한다. 압축 해제와 모델 실행은 자동으로 시작하지 않는다.

최초에는 20 MiB/s로 제한했으나, **2026-10-25 사용자 요청으로 속도 제한을 해제했다.** 현재 기본값은 `--limit-rate 0`이며 다운로더의 속도 제한 없이 이어받는다. 단일 연결과 낮은 CPU 우선순위는 유지한다.

```bash
python3 scripts/download-scout.py status                # 현재 진행률·속도·남은 시간
tail -f logs/scout-download.log                        # 1분 간격 진행 기록
python3 scripts/download-scout.py stop                 # 부분 파일을 보존하며 중지
python3 scripts/download-scout.py start --limit-rate 0   # 속도 제한 없이 이어받기
```

상태 파일이 `complete`여야 크기 및 gzip 무결성 검사가 끝난 것이다. 전송 중 상태는 약 5초마다 갱신한다. [다운로드·이용조건·검증 안내](docs/scout-download.md), [실시간 상태 파일](datasets/scout/archives/scout.tar.gz.status.json), [실행 로그](logs/scout-download.log).

**다운로드 완료:** 2026-10-26 20:17:08 KST에 371,017,893,820바이트 수신을 완료했고, 20:56:53에 크기·gzip CRC 검증을 통과했다. SHA-256은 `9353dffb03cb0fc9e1fc838da19f451d8e951cbf738706e7a578daad3f141ff5`이며 배포자 체크섬과 비교한 값은 아니다.

### SCOUT 압축 해제 — 2026-10-27 추가 요청

2026-10-27 **14:31:10 KST에 압축 해제 완료**했다(시작 12:07:03). 1,124,853개 파일, 374,109,919,942바이트를 추출했고 압축 파일 SHA-256 재대조도 통과했다. 원본 압축 파일은 보존했으며 실제 데이터 루트는 **`datasets/scout/extracted/scout`**이다.

```bash
python3 scripts/extract-scout.py status
tail -f logs/scout-extract.log
```

새 파일만 생성하며 50GiB 여유 공간을 유지한다. 모델 가중치·추론·학습·어댑터 전체 변환은 자동 시작하지 않는다. [추출 방식·상태·실패 시 안내](docs/scout-extraction.md), [합성 안전성 검사 8개](logs/scout-extract-script-tests.log). 다음 순서는 실제 구조·메타데이터 확인 → 작은 실제 구간의 어댑터 검증 → 시간/보정/가중치/실행기 연결 → 소규모 모델 추론이다.

## SCOUT 1차 어댑터

공식 로더와 실제 보정·타임스탬프/COCO 앞부분을 대조해 모델별 입력 배치, 카메라·frame 역매핑, 추적 결과 공통 CSV 변환을 구현했다. **입력 폴더 생성이 전체 모델 실행 준비 완료를 의미하지 않는다.** 시간 동기화, 왜곡·지면 보정 및 AI City 전용 코드 수정은 생성되는 보고서의 미해결 항목으로 남는다.

```bash
python3 scripts/adapter.py build
python3 scripts/adapter.py test
python3 scripts/adapter.py inspect --data datasets/scout/reference-metadata -- --sequence 1
# 실제 압축 해제 후 작은 clip 변환 예시:
python3 scripts/adapter.py export --data /실제/SCOUT/경로 \
  --output outputs/scout-yachiyo-preview \
  -- --model yachiyo --sequence 1 --cameras cam_3 cam_4 --max-frames 32
```

최초 v0.1은 합성 테스트 42개를 통과했다. **2026-10-26 v0.2에서는 요구사항 비교·적용 가능성 보고서 생성(`assess`)을 추가하고 총 54개 테스트를 통과했다.** 2026-10-27 현재 CPU 이미지는 `scout-mtmc/adapter:0.3`이며 v0.1/v0.2를 보존한다. v0.3 검사 95개에는 기존 검사도 포함된다. 실제 SCOUT 전체 이미지 변환·모델 추론은 미검증이다. [포맷·실행 안내](docs/adapter.md), [최초 기록](logs/adapter-validation.json), [v0.2 검증](logs/adapter-v02-validation.json), [포맷 검토 기준](adapters/scout-format.lock.json).

### 실제 데이터 검증 — 2026-10-27

사용자 요청으로 **sequence_1의 cam_0~cam_7, 각 0~7번 프레임(원본 JPEG 64장)**을 현재 표본으로 선정했다. 세 모델의 입력 배치·프레임/카메라/시간 대응과 영상 24개를 검증했다. MOT 정답은 1,401행으로 변환했다. **COCO는 선택 이미지 인덱스 48개가 누락되어 변환을 거부했다.** 주석 본문에는 대응 행이 있으므로 이미지 목록의 불완전성과 GT 행 부재를 구분한다. 모든 검사가 성공한 것으로 처리하지 않는다.

공식 논문의 수동 정리 범위와 공식 코드의 `scoutval` 카메라 목록이 이 범위에 대응한다. 1FPS 수동 정리 후 10FPS 보간된 주석이며, 모든 프레임의 정확성을 개별 검증한 것은 아니다. 이번 작업은 가중치 없는 CPU 입력 검증이다. 실행 명령·컨테이너 디코더 결과·COCO 문제·다음 단계는 [실제 표본 검증 문서](docs/scout-real-validation.md)를 따른다. 이전 두 sequence의 소형 점검 결과는 형식 진단 기록으로 보존하며 향후 비교 평가 범위에 포함하지 않는다.

### 제공 정보와 모델 요구사항 비교 — 2026-10-26

[종합 비교·추가 정보 생성 및 적용 검토](docs/scout-model-compatibility.md)에 영상/JPEG, K/R/T/dist, timestamp, GT, 3D mesh와 세 모델의 입력 계약을 비교했다. 폴더·MP4·번호 대응표는 변환 가능하며, 지면 H·왜곡 처리·시간 정렬은 조건 검증과 추가 구현이 필요하다. 검출·포즈·ReID는 SCOUT의 누락 정보를 채우는 것이 아니라 모델이 추론해야 할 중간 산출물이다.

```bash
python3 scripts/adapter.py assess \
  --data datasets/scout/reference-metadata \
  --output outputs/scout-compatibility-new \
  -- --sequence 1 --sync-tolerance 0.05
```

새 출력의 `comparison.md`와 `compatibility.json`에서 실제 가용 카메라, timestamp 누락, 같은 frame의 시각 차이, 생성 가능성과 미해결 항목을 확인한다. 이번 실제 결과는 `outputs/scout-compatibility-sequence{1,2}-20261026/`에 있다. 같은 frame에서 비교한 카메라 간 최대 시각 차이는 각각 약 2.950초/2.507초였으며, sequence 2의 cam_1/3/4 timestamp 부재는 숨기지 않고 표시했다. 이는 메타데이터 진단 결과이며 재동기화를 수행한 것은 아니다.

## 향후 데이터·결과 연결

SCOUT가 도착한 뒤 실제 호스트 경로를 설정한다. 다음 예시는 셸만 열며 모델을 실행하지 않는다.

```bash
SCOUT_DIR=/실제/SCOUT/경로
RESULT_DIR=/실제/결과/경로
WEIGHT_DIR=/실제/가중치/경로
bash scripts/run.sh posetrack all --gpu 1 \
  --data "$SCOUT_DIR" --output "$RESULT_DIR" --weights "$WEIGHT_DIR" \
  --interactive -- bash
```

| 호스트 인자 | 컨테이너 경로 | 권한 / 환경변수 |
|---|---|---|
| `--data` | `/data/scout` | 읽기 전용 / `SCOUT_DATA_DIR` |
| `--weights` | `/weights` | 읽기 전용 / `SCOUT_WEIGHTS_DIR` |
| `--output` | `/outputs` | 쓰기 가능 / `SCOUT_OUTPUT_DIR` |

데이터/가중치 인자는 선택 사항이다. 결과 경로 미지정 시 `outputs/<model>/<stage>`를 사용한다. 입력과 결과 폴더가 같거나 서로 포함되면 거부한다. 원본 모델은 이 환경변수에 자동 대응하지 않는다. 경로 인자, 디렉터리 구조, 카메라 ID/시간 동기화/보정 정보, 출력 규격을 연결하는 작업은 SCOUT 어댑터 단계에서 수행해야 한다. 원본 스크립트의 AI City 장면/GPU 하드코딩을 그대로 실행하면 안 된다.

컨테이너 코드 경로는 `/opt/yachiyo`, `/opt/uwipl`, `/opt/posetrack`이며 각 단계의 하위 프로젝트가 Dockerfile의 작업 디렉터리다. 원본 저장소의 Conda 환경 이름을 활성화하는 셸 스크립트는 컨테이너 단계별 Python 실행으로 바꿔 연결해야 한다.

## 파일 구성

- `repos/`: 커밋을 기록한 공식 소스 체크아웃. 원본 수정 없이 보존.
- `sources.lock.json`, `environments.json`: 저장소 버전과 단계/이미지 목록.
- `docker/<model>/`: Dockerfile, 의존성 설정, 가중치 없는 smoke 코드와 필요한 빌드 보조 파일.
- `scripts/`: 소스 준비, 서버 점검, 빌드/실행/검증 공통 명령.
- `docs/`: 서버 기록과 모델별 분석/호환성 변경/검증 결과/다음 단계 입력 목록.
- `logs/`: 실제 명령의 결과 및 실패/수정 이력.
- `outputs/`: 호스트 사용자 소유의 결과 저장 위치.
- `datasets/scout/archives/`: SCOUT 다운로드 파일과 상태/메타데이터. Git 및 Docker 빌드에서 제외.
- `adapters/scout_adapter/`, `docker/adapter/`, `scripts/adapter.py`: 1차 어댑터와 CPU 실행 환경.
- `tests/`: 합성 데이터 기반 포맷·프레임·ID·입출력 검증. 실제 모델 성능 평가가 아님.

## 다음 단계

소량 검출·포즈·ReID 추론은 별도 실행기로 검증했다. 원본 전체 추적 파이프라인은 아직 미검증이다. UWIPL 원본 ReID 실행기의 argparse 재호출·경로/인자 오류, PoseTrack README의 스크립트 파일명 불일치와 여러 모델의 AI City 전용 카메라/장면/보정 가정도 남아 있다. 환경 설치로 해결되는 문제가 아니므로 모델별 문서에 위치와 영향을 기록했다. 체크포인트를 사용하는 전체 실행 전에 후속 단계에서 수정해야 한다. TensorRT/ONNX export 및 선택적인 별도 GPU 재순위 확장은 이번 구축 범위에 포함하지 않았다.

1. 검증한 추론 경로를 짧은 연속 구간으로 확장하고 카메라별 추적 ID 생성에 연결한다.
2. SCOUT 시간 정렬·왜곡 처리·지면/거리 단위와 모델별 공간 규칙을 검토한다. AI City의 scene/camera/ROI/보정/거리 임계값을 그대로 재사용하지 않는다.
3. 카메라 간 ID 연결과 공통 추적 결과 변환을 확인한 뒤 HOTA/IDF1 평가를 계획한다.
4. UWIPL 합성 영상 경로를 사용할 경우 미확보 ReID와 창고용 YOLOv7 파일명 불일치를 별도로 해결하고 검증한다.
