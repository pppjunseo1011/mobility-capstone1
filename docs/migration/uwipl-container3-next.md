# 3번 UWIPL 컨테이너: 누락 코드 반영 후 실행 순서

2026-11-09. 기존 TITAN 서버에서 UWIPL V4/V4 설정 조합 소스를 업로드용 저장소에 보완했다. **새 H200 MIG 컨테이너의 설치·추론은 아직 검증하지 않았다.** 기존 결과를 다시 실행한 것으로 보고하지 않는다.

## 1. 코드 받기 및 확인

3번 컨테이너에서 실제 클론 폴더로 이동한 뒤 실행한다. 이 저장소의 루트에 `scripts/`, `docs/`가 있으며 `scout-mtmc/`를 한 단계 더 붙이지 않는다.

```bash
git status --short
git pull --ff-only origin main
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('docs/migration/uwipl-v4-source-manifest.json').read_text())
bad = []
for name, expected in m['files'].items():
    p = Path(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        bad.append(name)
if bad:
    raise SystemExit('Missing or locally changed; inspect without overwriting: ' + ', '.join(bad))
print('PASS: UWIPL source snapshot verified')
PY
```

로컬 수정 때문에 pull이 실패하면 reset/강제 덮어쓰기 대신 diff를 확인하고 보존한다. 해시 불일치도 로컬 변경 여부를 먼저 확인한다.

이번 추가: V4 관련 Python 13개, V4 문서 2개, 기존 UWIPL 안내 갱신. 모델·가중치·데이터·신경망 캐시는 포함하지 않는다. `run-uwipl-v4.py`, `run-uwipl-v4-config.py`, `finish-uwipl-v4.py`는 **이전 호스트의 Docker 실행기**다. MIG 컨테이너 내부에서 실행하지 않는다. `uwipl-v4-config-study.py`는 기존 추론 캐시 기반 설정 재실험이며 새 영상의 전체 추론 진입점이 아니다.

## 2. 배정된 컨테이너의 환경·경로 연결

컨테이너 안에서 다음을 확인하고 실제 경로를 기록한다.

```bash
pwd
id
cat /etc/os-release
nvidia-smi
python3 --version
df -h . /workspace/shared/datasets
```

전체 호스트 자원을 쓸 수 있다고 가정하지 않는다. CPU affinity/cgroup 메모리 제한과 노출된 MIG 장치를 확인한다. 다른 장치·컨테이너로 작업을 분산하지 않는다. 새 Docker 실행, 호스트 드라이버·MIG 변경은 필요하지 않다.

| 경로 항목 | 확인할 내용 |
|---|---|
| 프로젝트 루트 | 현재 Git 클론 위치 |
| UWIPL 소스 | `repos/uwipl`, 커밋 `8e23a69de57d2547b774399ae45792f83a2d17ef`. 빌드는 컨테이너 개인 사본에서 수행 |
| 데이터 루트 | `/workspace/shared/datasets` 아래 실제 `images`, `calibrations`, `timestamps`, `annotations`가 있는 SCOUT 루트 |
| 가중치 | `weights/uwipl` 또는 전달된 별도 경로. YOLOX + OSNet 5종, 포즈 사용 시 HRNet-W48 |
| 공통 5초 입력 | 4번 어댑터가 확정한 영상 프레임, manifest, frame mapping, 유효 슬롯 목록. 없으면 임의로 공통 구간을 만들지 않고 요청 |
| 결과 루트 | 공유 결과 폴더 안의 UWIPL 전용 새 run 디렉터리. 기존 결과를 덮어쓰지 않음 |

설치 근거는 `docker/uwipl/`의 Dockerfile과 requirements, `docs/mig-container-migration.md`다. Dockerfile의 설치 작업을 컨테이너 안의 전용 Python 환경으로 옮긴다. 기존 CUDA 11.3/Torch 1.11 조합의 H200 호환성을 가정하지 말고 실제 설치 상태·CUDA 연산·확장 로딩 결과를 확인한 뒤 필요한 변경을 기록한다. 단계별 환경은 같은 컨테이너의 서로 다른 venv/Conda로 분리할 수 있다. 공유 폴더의 다른 담당자 환경은 수정하지 않는다.

기존 UWIPL 실제 포즈 경로는 MMPose 0.29/HRNet-W48이다. HigherHRNet 또는 MMPose 1.x와 동일하게 취급하지 않는다. 기본 STCRA OFF 추적 검증에서는 포즈 후처리를 필수로 추가하지 않는다.

기존 소스의 고정 경로는 다음과 같다. 명시적인 인자/환경변수를 받는 **새 내부 실행기**에서 연결하되, 원본 파일과 이전 결과는 보존한다. 다른 서비스가 쓰는 루트 경로에 무조건 심볼릭 링크를 만들지 않는다.

| 기존 경로 | 새 실행기에서 지정할 대상 |
|---|---|
| `/project` | 프로젝트 루트 |
| `/opt/uwipl` | 개인 UWIPL 소스 사본 |
| `/vendor/TrackEval` | 복원한 TrackEval 소스 |
| `/data` | SCOUT 원본 루트 (읽기 전용 사용) |
| `/sample/input` | 공통 5초 입력 |
| `/sample/uwipl/detection` | 해당 run의 검출 결과 |
| `/weights` | UWIPL 가중치 |
| `/outputs`, `/out` | 단계별 출력 / 실험 출력. 서로 의미가 다르므로 구분 |

### 소스와 캐시 의존성 구분

```text
uwipl-v4-config-study.py
  -> uwipl-v4.py -> uwipl-v3.py -> evaluate-uwipl-study.py
                                         -> vendor/TrackEval
uwipl-v4-detect.py -> UWIPL BoT-SORT/YOLOX + checkpoint
scout-sample-stage.py --model uwipl --stage reid -> deep-person-reid + OSNet
```

위 의존 Python 파일은 저장소에 있다. 원본 저장소·TrackEval은 `upstream-sources.lock.json`의 고정 커밋으로 복원한다. `restore-upstreams.py`는 세 모델 전체를 대상으로 하므로 UWIPL만 필요한 경우 파일을 읽고 해당 repo/TrackEval 항목만 복원한다. 수정된 기존 checkout을 초기화하지 않는다.

**옛 결과 재현에 필요한 별도 캐시**: `outputs/uwipl-study-20261101/`, `outputs/uwipl-anchor-sweep-20261102/`, `outputs/uwipl-v3-20261107/`, `outputs/uwipl-v4-20261108/`. 기존 개발 평가 GT는 `outputs/osc-analysis-20261101/s30_th40/gt_selected.json`도 참조한다. STCRA에는 포즈·보정값과 `stcra_homographies.json` 등이 추가로 필요하다. 이 파일들은 git pull로 생기지 않는다. 문서의 과거 PDF/ZIP/outputs 링크도 기존 서버 경로이며 이번 업로드에 포함되지 않는다.

기존 V3는 실행 중 생성한 `outputs/uwipl-v3-20261107/run_stcra_scout.py`를 참조한다. 작은 소스 변경을 보존하기 위해 `patches/uwipl/stcra-eight-cameras.patch`를 추가했다. 고정 커밋의 `STCRA/run_stcra.py` **개인 사본**에 적용하면 같은 코드를 복원할 수 있다. manifest의 원본/적용 후 SHA-256과 대조한다. 이 패치는 카메라 가중치를 c001~c008 모두 1로 설정하는 변경일 뿐, SCOUT 보정값을 생성하지 않는다.

## 3. 공통 5초 입력으로 실제 추론 검증

V4 baseline: area > 1000, anchor distance 80, anchor score > 0.95, anchor 시점 11, 다수결 15개 관측, 검출 입력 800×1440, NMS 0.7, downstream objectness >= 0.3, 인원 상한 없음, BoT-SORT new_track_thresh=0.7/high_thresh=0.4, FPS=10, STCRA OFF. 이전 V3 코드의 기본값 vote=5/high=0.6을 그대로 사용하지 않는다.

1. 4번이 확정한 sequence_1/cam_0~7의 공통 5초 입력·frame map을 읽는다. 원본 frame277~326은 과거 디버깅 구간이며 새 공통 구간과 같다고 가정하지 않는다. 실제 공통 인덱스 50개, 카메라별 유효/결측 슬롯 및 이미지 좌표계를 확인한다.
2. 각 단계 환경에서 `pip check`, import, CUDA 작은 연산, torchvision NMS와 실제 필요한 확장을 검사한다. 1~2프레임에서 가중치 로드·출력 shape·NaN을 먼저 확인한다. 실패하면 실패 단계와 로그를 남긴다.
3. 신경망을 새로 실행한다면 YOLOX 검출 -> 동일 BBox에서 OSNet 5종 -> BoT-SORT -> 앵커/다수결을 순서대로 연결한다. STCRA OFF이므로 포즈/STCRA는 이 baseline 성공 이후 별도로 확인할 수 있다. 5초 전용 runner를 만들어 30초 하드코딩·기존 캐시 자동 참조·기존 실험 결과와의 강제 동등성 검사에 들어가지 않게 한다. 알고리즘 조건 자체는 위 baseline을 유지한다.
4. 기존 `uwipl-v3.py`의 `sct`/`predict`를 사용할 경우 경로·고정 프레임 수·legacy cache fast path를 확인한다. V4 `tracking()`은 300프레임과 V3 캐시를 가정하므로 공통 5초 입력에 그대로 호출하지 않는다. GT는 추론에 전달하지 않는다.
5. 추론 결과를 공통 CSV로 변환한다. 각 행에 sequence_id, camera_id(cam_0~7), source_frame_id, common_time_index, global_id, x/y/width/height(원본 왜곡 영상 픽셀)를 보존한다. confidence/local_id가 있으면 추가한다. 4번의 확정 schema가 있으면 그것을 우선한다. `model_frame`을 시간 매핑 없이 `common_time_index`로 이름만 바꾸지 않는다.
6. run 디렉터리에 예측 CSV, 설정, 입력/frame map 해시, 코드 커밋, 가중치 해시, 환경 버전, 단계별 로그·소요시간, 출력 검증 결과를 저장한다. 데이터가 유효하지만 검출이 없는 프레임도 공통 슬롯 목록에서 제외하지 않는다.

완료 판정: 같은 컨테이너에서 새로운 5초 입력의 neural inference와 추적까지 성공하고, 원본 프레임·BBox·Global ID 대응 검사가 통과해야 한다. 캐시 추적 성공은 별도로 표시한다. GPU 초기화 성공, 소스 다운로드 성공, 과거 CSV 복사만으로 추론 완료라고 보고하지 않는다. 이 단계 다음에 4번에서 동일 GT/평가 규약으로 평가한다.

## 3번 컨테이너 Codex에 전달할 지시

> `git pull --ff-only origin main`으로 누락된 UWIPL V4 파일을 받은 뒤 `docs/migration/uwipl-container3-next.md` 순서대로 진행해 주세요. 기존 모델/로컬 변경을 보존하고, 먼저 source manifest와 컨테이너 자원·경로·가중치·공통 5초 입력을 확인하세요. 새 Docker를 만들거나 Docker-in-Docker를 사용하지 마세요. 기존 Docker 실행기를 직접 실행하지 말고, 이 컨테이너의 단계별 Python 환경과 실제 경로를 사용하는 5초 전용 실행기를 구성하세요. V4 baseline을 명시적으로 적용하고 YOLOX → OSNet → BoT-SORT → 앵커/다수결 → 공통 CSV까지 실제 실행하세요. STCRA는 OFF로 시작합니다. GT는 추론에 넣지 마세요. 필요한 입력/가중치가 없으면 누락 경로를 알려주고 가능한 환경 검증을 계속하세요. 완료 시 실행 명령, 환경 변경, 결과 CSV 위치, frame map, 입력·가중치·코드 해시, 검증 로그를 보고하고 새 추론과 기존 캐시 재사용을 구분하세요.
