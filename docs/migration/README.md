# 새 서버 이동 파일 — 2026-11-08

**2026-11-09 UWIPL 보완:** 누락된 V4 코드·문서를 추가했다. 3번 MIG 컨테이너의 업데이트·환경 확인·공통 5초 추론 절차는 [3번 컨테이너 실행 안내](uwipl-container3-next.md)를 따른다. 이 추가는 새 서버에서의 실행 완료를 의미하지 않는다.

SCOUT 원본은 이미 전송됐다는 전제다. 실제 파일 존재를 확인하고 UWIPL V3까지 반영했다. 전송·압축·삭제는 실행하지 않았으며, 이동 경로 목록을 준비했다.

| 목록 파일 | 내용 | 대략 용량 |
|---|---|---:|
| 01-core.txt | 모델/어댑터 코드, 가중치, 설치 명세, 문서, 평가 코드 | 7.5 GiB |
| 02-latest-results.txt | OSC·PoseTrack·UWIPL 최신 인계 ZIP과 SHA-256 | 0.28 GiB |
| 03-resume-experiments.txt | 기존 5/30초 실험의 입력 캐시·설정·결과 | 12.9 GiB |

기존 연구를 이어가려면 세 목록을 모두 옮기는 구성을 권장한다. 합계 약 **20.6 GiB**이며 압축하지 않은 파일 크기·하드링크 보존 기준이다. 파일시스템 오버헤드나 하드링크 중복 복사 시 크기는 달라진다. 상세 값은 transfer-size-summary.json에 기록했다.

새 데이터로 신경망 추론부터 다시 실행한다면 01+02만 가져가도 되지만, 이전 실험과 캐시를 이용한 재현에는 03이 필요하다. 03은 대표 실험의 이어하기용이며 모든 과거 디버깅 실행의 의존성을 보장하지 않는다.

## 배정

- 1번 OSC: repos/yachiyo, docker/yachiyo/vendor, weights/yachiyo 및 osc-* 결과.
- 2번 PoseTrack: repos/posetrack, weights/posetrack, 시간 정렬된 aligned_input과 frame_mapping, 이전 full_input30.
- 3번 UWIPL: repos/uwipl, weights/uwipl 및 shared 가중치. 기존 uwipl-study, anchor-sweep, 최신 uwipl-v3 세 실험 디렉터리를 함께 보존.
- 4번 어댑터: adapters, scripts, tests, vendor/TrackEval 및 공통 프레임 매핑·평가 GT. 베이스라인 코드/가중치는 모델 선정 후 별도 추가.

코드 기준본·가중치·결과는 공유 경로에 한 번만 전송한다. 모델 소스 수정·editable 설치·확장 빌드는 컨테이너별 개인 작업 사본에서 한다. 라이브러리 환경은 공유하지 않는다. docker 폴더는 설치 명세와 OSC 외부 소스를 가져오기 위한 것으로, 새 컨테이너를 실행하기 위한 것이 아니다.

## 전송 경로와 예시

출발 루트: `/media/aimlab/HDD00/users/junseo-capstone/scout-mtmc/`

도착 루트 예시: `/workspace/shared/scout-mtmc/`. 목록은 모두 출발 루트 기준 상대 경로이며 도착지에서도 상대 구조를 유지한다. 이미 옮긴 `/workspace/shared/datasets/`의 SCOUT는 다시 전송하지 않는다.

SSH 주소·포트와 공유 경로는 실제 접근 가능한 값으로 바꿔야 한다. 다음은 명령 예시이며 자동 실행하지 않았다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
cat docs/migration/01-core.txt docs/migration/02-latest-results.txt docs/migration/03-resume-experiments.txt > /tmp/scout-transfer-files.txt

# 먼저 전송 예정 파일 확인. USER와 NEW_SERVER, 도착 경로를 실제 값으로 변경.
rsync -aHr --dry-run --info=stats2 \
  --files-from=/tmp/scout-transfer-files.txt \
  --exclude='__pycache__/' --exclude='build/' \
  --exclude='*.pyc' --exclude='*.so' --exclude='*.so.*' \
  --exclude='*.part' --exclude='*.lock' \
  ./ USER@NEW_SERVER:/workspace/shared/scout-mtmc/
```

전송 목록 확인 후 같은 명령에서 `--dry-run`을 빼면 실제 전송한다. `--delete`는 사용하지 않는다. `-H`는 가중치와 캐시의 하드링크 중복을 줄인다. `--files-from`으로 디렉터리를 지정할 때 하위 파일을 전송하도록 `-r`도 명시했다. 기존 목적지의 같은 경로 파일은 rsync가 갱신할 수 있으므로 전용 프로젝트 경로를 사용한다.

## 제외 대상과 이관 후 작업

데이터셋 압축·원본, Docker 이미지 tar, 기존 venv·Conda, 임시 다운로드 파일, 예전 컴파일 확장, 대용량 과거 실패 로그는 기본 이동 목록에 없다. 현재 서버에서 삭제하지 않는다.

현재 .so는 TITAN RTX 및 기존 Python ABI에 맞춘 파일일 수 있어 복사 목록에서 제외한다. 새 컨테이너에서 호환 라이브러리·확장을 설치/빌드해야 한다. 단계별 requirements 외에 Dockerfile의 소스 설치·패치 절차도 확인한다.

기존 runner의 docker run 호출과 /project, /opt/uwipl, /opt/posetrack, /data, /out 경로는 새 컨테이너 구조로 바꿔야 한다. 파일을 옮기는 것만으로 H200에서 실행 완료되는 것은 아니다. [환경 및 설치 안내](../mig-container-migration.md)를 따른다.
