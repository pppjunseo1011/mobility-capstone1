# SCOUT 데이터셋 다운로드

2026-10-25 추가 요청에 따라 공식 SCOUT 1.0 전체 압축 파일의 다운로드를 시작했다.
시작 시각은 **2026-10-25 18:58 KST**다. 아래 상태 파일의 `state`가 `complete`가 되기 전에는 다운로드·검증 완료가 아니다.

**완료 기록:** 2026-10-26 20:17:08 KST에 전체 371,017,893,820바이트 수신이 끝났고, 20:56:53 KST에 gzip CRC 검사 및 로컬 SHA-256 계산이 완료됐다. 현재 다운로드 상태는 `complete`다. SHA-256은 `9353dffb03cb0fc9e1fc838da19f451d8e951cbf738706e7a578daad3f141ff5`다. 이후 2026-10-27의 별도 사용자 요청으로 [압축 해제](scout-extraction.md)를 시작했으며, 다운로드 프로그램이 자동으로 압축 해제를 시작한 것은 아니다.

## 이용조건과 출처

- 공식 다운로드 및 약관: <https://scout.epfl.ch/download>
- 공식 파일: <https://scout.epfl.ch/datas/datasets/scout/1.0/scout.tar.gz>
- 등록과 약관 동의는 **사용자가 완료했다고 밝힌 내용**을 전제로 한다. 이 스크립트가 등록 여부를 독립적으로 확인하거나, 개인정보를 등록하거나, 동의를 대신 제출하지 않는다.
- 확인한 조건: 학술·연구 목적, 지정 논문 인용, 명시적 허락 없는 전체·일부 재배포 금지, 관련 법률·윤리 준수. 공식 약관이 우선하며 상업적 이용이나 공개 재배포 허락을 의미하지 않는다.
- 인용 논문: Martin Engilberge et al., *One Graph to Track Them All: Dynamic GNNs for Single- and Multi-View Tracking*, arXiv:2507.08494, 2025.
- 신규 데이터 폴더는 계정 소유자 접근 권한 `0700`, 파일은 기본 `0600`으로 만든다. 공유 계정을 사용하는 사람까지 구분하는 접근 통제는 아니다. Git과 Docker 빌드에서 `datasets/`는 기존 설정으로 제외된다.

## 파일과 공간

공식 HEAD 응답의 크기는 **371,017,893,820 바이트**(371.02 GB / 345.54 GiB)다. 시작 시 HDD 여유 공간은 약 936.46 GiB였다. 압축 해제 후 필요 공간은 아직 확인하지 않았고, 압축 해제는 자동 실행하지 않는다.

- 저장 위치: `datasets/scout/archives/`
- 수신 중 파일: `scout.tar.gz.part`
- 무결성 검증 통과 후 파일: `scout.tar.gz`
- 공식 응답 기록: `scout.tar.gz.meta.json` (URL, 크기, ETag, 수정 시각)
- 실시간 상태: `scout.tar.gz.status.json`
- 중복 실행 방지 잠금: `scout.tar.gz.lock`
- 완료 후 로컬 체크섬: `scout.tar.gz.sha256`
- 실행·진행 로그: `logs/scout-download.log`

파일 크기와 ETag가 일치할 때만 이어받는다. ETag는 `"01336e855ce99068538475070ce6ca62-5529"`이며, 일반적인 파일 MD5로 해석하면 안 된다. 완료 후 gzip 전체 스트림의 CRC/길이를 검사하고 SHA256을 동시에 계산한다. SHA256은 로컬 보관용 지문이며, 배포자가 공개한 체크섬과 대조한 결과가 아니다. gzip 검증은 tar 내부 디렉터리 구조나 SCOUT 주석의 정확성 검증과 별개다.

## 실행과 진행 확인

### 기존 컨테이너의 공유 저장소에 다운로드 (2026-11-04)

수정된 `scripts/download-scout.py`를 새 서버에 복사한 뒤, 프로젝트 루트에서 아래처럼 실행한다. 현재 터미널의 `pwd`가 아니라 `--download-dir`로 저장 위치를 지정한다.

```bash
python3 scripts/download-scout.py start --download-dir /workspace/shared/datasets --limit-rate 0
python3 scripts/download-scout.py status --download-dir /workspace/shared/datasets
tail -f /workspace/shared/datasets/scout-download.log
python3 scripts/download-scout.py stop --download-dir /workspace/shared/datasets
```

압축파일은 `/workspace/shared/datasets/scout.tar.gz`로 저장되며, 부분 파일·메타데이터·상태·잠금·로그도 같은 디렉터리에 저장한다. 재개할 때는 같은 `start` 명령을 사용한다. 모든 제어 명령에 동일한 `--download-dir`를 지정해야 한다. 옵션을 생략하면 기존 프로젝트 내부 경로를 사용한다. 기존에 다른 위치에서 받던 파일은 자동 이동하지 않는다. 공유 경로라도 신규 파일은 기존 정책대로 소유자 전용 권한으로 생성하므로, 다른 UID의 컨테이너에서 사용할 때는 관리자와 공통 그룹/ACL을 맞춰야 한다.

경로 옵션 및 백그라운드 자식 프로세스 전달을 로컬 모의 검사로 확인했다. 새 서버 접근이나 실제 다운로드는 이 변경에서 수행하지 않았다.

프로젝트 루트에서 실행한다. 모델 의존성 설치와 무관한 표준 Python 3 및 시스템 curl만 사용하며, GPU를 사용하지 않는다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc

# 상태: 진행률, 받은 바이트, 평균 속도, 예상 남은 초
python3 scripts/download-scout.py status

# 터미널에서 30초마다 상태 보기 / Ctrl+C로 표시만 종료
watch -n 30 python3 scripts/download-scout.py status

# 1분마다 기록되는 진행 로그 보기 / Ctrl+C로 로그 표시만 종료
tail -f logs/scout-download.log

# 이 다운로더만 중지하고 부분 파일은 보존
python3 scripts/download-scout.py stop

# 중단 이후 속도 제한 없이 이어받기 (백그라운드 실행)
python3 scripts/download-scout.py start --limit-rate 0
```

최초 2026-10-25 18:58 KST에는 최대 20 MiB/s로 시작했고, **같은 날 사용자 요청으로 속도 제한을 해제했다.** 현재 기본값은 `--limit-rate 0`이다. 값이 `0`이면 curl에 속도 제한 인자를 전달하지 않는다. 단일 연결과 낮은 CPU 우선순위는 유지한다.

실제 전송 속도는 서버와 네트워크 조건에 따라 달라진다. `eta_seconds`는 해당 전송 시도의 실측 평균에 기반하며, 이후 gzip 검사 시간은 포함하지 않는다. 초기 ETA는 변동이 크다. 전송 중 상태 파일은 약 5초마다, 로그는 약 1분마다 갱신한다. 무결성 검증 중에는 상태 파일만 주기적으로 갱신하며, 로그에는 검증 완료 또는 실패를 기록한다. 이 기록은 채팅 알림을 자동 발송하는 기능은 아니다.

디스크 여유 공간은 시작 시 남은 전송량 + 50 GiB를 요구하고, 전송 중에도 50 GiB 아래로 내려가면 중지한다. 일시적인 네트워크 오류는 최대 6회 시도하며 부분 파일을 이어받는다. HTTP 오류·쓰기 오류·이어받기 거부·인증서 오류 등은 자동 반복하지 않는다. `start` 명령은 중복 실행 시 기존 잠금 때문에 새 다운로드를 거부한다.

상태 구분:

- `preflight`: 공식 파일 정보와 공간 확인 중
- `downloading`: 전송 중
- `retry_wait`: 네트워크 재시도 대기
- `verifying`: 전송 완료, gzip CRC 및 로컬 SHA256 확인 중
- `complete`: 크기와 gzip 검증까지 통과
- `interrupted` / `failed`: 중단 또는 실패. 로그를 확인하고 재개 여부를 결정한다.

서버 재부팅 뒤에는 자동으로 다시 시작하지 않는다. 실행 중이던 터미널/채팅이 종료되어도 정상적인 백그라운드 작업은 계속된다. 데이터 연결, 어댑터 구현, 압축 해제, 가중치 다운로드, 추론·학습은 이 다운로드 작업에 포함하지 않는다.
