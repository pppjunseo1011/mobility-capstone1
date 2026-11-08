# 서버 확인 기록

확인일: 2026-10-25 (Asia/Seoul). 원본은 `logs/server-before.txt`, 컨테이너 GPU 전달 결과는 `logs/nvidia-container-runtime.txt`.

| 항목 | 확인 결과 |
|---|---|
| 호스트 | AIMLab-DL3 |
| OS / 커널 / 아키텍처 | Ubuntu 20.04.6 LTS / 5.15.0-139-generic / x86_64 |
| CPU | Intel Core i9-13900K, 논리 CPU 32개 |
| RAM / swap | RAM 약 62.6GiB, swap 2GiB |
| GPU 0 | NVIDIA TITAN RTX, 24,576MiB, UUID `GPU-222f9d37-9542-4268-4e1c-29addae85aad` |
| GPU 1 | NVIDIA TITAN RTX, 24,576MiB, UUID `GPU-20a83b59-eff7-678f-212d-52e75eca9935` |
| NVIDIA 드라이버 | 580.126.09 |
| Docker / Buildx / Compose | 28.1.1 / 0.23.0 / 2.35.1 |
| NVIDIA Container Toolkit | nvidia-ctk / nvidia-container-cli 1.17.6 |
| Docker 런타임 | runc, nvidia 등록; 기본 nvidia |
| Docker 저장소 | `/var/lib/docker`, overlay2, cgroup v1 |
| 최초 디스크 여유 | 작업 HDD 약 938GiB, Docker가 사용하는 루트 SSD 약 338GiB |
| 기존 컨테이너 / 이미지 | 48개 중 33개 실행 / 이미지 24개 |
| 계정 | aimlab, UID/GID 1000:1000, 호스트 docker 그룹 소속 |

최초 샌드박스 점검의 `nvidia-smi` 실패와 Docker 소켓 `operation not permitted`는 호스트 장애가 아니었다. 승인된 호스트 읽기 전용 재점검에서 두 기능 모두 정상 응답했다. 새 임시 `ubuntu:22.04` 컨테이너에서 두 GPU 정보 조회도 성공했다. Toolkit 설치/런타임 등록 확인, 컨테이너 GPU 전달 확인, 모델 환경 CUDA 연산 확인은 서로 다른 검증이며 마지막 항목은 모델별 로그에 기록한다.

현재 작업 폴더는 `/media/aimlab/HDD00/users/junseo-capstone/scout-mtmc`이다. 최초 사용자 폴더에는 보호된 `.git`, `.agents`, `.codex`, `.aws` 항목 외 프로젝트 파일이 없었다. 실제 작성 범위는 사용자 작업 폴더와 필요 시 `/tmp`로 한정했다. Docker 이미지/빌드 캐시는 Docker 서비스의 저장소에 생성된다. 기존 프로젝트, 다른 사용자의 컨테이너/이미지/프로세스는 변경하지 않았다.

최초 GPU 사용량은 GPU 0 약 82MiB, GPU 1 약 6MiB였고 Xorg/gnome-shell 그래픽 사용만 확인됐다. 이 순간의 유휴 상태가 영구 배정을 뜻하지는 않는다. 사용자가 두 GPU 사용을 허용했으며 이번 검증과 문서의 배정 예시는 YACHIYO/UWIPL → GPU 0, PoseTrack → GPU 1이다. 동시 전체 추론의 메모리 사용량은 아직 측정하지 않았으므로 본 단계에서는 짧은 연산만 수행한다. 실행 스크립트에는 GPU 기본 번호를 넣지 않아 호출 시 명시해야 한다. 향후 어댑터의 GPU 필요 여부는 어댑터 구현 단계에서 결정한다.

`nvidia-smi`의 CUDA 13.0 표시는 드라이버가 지원하는 CUDA 상한이며 컨테이너가 CUDA 13을 사용해야 한다는 뜻이 아니다. 공식 모델 코드에 맞춘 CUDA 11.3/11.8 런타임을 사용한다. 580 드라이버의 이전 CUDA 바이너리 호환성은 [NVIDIA 공식 호환성 문서](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)를 기준으로 검토했다. TITAN RTX의 compute capability는 [NVIDIA GPU 목록](https://developer.nvidia.com/cuda/gpus)의 7.5이며, 소스 CUDA 확장이 필요할 때 `sm_75`를 지원해야 한다. 호스트 CUDA Toolkit 설치나 드라이버 교체는 필요하지 않다.

공용 서버 설정 변경, Docker 서비스 재시작, 그룹 변경, GPU 리셋/MIG 변경을 수행하지 않았다. 이번에 생성하는 검증 컨테이너에만 `--rm`을 적용하며 기존 컨테이너를 삭제하지 않는다.

종료 점검(`logs/server-after.txt`): 컨테이너는 여전히 총 48개/실행 33개이고, 완성한 프로젝트 이미지 9개가 모두 존재한다. 전체 이미지 수는 빌드 재시도에 따른 중간 이미지까지 포함해 38개다. 기존 이미지와 빌드 캐시는 삭제하지 않았다. 작업 HDD 여유 약 937GiB, Docker SSD 여유 약 314GiB로, 시작 대비 SSD 사용량은 약 24GiB 증가했다. 이미지별 논리 크기에는 공유 레이어가 중복 포함되므로 크기를 단순 합산한 값과 실제 디스크 증가는 다르다. 최종 모델 환경은 CPU 9/9와 GPU 8/8 검증을 모두 통과했으며 [검증 기록](validation.md)에 증거를 연결했다.
