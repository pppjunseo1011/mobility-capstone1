# 공식 가중치 링크 확인 — 2026-10-31

공식 README, 모델 실행 코드, 공개 다운로드 안내 페이지를 대조했다. 세 모델 모두 공식 저장소에서 안내하는 가중치 링크가 있다. 서버에 가중치가 없다는 기존 기록은 공개 배포 여부와 별개다.

최초 링크 조사에서는 Drive 공개 HTML을 요청당 최대 1 MiB로 제한하고 OpenMMLab 파일은 HEAD만 조회했다. 이후 사용자 요청으로 **PoseTrack 3개, YACHIYO 3개, UWIPL 실영상용 7개 및 추가 YOLOv7 5개를 확보**했다. 모델 간 공통 파일은 포함하여 센 개수다. UWIPL 합성 영상용 ReID 1개는 공식 링크가 HTTP 404여서 미확보다. 이후 소량 SCOUT 16장으로 세 모델의 실영상용 검출·포즈·ReID 로딩과 추론을 통과했다. UWIPL 합성 영상용 YOLOv7은 아직 추론하지 않았다. [소량 추론 결과](scout-small-inference.md).

## PoseTrack 다운로드 완료 — 2026-10-31 13:44 KST

저장 위치: `weights/posetrack/`. 공식 Drive 링크에서 총 **1,419,500,774 bytes (약 1.42 GB)**를 받았다. 세 파일 모두 사전에 확인한 파일 크기와 일치하고 ZIP 내부 CRC 검사를 통과했다. SHA-256은 로컬 파일 식별용이며, 배포자가 공개한 전체 체크섬과 대조한 결과는 아니다. 다운로드 단계에서는 모델 의존성 설치나 GPU 추론을 실행하지 않았다. 후속 소량 추론 결과는 위 문서에서 구분한다.

| 파일 | bytes | SHA-256 |
|---|---:|---|
| `bytetrack_x_mot17.pth.tar` | 792835731 | `e3945f3523fde1e107708aacd64dab0670c34e371d136e54587cac7a50d3cfba` |
| `td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth` | 121739901 | `81c58e403a1ba13a741eb02ba790b5b78bff2a45a9390feebf875f43b8502c8f` |
| `aic24.pkl` | 504925142 | `bb8df466262e698e1d8d5d4a8f598240b9e14c02375a688a04f0f7d94179c822` |

출처·크기·해시는 `weights/posetrack/manifest.json`, `weights/posetrack/SHA256SUMS`에 저장했다. [다운로드 로그](../logs/posetrack-weights-download-20261031.log), [검증 기록](../logs/posetrack-weights-download-validation-20261031.json).

프로젝트 루트에서 재다운로드/중단 재개 명령은 `python3 -u scripts/download-posetrack-weights.py`이다. 기존 완성 파일은 크기를 확인한 뒤 보존하고 SHA-256을 다시 기록한다. 부분 다운로드는 `.part`로 보관하고 서버의 Range 응답을 검증한 뒤 이어받는다. 호스트 추가 패키지는 필요 없으며 속도 제한을 설정하지 않았다. 다운로드 폴더는 Git과 Docker 빌드에서 제외된다.

## YACHIYO·UWIPL 다운로드 결과 — 2026-10-31 14:04 KST

| 모델/경로 | 확보 | 상태 |
|---|---:|---|
| YACHIYO | 검출 1 + ReID 1 + 포즈 1 = 3개 | 다운로드 완료 |
| UWIPL 실제 영상 경로 | 검출 1 + ReID 5 + 포즈 1 = 7개 | 다운로드 완료 |
| UWIPL 합성 영상 경로 | YOLOv7 5개 | 다운로드 완료, 창고용 파일명 불일치 주의 |
| UWIPL 합성 영상용 ReID | 0/1개 | 공식 링크를 재조회해도 HTTP 404, 미확보 |

이번에 받은 서로 다른 파일은 총 **12개, 5,595,891,943 bytes (약 5.60 GB)**다. `weights/shared/`에 저장하고 `weights/yachiyo/`, `weights/uwipl/`에 하드링크로 연결했다. 각 모델 폴더만 컨테이너에 읽기 전용으로 연결해도 외부 경로를 참조하는 심볼릭 링크 문제 없이 파일에 접근할 수 있다. 원본 모델 저장소는 수정하지 않았다. 기존 PoseTrack 폴더도 보존했다.

12개 모두 사전 확인한 크기와 일치하며 SHA-256을 기록하고 재검사했다. ZIP 형식 9개는 내부 CRC 검사를 통과했다. 구형 비-ZIP 형식 3개(OSNet Market/MSMT17, HRNet-W48)는 CRC 검사 대상이 아니며 크기·해시 확인까지만 수행했다. 다운로드 확인 시점에는 체크포인트 역직렬화나 모델 로딩을 하지 않았다. 이후 실영상용 체크포인트는 소량 추론에서 로딩을 검증했다. 해시는 로컬 식별용이고 배포자의 공개 전체 체크섬과 대조한 결과는 아니다.

### 확보한 파일과 실제 경로

`weights/<모델>/` 아래 상대 경로다. 공통 파일은 양쪽 모델 폴더에 있다.

| 모델 | 상대 경로 | bytes | 공식 출처 |
|---|---|---:|---|
| yachiyo, uwipl | `bytetrack_x_mot17.pth.tar` | 792835731 | [다운로드 안내](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view) |
| yachiyo, uwipl | `osnet_ms_m_c.pth.tar` | 56337627 | [다운로드 안내](https://drive.google.com/file/d/1UxUI4NsE108UCvcy3O1Ufe73nIVPKCiu/view) |
| uwipl | `osnet_ibn_ms_m_c.pth.tar` | 56357648 | [다운로드 안내](https://drive.google.com/file/d/1Sk-2SSwKAF8n1Z4p_Lm_pl0E6v2WlIBn/view) |
| uwipl | `osnet_ain_ms_m_c.pth.tar` | 56328340 | [다운로드 안내](https://drive.google.com/file/d/1YjJ1ZprCmaKG6MH2P9nScB9FL_Utf9t1/view) |
| uwipl | `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth` | 10399605 | [다운로드 안내](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view) |
| uwipl | `osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth` | 10994685 | [다운로드 안내](https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M/view) |
| yachiyo, uwipl | `hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth` | 255011654 | [다운로드 안내](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) |
| uwipl | `synthetic/yolov7_base/epoch_100_base.pth` | 945441997 | [다운로드 안내](https://drive.google.com/file/d/1ld1Y0n0jHoiwGzeMlEr47QRCjX61K55Y/view) |
| uwipl | `synthetic/yolov7_hospital/epoch_10_S014_hospital.pth` | 851957147 | [다운로드 안내](https://drive.google.com/file/d/13waNxnEhxuv4p7ceWPT3-TcNB4QEstVA/view) |
| uwipl | `synthetic/yolov7_market/epoch_10_S003_market.pth` | 854080845 | [다운로드 안내](https://drive.google.com/file/d/1SoJf9unvV_YcWccozHjt562aPaCyKjp7/view) |
| uwipl | `synthetic/yolov7_office/epoch_10_S009_office.pth` | 854289613 | [다운로드 안내](https://drive.google.com/file/d/1LUZeWek6QxRhjqB7gbvap7z0vOl4YksJ/view) |
| uwipl | `synthetic/yolov7_storage/epoch_10_S018_storage.pth` | 851857051 | [다운로드 안내](https://drive.google.com/file/d/1vXSofMptvS1fGiT6gx6WJH3BXzZlrxhn/view) |

### 남은 문제와 사용 안내

- **합성 ReID 미확보:** `1cP-3esZSnktw64SXMHn5cDk6BX15e2-q` 공식 Drive 링크가 HTTP 404를 반환한다. 다른 가중치로 임의 대체하지 않았다. UWIPL 실영상 경로의 OSNet 5개는 모두 확보했다.
- **창고용 파일명 불일치:** 공식 `scripts/4_inference_det_syn.sh`는 `epoch_100_S018_storage.pth`를 지정하지만, 공식 폴더의 배포 파일은 `epoch_10_S018_storage.pth`이다. 배포 이름을 그대로 보존했다. 해당 경로로 실행 설정을 바꾸고 모델 호환성을 검증해야 하며, 파일 이름만 보고 학습 epoch가 같다고 간주하지 않는다.
- **검출 체크포인트 동일성:** 이번 YACHIYO/UWIPL 검출 파일은 기존 PoseTrack 검출 파일과 SHA-256이 같다. URL은 서로 다르지만 다운로드 후 파일 내용의 동일성을 확인했다.
- 각 `weights/<모델>/manifest.json`, `SHA256SUMS`에 출처·크기·해시를 기록했다. UWIPL manifest의 `complete=false`는 합성 ReID 미확보를 나타내며 `real_video_weights_complete=true`와 구분한다.
- 재실행/중단 재개: 프로젝트 루트에서 `python3 -u scripts/download-other-model-weights.py`. 완료 파일은 보존하고 해시를 다시 기록하며 남은 파일만 다운로드한다. 접근 오류는 파일별로 기록하고 다른 파일의 작업은 계속한다. 이 명령은 다운로드 결과가 일부 미완료여도 정상 종료할 수 있으므로 report의 `failures`를 확인한다.
- 컨테이너에는 `--weights weights/yachiyo` 또는 `--weights weights/uwipl`처럼 모델별 폴더를 지정한다. 소량 실행기는 체크포인트를 `/weights/<상대 경로>`로 연결했다. 원본 전체 추적 실행기의 경로 연결은 별도 작업이다.

[다운로드 로그](../logs/other-model-weights-download-20261031.log), [크기·해시·CRC·미확보 기록](../logs/other-model-weights-download-validation-20261031.json), [다운로드 스크립트](../scripts/download-other-model-weights.py).

## 모델별 사용 범위

| 모델 | 검출 | ReID | 포즈 | 현재 판단 |
|---|---|---|---|---|
| YACHIYO | ByteTrack YOLOX-X | OSNet `osnet_ms_m_c.pth.tar` 1개 | HRNet-W48 | 세 단계 링크 접근 확인 |
| UWIPL 실제 영상 S001 경로 | ByteTrack YOLOX-X | OSNet 계열 5개 | HRNet-W48 | 실제 영상 경로에 쓰는 링크 접근 확인 |
| UWIPL 합성 영상 경로 | 별도 YOLOv7 폴더 | 별도 합성 데이터용 ReID | HRNet-W48 | 검출 폴더 접근 가능, 합성 ReID 링크 HTTP 404 |
| PoseTrack | ByteTrack YOLOX-X | 미세 조정된 `aic24.pkl` | HRNet-W32 | 첫 추론용 세 파일 메타데이터 확인 |

UWIPL의 S001 경로를 SCOUT 실영상에 적용하는 것은 후속 실험 선택이다. 가중치 확보만으로 AI City 전용 카메라 구성·보정·임계값까지 SCOUT에 적용 가능해지는 것은 아니다.

## 개별 링크와 확인 결과

MB는 1,000,000 bytes 기준이며 Drive 크기는 공개 메타데이터 값이다.

| 용도/이름 | 공식 안내 파일/링크 | 크기 | 확인 결과 |
|---|---|---|---|
| yachiyo_uwipl_detector | [bytetrack_x_mot17.pth.tar](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view) | 792.84 MB | HTTP 200 · 파일명/크기 확인 |
| osnet_ms_m_c | [osnet_ms_m_c.pth.tar](https://drive.google.com/file/d/1UxUI4NsE108UCvcy3O1Ufe73nIVPKCiu/view) | 56.34 MB | HTTP 200 · 파일명/크기 확인 |
| osnet_ibn_ms_m_c | [osnet_ibn_ms_m_c.pth.tar](https://drive.google.com/file/d/1Sk-2SSwKAF8n1Z4p_Lm_pl0E6v2WlIBn/view) | 56.36 MB | HTTP 200 · 파일명/크기 확인 |
| osnet_ain_ms_m_c | [osnet_ain_ms_m_c.pth.tar](https://drive.google.com/file/d/1YjJ1ZprCmaKG6MH2P9nScB9FL_Utf9t1/view) | 56.33 MB | HTTP 200 · 파일명/크기 확인 |
| osnet_x1_0_market | [osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view) | 10.40 MB | HTTP 200 · 파일명/크기 확인 |
| osnet_x1_0_msmt17 | [osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth](https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M/view) | 10.99 MB | HTTP 200 · 파일명/크기 확인 |
| uwipl_synthetic_reid | [uwipl_synthetic_reid](https://drive.google.com/file/d/1cP-3esZSnktw64SXMHn5cDk6BX15e2-q/view) | 미확인 | HTTP 404 · 현재 접근 실패 |
| posetrack_detector | [bytetrack_x_mot17.pth.tar](https://drive.google.com/file/d/1LVFqYqx88R0TUjCMbTaKrJkL7-SdCSmC/view) | 792.84 MB | HTTP 200 · 파일명/크기 확인 |
| posetrack_pose | [td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth](https://drive.google.com/file/d/1tNT6gOBB95qYPCypvCctj1o-r7bzdwxA/view) | 121.74 MB | HTTP 200 · 파일명/크기 확인 |
| posetrack_reid | [aic24.pkl](https://drive.google.com/file/d/17qbBmBX7DiT2lOuQ6rGHl8s9deKHkVn2/view) | 504.93 MB | HTTP 200 · 파일명/크기 확인 |
| uwipl_yolov7_folder | [detector_weights - Google Drive](https://drive.google.com/drive/folders/10LT1BlBAfYnr-fJjka_Lnzf4nH0N723-) | 미확인 | 폴더 페이지 접근 · 내부 파일 미검증 |
| yachiyo_uwipl_hrnet_w48 | [yachiyo_uwipl_hrnet_w48](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | 255.01 MB | HTTP 200 · HEAD 확인 |

## 실행 시 주의할 구체적인 차이

- YACHIYO `embedder/aic24_extract.py`는 `deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar`를 지정한다. README의 자동 다운로드 설명만 믿으면 안 된다. 사용 중인 FeatureExtractor는 지정 파일이 없을 때 `pretrained=True`로 모델을 구성하므로 의도한 ReID 체크포인트를 명시적으로 준비하고 로딩을 검증해야 한다.
- UWIPL `deep-person-reid/torchreid/aic_extract_S001.py`는 표의 OSNet 5개를 사용한다. README의 미리 계산된 검출 TXT와 임베딩 NPY는 AI City 데이터의 결과이며 SCOUT 입력에 재사용할 가중치가 아니다.
- PoseTrack은 검출 792,835,731 bytes, HRNet-W32 121,739,901 bytes, `aic24.pkl` 504,925,142 bytes로 총 **1,419,500,774 bytes (약 1.42 GB)**이다. README의 LUPerson 사전학습 모델은 미세 조정 학습 준비와 구분하며, 첫 테스트는 제공된 미세 조정 ReID 모델을 사용한다.
- YACHIYO/UWIPL은 동일한 검출 링크를 안내한다. PoseTrack의 검출 링크는 별도 ID이나, 이번 다운로드 후 SHA-256 일치로 파일 내용의 동일성을 확인했다.
- UWIPL 합성 ReID의 HTTP 404는 이번 비로그인 요청의 결과다. 삭제인지 권한 문제인지는 확인하지 못했다. YOLOv7은 최초 조사 당시 폴더 페이지만 확인했으나, 이후 위 다운로드 작업에서 개별 체크포인트 5개를 확보했다.

## 다음 소규모 검증 범위

PoseTrack뿐 아니라 세 모델 모두 기존 Docker에서 실영상용 체크포인트 로딩과 SCOUT 16장의 검출·포즈·ReID 추론을 완료했다. PoseTrack은 GPU 1, YACHIYO/UWIPL은 GPU 0을 사용했다. 다음은 짧은 연속 구간의 추적 ID 연결이다. 전체 추론·학습·성능 평가와 다중 카메라 ID 연결 성공 여부는 이 단계와 구분한다.

## 근거

- [YACHIYO 공식 README](https://github.com/riips/AIC24_Track1_YACHIYO_RIIPS) 및 `scripts/pose.sh`, `embedder/aic24_extract.py`
- [UWIPL 공식 README](https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI) 및 `deep-person-reid/torchreid/aic_extract_S001.py`
- [PoseTrack 공식 master README](https://github.com/ZhenyuX1E/PoseTrack/tree/master)
- [torchreid 공식 모델 목록](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
- 기계 판독용 조회 결과: [`../logs/weight-link-audit-20261031.json`](../logs/weight-link-audit-20261031.json). 커밋은 해당 파일의 `sources`에 기록.
