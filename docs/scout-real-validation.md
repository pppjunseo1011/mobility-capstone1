# 실제 SCOUT 표본 검증 — 2026-10-27

이 문서는 **v0.2의 최초 실제 8프레임 검증 기록**이다. 이후 진행한 연속 20초 구간·픽셀 일치 수정·YACHIYO frame 연결은 [v0.3 다음 단계 문서](scout-next-stage.md)를 따른다.

압축 해제된 실제 데이터로 기존 어댑터 v0.2와 모델 컨테이너의 입력 읽기를 검증했다. **가중치는 사용하지 않았으며 검출·포즈·ReID·추적 추론과 성능 평가는 실행하지 않았다.** 원본 데이터와 모델 저장소, 기존 이미지를 보존했다.

## 현재 테스트 범위

사용자 지정 범위는 **sequence_1의 cam_0~cam_7, 총 8개**다. 이번 표본은 각 카메라의 source frame **0~7(8장)**, 총 **JPEG 64장**이다. 공식 논문은 첫 시퀀스의 첫 8개 카메라를 1FPS로 수동 정리하고 10FPS로 선형 보간했다고 설명한다. 모든 프레임의 주석이 완벽하다는 의미는 아니다. [논문 §4.3](https://arxiv.org/html/2507.08494v2#S4.SS3).

공식 UMPN의 `scoutval`은 `sequence_1`과 camera ID `[0,1,2,3,4,5,6,7]`을 사용하고, 로더는 그 ID를 `images/sequence_1/cam_{id}`에 연결한다. 따라서 cam_8은 현재 범위에 포함하지 않는다. [공식 factory.py](https://github.com/cvlab-epfl/UMPN/blob/main/datasets/factory.py#L27-L39), [공식 scout.py](https://github.com/cvlab-epfl/UMPN/blob/main/datasets/scout.py#L61-L69).

이 근거는 주석 생성 방식과 카메라 범위를 확인한 것이다. 개별 bbox의 정확성을 검수한 것은 아니므로 기존 공통 CSV의 `label_quality=unverified`는 유지한다. 이번 8프레임은 입력 점검용이며 최종 평가 구간·학습/검증 분할로 지정하지 않는다.

## 실제 데이터 확인

- 2026-10-27 14:31:10 KST에 추출 완료: 1,124,853개 파일, 374,109,919,942바이트. 원본 압축 파일 SHA-256 재대조 성공.
- 실제 JPEG 파일명 562,427개: sequence_1 299,054개, sequence_2 263,373개. 보정 25개와 timestamp 2개는 기존 분석용 메타데이터와 해시가 모두 같다.
- 전체 데이터에 timestamp 없는 끝부분 이미지 **307장**이 있다. timestamp가 있는 camera/frame에 대응하는 JPEG 누락은 없었다. 전체 파일명 조사는 전체 이미지 디코딩 검증과 다르다.
- 현재 선택한 cam_0~cam_7은 각각 JPEG와 timestamp가 0~11959 범위로 존재한다. 표본 64장은 실제 디코딩해 모두 1920×1080임을 확인했다.
- 표본 카메라 간 동일 frame의 최대 시간 차이는 **29ms**, 현재 진단 기준 50ms 이내다. 시간 재정렬·보간은 하지 않았고, 전체 길이에서 동기화가 보장된다는 의미도 아니다.
- mesh 7개 파일은 존재·크기·헤더만 확인했다. 지면 평면, 보정 좌표계와의 일치, 단위는 검증하지 않았다.

전체 파일 목록과 차집합: [inventory.json](../outputs/scout-real-inventory-20261027/inventory.json).

## 어댑터 결과

현재 출력 루트는 `outputs/scout-first8-validation-20261027`이다.

| 항목 | 결과 |
|---|---|
| 모델별 입력 배치 | YACHIYO·UWIPL·PoseTrack 3개 export 생성 |
| JPEG 복사 | 모든 복사본이 원본과 SHA-256 일치; UWIPL의 img/frame 복사본 모두 확인 |
| 카메라·frame·timestamp | 각 export 64행, source ID·시간 보존; 중복·누락 거부 |
| 번호 체계 | cam_N → 모델 ID N+1; YACHIYO frame 1~8, UWIPL/PoseTrack 0~7 |
| MP4 | 모델별 8개, 총 24개; 각 8프레임·1920×1080·10FPS |
| 원본 JPEG → MP4 | FFmpeg 기준 디코딩 픽셀과 순서 정확히 일치 |
| 원본 보존 | 선택 JPEG의 SHA-256과 수정 시각이 실행 전후 일치 |
| MOT GT | 공통 ground_truth.csv 1,401행 생성 |
| COCO GT | 이미지 인덱스 48/64 누락으로 엄격 변환 실패; 빈 GT로 대체하지 않음 |
| 모델 추론 | 미실행; model_inference_ready=false 유지 |

[어댑터 상세 결과](../outputs/scout-first8-validation-20261027/validation.json)는 **31개 PASS, 1개 실패**이며 `state=completed_with_failures`다. 실패는 COCO 인덱스 누락으로 재현됐고, 명령 종료 코드도 1로 보존했다. 입력 배치·영상 변환은 통과했지만 전체 검증 성공으로 바꾸지 않는다.

### COCO와 MOT의 차이

선택 64장 중 COCO `images`에는 cam_3/4의 16장만 있고 나머지 48장은 없다. 반면 COCO `annotations` 본문에는 누락 이미지에 해당하는 950행을 포함해 **1,401행이 모두 존재**한다. 독립적으로 읽으면 MOT와 camera/frame/person ID와 bbox가 전부 일치한다. world 좌표의 최대 차이 약 0.005는 MOT 소수 둘째 자리 반올림에 해당한다.

따라서 이를 “사람이 없는 프레임”이나 “주석이 없는 카메라”로 해석하면 안 된다. 현재 검증에는 **MOT GT를 사용**한다. COCO 원본 수정이나 인덱스 자동 복구는 하지 않았다. [독립 GT 대조](../outputs/scout-first8-audit-20261027/audit.json).

### JPEG와 MP4의 픽셀 차이

FFmpeg가 JPEG를 읽은 결과와 MP4 복원 결과는 정확히 같다. 다만 OpenCV가 원본 JPEG를 직접 읽은 결과와 MP4를 읽은 결과는 JPEG 디코더의 색 변환 차이 때문에 완전히 같지 않다. 현재 표본의 8bit 채널 값 기준 평균 절대 차이는 약 0.525~0.627, 최대 차이는 27이었다. 좌표 이동이나 프레임 누락으로 처리할 문제는 아니지만, 비교 실험에서는 **세 모델의 입력 디코딩 정책을 통일할 필요가 있다.** 이 차이가 추적 성능에 미치는 영향은 아직 측정하지 않았다.

## 모델 컨테이너의 영상 입력 검사

기존 YACHIYO pose, UWIPL pose, PoseTrack all 이미지에서 GPU·가중치 없이 `cv2.VideoCapture`, `mmcv.VideoReader`, FFmpeg로 순차 디코딩했다. **각 모델 8개 영상, 총 24/24개를 통과**했다. 각 영상의 8프레임·1920×1080·10FPS와 프레임별 픽셀 해시가 세 reader 사이에 일치했다. 실제 신경망이나 원본 추론 CLI는 호출하지 않았다. [컨테이너 검사 JSON](../logs/model-video-first8-inputs-20261027/validation.json).

어댑터의 원본→영상 검사 당시 MP4 해시와 모델 컨테이너가 읽은 MP4 해시, 최종 파일 해시도 **24개 모두 일치**한다. 원본 순서 검사와 후속 영상 reader 검사가 동일한 파일에 대한 것임을 확인했다. [종합 결과와 해시 연결](../logs/scout-first8-summary-20261027.json).

검증에 쓰는 이미지는 아래와 같고, 실행 전 ID를 확인한 뒤 그 ID로 컨테이너를 시작한다.

| 용도 | 이미지 |
|---|---|
| 공통 어댑터 | scout-mtmc/adapter:0.2 |
| YACHIYO 영상 입력 | scout-mtmc/yachiyo-pose:f881fe0 |
| UWIPL 영상 입력 | scout-mtmc/uwipl-pose:8e23a69 |
| PoseTrack 영상 입력 | scout-mtmc/posetrack-all:ba727e6 |

원래 환경 코드 25개와 어댑터 코드·테스트 등 18개 파일의 기존 해시가 유지됐고, 모델 저장소 3개도 잠긴 커밋에서 변경 없이 보존됐다. [보존 점검](../logs/scout-real-preservation-20261027.json). 기존 CPU/CUDA 환경 smoke 결과는 [환경 검증 문서](validation.md)를 따른다. 이번에 GPU 연산을 추가 실행하지 않았다.

## 재실행

항상 **새 출력 폴더**를 사용한다. 모델 의존성은 기존 컨테이너 안에 있으며 호스트에는 설치하지 않는다. CPU 2개·RAM 4GiB·네트워크 없음·입력 읽기 전용으로 실행한다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
python3 logs/scout-v02-final-validator.py run \
  --data datasets/scout/extracted/scout \
  --output outputs/scout-first8-validation-new \
  --sequence 1 --cameras cam_0 cam_1 cam_2 cam_3 cam_4 cam_5 cam_6 cam_7 \
  --frames 8

# 위 명령은 알려진 COCO 인덱스 문제로 exit 1을 반환한다.
# 생성된 입력·영상과 MOT GT는 유효하며, 오류는 validation.json에서 확인한다.
python3 scripts/validate-model-video-inputs.py run \
  --exports-root outputs/scout-first8-validation-new \
  --output logs/model-video-first8-inputs-new \
  --sequences 1 --camera-count 8 --frames 8
```

두 스크립트의 기본 범위도 first8 표본에 맞춘다. 일반 어댑터 CLI는 다른 연구용 입력도 다룰 수 있지만 현재 비교 실험의 대상은 위 8개 카메라다. 결과 CSV는 `sequence1-gt-mot/ground_truth.csv`, 모델별 입력은 `sequence1-{yachiyo,uwipl,posetrack}/`에 있다.

## 다음 단계

1. 같은 8개 카메라에서 더 긴 공통 구간을 정하고 시간 정렬 및 JPEG/영상 디코딩 정책을 확정한다. 현재 0.8초 표본은 카메라 간 이동·ID 유지 성능을 판단하기에 부족하다.
2. SCOUT의 왜곡·지면·세계 좌표 단위를 검증하고 모델용 보정과 실행 경로를 연결한다. YACHIYO의 포즈 frame 1/0 기준 차이, UWIPL의 7카메라 전용 설정·anchor frame·특정 인물 처리, PoseTrack의 P/H·경로·검출/포즈/ReID 행 대응도 해결해야 한다. **8개 카메라 폴더 생성만으로 이 문제가 해결되지는 않는다.**
3. 팀원 가중치가 준비되면 먼저 검출·포즈·ReID의 작은 실제 추론부터 실행하고, 추적 연결과 공통 결과 정규화·평가로 확장한다. 가중치 없이 만든 가짜 예측이나 GT를 모델 출력으로 사용하지 않는다.

사용자가 범위를 지정하기 전에 수행한 두 sequence·각 2카메라·32프레임 검사는 `outputs/scout-real-validation-20261027`과 `logs/model-video-inputs-20261027`에 보존했다. 입력/영상 검사와 모델 이미지별 영상 읽기는 통과했고, sequence_2 COCO는 같은 종류의 인덱스 누락으로 실패했다. 이는 과거 형식 진단이며 현재 평가 대상에 포함하지 않는다.
