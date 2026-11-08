# SCOUT 첫 8개 카메라 기하 진단

**후속 작업(2026-10-28):** [어댑터 v0.4](scout-ground-projection.md)에서 왜곡 역변환과 실제 광선–지면 메시 교차를 구현했다. 아래는 2026-10-27의 보존 기록이다.

2026-10-27. [진단 스크립트](../scripts/audit-scout-geometry.py)는 `cam_0..cam_7`의 실제 보정과 `mesh_ground_only.ply`를 읽고 대수적 일관성·왜곡 영향·전역 평면 잔차를 계산한다. **모델용 calibration/H는 발행하지 않았다.** GT를 읽거나 보정·평면 추정에 사용하지 않았으며, 가중치·모델 추론도 실행하지 않았다.

기존 `scout-mtmc/adapter:0.3` CPU Docker에서 데이터·스크립트 읽기 전용, 출력만 쓰기 가능, 네트워크·GPU 차단, CPU 1개·RAM 2GiB로 실행했다. NumPy 1.23.5/OpenCV 4.8.1을 사용했으며 호스트 패키지는 설치하지 않았다. 결과는 [geometry-audit.json](../outputs/scout-first8-geometry-20261027/geometry-audit.json), 실행 로그는 [geometry 로그](../logs/scout-first8-geometry-20261027.log)에 있다.

## 보정행렬과 왜곡

| 검사 | 실제 결과 | 의미 |
|---|---|---|
| K/R/T/dist 유한값·구조, K 비특이성 | 8개 모두 정상 처리 | 파일의 수치 구조 확인 |
| R determinant / `RᵀR−I` | det≈1, 최대 절대 직교 오차 `8.88×10⁻¹⁶` | 회전행렬의 수학적 조건 충족 |
| `P=K[R\|T]` 합성 좌표 왕복 | 최대 `1.69×10⁻¹¹px` | 동일 행렬로 구성한 대수적 왕복, 실세계 보정 정확도 검증은 아님 |
| ideal→distorted 픽셀 변위 | 중앙값 5.608px, p95 65.098px, 최대 106.471px | 지정 격자에서 왜곡의 영향 |
| raw→기본 undistort→redistort 왕복 | 중앙값 약 `9.85×10⁻¹¹px`, p95 1.284px, 최대 8.792px | 기본 역변환의 일부 점 수렴 정밀도가 부족할 수 있음 |

1920×1080 범위의 21×13, 총 273개 합성 격자점을 사용했다. 왜곡 변위는 ideal 격자에서, undistort 왕복은 raw 격자에서 측정했다. 실제 관측점의 보정 잔차나 선택 영상의 사람 위치 오차를 측정한 결과가 아니다. 8개 보정 JSON의 K와 dist 값은 서로 같고 R/T는 다르므로 왜곡 격자 통계가 동일하다.

기본 `cv2.undistortPoints`의 최대 왕복 오차만으로 보정계수 오류를 확정할 수 없다. 후속 적용에서는 역변환 반복 수·수렴 조건과 유효 픽셀 영역을 점검하고, bbox/발 위치 및 최종 결과의 좌표 체계를 일관되게 처리해야 한다.

## 지면 mesh와 평면 가정

PLY는 binary little-endian, vertex는 float xyz **497,662개**, header의 face 수는 984,425개다. 전체 vertex를 구조화 배열로 읽었고 face topology·광선 교차·가림은 검사하지 않았다.

- XYZ 범위: `[-115.013,-158.462,-4.345]`부터 `[58.243,199.066,0.849]`까지.
- GT를 사용하지 않은 전체 vertex PCA 평면의 중심: `[6.7010,18.6504,-0.1471]`.
- 평면 단위 법선: `[-0.00295462,0.00237144,0.99999282]`.
- 평면까지 절대 거리: 중앙값 **0.20273**, RMS **0.39394**, p95 **0.66151**, 최대 **3.82734**. 단위는 파일의 world 좌표 단위다.

이는 vertex별 동일 가중치의 전역 적합으로, 삼각형 면적 가중치나 첫 8개 카메라의 실제 보행 영역을 반영하지 않았다. [논문 §4.2](https://arxiv.org/html/2507.08494v2#S4.SS2)는 metric scene reconstruction을 설명하지만 이 검사에서 실제 길이 기준점과 mesh·카메라의 축/단위를 독립적으로 대조하지 않았다. 따라서 수치를 곧바로 cm/m 정확도로 확정하지 않으며, `Z=0`이나 단일 전역 평면이 모델에 적합하다고 간주하지 않는다. 카메라별 유효 지면 영역·경사·계단·mesh 이상점과 공통 좌표계를 먼저 확인해야 한다.

## 기준과 다음 연결

공식 toolkit의 고정 커밋 `1acd79513bfcf16aecef1150283af58c8273542e` [geometry.py](https://github.com/cvlab-epfl/scout_toolkit/blob/1acd79513bfcf16aecef1150283af58c8273542e/src/geometry.py)는 world→camera R/T와 distortion을 포함한 `projectPoints`를 사용한다(8–44행). 역방향은 undistort 후 `−RᵀT`에서 시작하는 광선을 mesh와 교차시킨다(47–145행). 로컬 `/tmp/scout-format-reference/src/geometry.py`와 `loading.py`의 SHA-256이 [형식 lock](../adapters/scout-format.lock.json)과 일치함을 확인했다.

이번 스크립트는 공식 함수를 실행·복사한 모델 실행기가 아니라 NumPy/OpenCV의 작은 수치 진단이다. JSON에 저장한 P도 참고값이며 원본 모델이 자동으로 읽는 보정 파일이 아니다. 다음 연결에는 유효 지면과 단위 확인, 왜곡 처리 및 결과 역변환, 검증된 기하를 각 모델에 전달하는 실행 계층이 필요하다. GT 비교, 광선–mesh 교차 구현과 모델별 native H 생성은 이번 축소 범위에서 수행하지 않았다.

컨테이너 안의 호출 형식은 다음과 같다. 데이터와 스크립트는 읽기 전용으로 연결하고 새 출력 폴더만 쓰기 가능하게 연결한다. 기존 `geometry-audit.json`은 덮어쓰지 않는다.

```bash
python -B /audit.py --data /data/scout --output /outputs
```
