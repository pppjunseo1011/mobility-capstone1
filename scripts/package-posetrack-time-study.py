#!/usr/bin/env python3
"""Prepare a review bundle, then seal it after standalone evaluation verification."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
O = R / 'outputs/posetrack-time-study-20261106'
OLD = R / 'outputs/posetrack-mesh-20261104'
H = O / 'handoff'
RUNS = ['original_shared', 'B_timestamp', 'C_timestamp', 'C_height_abstain', 'B_same_output', 'C_same_output']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def put(src, dest):
    assert src.is_file(), src
    target = H / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)

def tree(src, dest):
    for f in sorted(src.rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:
            put(f, Path(dest) / f.relative_to(src))

def write(dest, text):
    p = H / dest
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + '\n')

def prepare():
    H.mkdir(exist_ok=True)
    put(O / 'report/START_HERE.md', 'START_HERE.md')
    for f in (O / 'report').glob('PoseTrack*'):
        put(f, '00_report/' + f.name)
        shutil.copy2(f, R / 'docs/handoff' / f.name)
    tree(O / 'summary', '01_summary')
    for name in ['frame_mapping.csv', 'frame_mapping_all_sources.csv', 'timestamp_before_after.csv', 'timestamp_policy.json']:
        put(O / name, '01_summary/' + name)
    put(O / 'aligned_input/preparation_summary.json', '01_summary/aligned_input_summary.json')
    tree(O / 'tracking_results', '02_tracking_results')
    put(O / 'summary/per_camera_summary.csv', '02_tracking_results/per_camera_summary.csv')
    for name in ['metrics_comparison.csv', 'metrics_comparison.json', 'ground_affinity_B_vs_C.csv']:
        put(O / 'summary' / name, '03_evaluation/' + name)
    for run in RUNS:
        for name in ['metrics.json', 'per_camera_metrics.csv', 'protocol.json', 'gt_matching_results.csv']:
            put(O / 'evaluation' / run / name, f'03_evaluation/{run}/{name}')
    put(O / 'evaluation/C_timestamp/selected_gt.json', '03_evaluation/shared_gt.json')
    tree(O / 'evaluation/same_output_observations', '03_evaluation/same_output_observations')
    tree(O / 'geometry', '04_geometry')
    put(O / 'association/affinity_examples.csv', '04_geometry/geometry_affinity_examples.csv')
    tree(O / 'association', '05_association')
    put(OLD / 'C_30s/feature_bank_updates.npz', '05_association/original_feature_bank_updates.npz')
    for frame, cam in [(17, 3), (20, 7)]:
        name = f'frame_{frame:03d}_cam_{cam}.npz'
        put(OLD / 'C_30s/matrices' / name, '05_association/original_case_matrices/' + name)
    tree(O / 'visualizations', '06_visualizations')
    scripts = [
        'prepare-posetrack-time-study.py', 'run-posetrack-time-study.py', 'posetrack-time-extra-trace.txt',
        'evaluate-posetrack-time-study.py', 'scout-common-evaluation-metrics.py', 'normalize-scout-common-evaluation.py',
        'audit-posetrack-time-geometry.py', 'audit-posetrack-neural-gt.py', 'analyze-posetrack-time-identities.py',
        'audit-posetrack-output-loss.py', 'render-posetrack-time-study.py', 'verify-posetrack-time-study.py',
        'replay-posetrack-time-study.sh', 'summarize-posetrack-time-study.py',
        'extract-posetrack-time-review-evidence.py', 'report-posetrack-time-study.py',
        'package-posetrack-time-study.py', 'posetrack-mesh-geometry.py', 'trace-posetrack-flat.py',
        'evaluate-uwipl-study.py', 'run-posetrack-mesh.py',
    ]
    for name in scripts:
        put(R / 'scripts' / name, '07_source_changes/modified_files/scripts/' + name)
    for f in (R / 'adapters/scout_adapter').glob('*.py'):
        put(f, '07_source_changes/modified_files/adapters/scout_adapter/' + f.name)
    for f in (R / 'vendor/TrackEval/trackeval').rglob('*.py'):
        put(f, '07_source_changes/modified_files/vendor/TrackEval/trackeval/' + str(f.relative_to(R / 'vendor/TrackEval/trackeval')))
    put(R / 'vendor/TrackEval/LICENSE', '07_source_changes/modified_files/vendor/TrackEval/LICENSE')
    put(O / 'summary/code_changes.patch', '07_source_changes/code_changes.patch')
    for run in RUNS[1:4]:
        for name in ['configuration.json', 'summary.json', 'source_verification.json', 'solver.jsonl',
                     'PoseTracker.patched.py', 'native_changes.patch', 'trace_runner.executed.py']:
            put(O / run / name, f'07_source_changes/configs/{run}/{name}')
        for name in ['state_history.csv', 'feature_bank_updates.npz', 'global_predictions.csv']:
            put(O / run / name, f'05_association/new_runs/{run}/{name}')
        put(O / (run + '-retry.log'), f'08_reproducibility/execution_logs/{run}.log')
        put(O / (run + '_failed_bulk_logging/failure_record.json'), f'08_reproducibility/execution_logs/{run}_initial_failure.json')
    for name in ['configuration.json', 'summary.json', 'source_verification.json', 'native_changes.patch']:
        put(OLD / 'C_30s' / name, '07_source_changes/configs/original_C_30s/' + name)
    # Preserve the legacy full-frame output separately from the fair shared-mask comparison.
    put(OLD / 'C_30s/global_predictions.csv', '02_tracking_results/legacy_original/global_predictions_undistorted.csv')
    put(OLD / 'C_30s/predictions_raw.csv', '02_tracking_results/legacy_original/predictions_raw.csv')
    for name in ['verification.json', 'input_sha256.json', 'dependency_versions.txt']:
        put(O / name, '08_reproducibility/' + name)
    put(R / 'sources.lock.json', '08_reproducibility/sources.lock.json')
    put(R / 'weights/posetrack/manifest.json', '08_reproducibility/previous_weights_verification.json')
    tree(O / 'aligned_input/calibration', '08_reproducibility/aligned_calibration')
    raw = []
    for directory in [OLD / 'C_30s', O / 'B_timestamp', O / 'C_timestamp', O / 'C_height_abstain']:
        for name in ['events.json', 'affinity_pairs.csv', 'track_snapshots.json', 'mesh_events.json']:
            p = directory / name
            if p.is_file():
                raw.append(dict(server_path=str(p), bytes=p.stat().st_size, sha256=sha(p), included_in_zip=False))
    with (H / '08_reproducibility/raw_log_index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, ['server_path', 'bytes', 'sha256', 'included_in_zip']); w.writeheader(); w.writerows(raw)
    write('01_summary/changes_summary.md', '''# 수정 사항과 적용 범위
- 실제 timestamp와 calibration view_id를 연결하고 중복 영상을 제거해 10 FPS 공통 시점으로 대응했다. 새로운 신경망 관측이나 GT 보정 입력은 만들지 않았다.
- B/C는 같은 시간 매핑, 특징, 포즈, BBox, solver, 높이 판정 및 기존 NumPy 행 교환 수정 코드를 사용한다. Ground 항의 추적 연결·신규 초기화 사용 여부만 다르다. Ground 항 유무에 따른 기존 정규화는 유지했다.
- run-posetrack-time-study.py는 원본을 메모리에서 패치한다. 원본 저장소 파일은 수정하지 않았다. native_changes.patch는 원본→실행본, code_changes.patch는 이전 Mesh runner→이번 runner 비교이다.
- 추가 후보의 유일한 주요 변경은 유한한 3D 관절의 수직 지면 높이가 미확정일 때 높이 판정을 유보한 것이다. 비정상 관절 좌표와 확인된 높이 경계는 그대로 검사한다. 이 후보는 성능 저하로 채택하지 않았다.
- 초기 세 실행은 추적 종료 후 대용량 JSON 일괄 저장 중 exit 137로 실패했다. 확정된 커널 OOM 로그는 없으므로 메모리 부족은 추정이다. 레코드 단위 저장으로 변경해 모두 재실행했고 300개 프레임별 결과 개수·상태 집계가 동일했다.
- 함수별 실제 계산과 조건은 patched source, geometry module, trace runner 및 사례 CSV를 함께 확인한다. Geometry 수정은 별도 module과 runner에 있으므로 native patch 하나만으로 전체 변경을 설명할 수 없다.
''')
    write('01_summary/remaining_issues.md', '''# 남은 문제
- 공유 Global IDF1 50, AssA 60 목표는 달성하지 못했다. 시간 정렬을 정확도 개선으로 표현하지 않는다.
- Mesh 실패 2059건의 직접 조건은 표면 교차 없음이다. 경계 상자 교차만으로 지면 구멍, 카메라 오차, 검출 발 위치 중 물리 원인을 확정할 수 없다.
- 원본 Missing 재활성화 14회는 과거 GT 이력 충돌 1회, 신뢰할 판정 불가 13회이다. 1회도 즉시 직전 GT가 아닌 147프레임 전 이력 비교임을 유지한다. 정상 복구 시각화와 전체 복구 정확도는 제시하지 않았다.
- ID28/38 최초 GT 불일치는 사후 IoU 대응으로 확인한 시점이다. 최초 실제 추적 오류라는 단정은 하지 않는다.
- BBox·GT는 원본 왜곡 영상 좌표로 평가했다. shared IDF1은 이 실험의 카메라/시간 분리 프로토콜이며 공식 SCOUT 3D 지표가 아니다.
- 4개 시스템 비교, 60초 실험 및 독립적인 물리 보정 검증은 이번 결과에 포함하지 않는다.
- 전체 영상·가중치·신경망 캐시·대용량 전체 로그는 ZIP에서 제외했다. 이미지 사례와 관련 수치 로그, 결과 CSV, 평가 GT와 코드는 포함했다. 전체 로그 위치/해시는 08_reproducibility/raw_log_index.csv 참조.
''')
    write('03_evaluation/evaluation_protocol.md', '''# 공통 평가 조건
Sequence 1 / cam_0~cam_7 / 원본 277~576 / 10 FPS 공통 시점 300개.
frame_mapping.csv의 selected 2359개 카메라·시간 슬롯과 동일 GT 51500행을 모든 주 실험에 적용한다. 이용 불가 41슬롯은 GT도 생성하지 않는다. 영상은 존재하지만 검출이 없는 슬롯의 GT는 유지하여 FN으로 센다.
공통 CSV의 x/y/width/height는 원본 왜곡 영상 픽셀 좌표이며 왜곡 보정 BBox와 혼합하지 않는다. source_frame_id는 실제 파일 번호이고 common_time_index는 0부터 시작한다. local_id 공란은 별도 Local ID 미노출이다.
HOTA/DetA/AssA: TrackEval IoU 0.05~0.95 평균, 카메라별 시퀀스 결합. IDF1 2D/IDSW: IoU>=0.5, 카메라별 평가 결합. 공유 Global IDF1은 전 카메라 공유 ID namespace와 카메라·시점별 관측 비교를 사용한다. 공식 SCOUT multi-view/3D 평가와 구분한다.
모든 지표 JSON과 CSV의 비율은 0~1이며 PDF/START_HERE 표는 0~100이다. IDSW는 정수이다.
기존 original_shared는 기존 추적 결과를 동일 유효 슬롯에 사후 대응한 결과이다. 기존 추적 내부 상태는 당시 전체 입력을 이용했으므로 시간 정렬 효과를 포함한다. 기존 legacy GT 52713행 점수와 직접 비교하지 않는다.
B_same_output/C_same_output은 두 실험이 공통 출력한 28169개 관측만 평가한 사후 진단이다. 정식 전체 평가를 대체하지 않는다.
이번 결과는 같은 PoseTrack 신경망 입력을 이용한 association 비교다. 다른 검출·Re-ID를 사용하는 모델 전체 비교는 별도 실험으로 표시한다. 공통 CSV 평가 API는 임의 결과의 좌표계·ID 의미를 자동 추론하지 않는다. normalize 변환기의 입력 조건과 모델별 매핑을 먼저 확인한다.
''')
    write('04_geometry/mesh_error_analysis.md', '''# Mesh 진단
기존 30초 37023관측 중 BBox 하단 ray 표면 교차 실패 2059건. cam_0~7: 9,151,57,420,565,10,627,220. 모든 실패 ray는 mesh 전체 경계 상자를 통과하지만 유효 삼각형 교차는 없다. 물리 원인을 더 세분화할 증거는 부족하다.
BBox 실패 중 발목 ray 성공은 7건뿐이었다. 임의의 전면 발목 대체나 거리 임계값 완화는 하지 않았다. 실제 fallback/neutral 처리와 좌표 방향은 동봉 posetrack-mesh-geometry.py 및 mesh_intersections_all.csv에서 확인한다.
교차 실패와 미출력은 별개다. 미출력 ledger는 각 관측의 직접 탈락 원인 하나를 기록하며 AND 초기화 조건에서는 최초 실패 조건을 집계한다. 이 숫자는 사람 수가 아니다.
높이 미확정 유보 후보에서 높이 보정 제거는 108→51건으로 줄었지만 다른 후속 연결 변화로 출력은 29226→29161, 공유 IDF1은 47.18→46.60으로 낮아졌다. 후보는 보존하고 기본 결과에 채택하지 않았다.
''')
    write('07_source_changes/code_condition_index.md', '''# 계산과 조건을 확인할 코드 위치
아래 줄 번호는 동봉 configs/C_timestamp/PoseTracker.patched.py 기준이다. 원본 저장소와의 차이는 같은 폴더 native_changes.patch에 있다. 기존 ID28/38 분석은 original_C_30s 실행 로그이며 별도 표시한다.

| 항목 | 함수·줄 | 실제 조건 / 관련 파일 |
|---|---|---|
| State-aware Re-ID | compute_reid_aff, 425~452 | 검출 confidence<0.88이면 비활성; 해당 카메라 상태 이력에 Occluded/Missing이 있을 때 Bank의 양수 유사도 최대값에서 0.5를 빼 As를 계산. 활성 가중치 1, 비활성 1e-5 |
| BBox Affinity | compute_bboxiou_aff, 509~553 | age<=15와 검출 점수 및 IoU-0.5를 사용 |
| Geometry | compute_epi_homo_aff, 506 | modified_files/scripts/posetrack-mesh-geometry.py의 MeshGeometry.affinity로 위임. 거리·validity·가중합 조건은 이 함수 참조 |
| 최종 Affinity | mv_update_wo_pred, 708~718 | Geometry 하한 적용, (a_epi*Aepi+a_box*Ab+a_homo*Ah+a_reid*As*w)/(1+w), norm 보정 후 음수 0 |
| Hungarian | mv_update_wo_pred, 720·753 | linear_sum_assignment(-aff_final); 선택되더라도 최종 점수<=0이면 업데이트 거절 |
| Occluded/Missing | mv_update_wo_pred, 729~754 | 배정되지 않은 대상의 최대 IoU>0.5이면 Occluded, 아니면 Missing; 단일 뷰 갱신의 겹침 보정은 single_view_2D_update(240) 참조 |
| 신규 초기화 | target_init, 609~661 | 2*Aepi+Ah로 클러스터링; seed confidence>0.9, main joints>0.5, 기존 대상 IoU>0.15 개수<1, 검출 overlap>0.3 개수<2 |
| Track 상태 | mv_update_wo_pred, 780~785 | Unconfirmed이며 갱신 없음→Deleted; update_age>=15→Missing |
| 최종 출력 | output, 812~825 | Confirmed이고 update_age==1인 대상의 valid_views만 출력 |
| Missing 복구 | match_with_miss_tracks, 576~606 | 기존/new Bank 최대 내적, 후보 중 최대값>0.5이면 reactivate. 이 경로는 Geometry 또는 Hungarian을 복구 수락 조건으로 사용하지 않음 |
| Feature Bank | multi_view_init(218), multi_view_3D_update(364) | upper-body pose>0.5, bbox score>0.9, IoU/overlap 조건 통과 후 기존 Bank와 유사도 조건을 검사. 정확한 갱신은 feature_bank_events.csv 및 실제 npz로 대조 |
| 높이 판정 | MeshGeometry.invalid_height, geometry module 51 | h 미확정 또는 h<-1 또는 h>(발목1, 기타2.5)m이면 invalid. 후보는 run-posetrack-time-study.py 64의 함수에서 유한 관절의 h 미확정만 유보 |

global_id_28/38_analysis.csv는 출력의 GT 대응, affinity_examples.csv는 최초 GT 충돌 시 모든 후보 점수, original_case_matrices는 전체 후보 행렬, review_observation_embeddings 및 review_track_snapshots는 실제 입력 특징과 Bank 대조 자료이다. GT는 위 수락 조건에 사용하지 않았다.
''')
    write('07_source_changes/run_commands.md', '''# 검토 및 실행
## ZIP만으로 공통 CSV 재평가
압축을 푼 폴더에서 실행한다. 검증 환경은 Python 3.9, NumPy 1.23.5, SciPy 1.10.1이다. TrackEval 소스와 라이선스를 동봉했다. 의존성은 별도 환경에 준비한다.

```bash
python 07_source_changes/modified_files/scripts/evaluate-posetrack-time-study.py \
  --predictions 02_tracking_results/final_recommended_30s.csv \
  --frame-map 01_summary/frame_mapping.csv \
  --gt 03_evaluation/shared_gt.json \
  --out review_metrics --config offline_review
```

다른 모델이 기존 scout_adapter 표준 CSV를 출력한 경우 아래 변환 후 같은 평가기에 넣는다. kind=prediction, image_space=raw, id_scope=sequence가 필수이며 frame은 원본 파일 번호, native_track_id는 모델의 Global ID여야 한다. 카메라별 ID를 임의로 전역 ID로 취급하면 안 된다.

```bash
python 07_source_changes/modified_files/scripts/normalize-scout-common-evaluation.py \
  --input other_model_canonical.csv --frame-map 01_summary/frame_mapping.csv \
  --output other_model_common.csv --model UWIPL --config team_setting_01
```

## 현재 서버에서 Tracking 재현
전체 데이터와 기존 neural cache는 ZIP에 없다. 현재 서버 프로젝트에서 기존 이미지를 사용한다. 새 출력 경로가 필요하며 기존 폴더는 덮어쓰지 않는다.

```bash
cd /media/aimlab/HDD00/users/junseo-capstone/scout-mtmc
bash scripts/replay-posetrack-time-study.sh ground-off outputs/replay_ground_off_new
bash scripts/replay-posetrack-time-study.sh mesh outputs/replay_mesh_new
bash scripts/replay-posetrack-time-study.sh height-abstain outputs/replay_height_candidate_new
```

PROJECT_ROOT, DATA_ROOT, STUDY_ROOT, INPUT_ROOT, IMAGE_NAME으로 경로·이미지를 지정할 수 있다. Docker 컨테이너 CPU 추적만 실행하며 신경망 추론은 기존 캐시를 재사용한다. runtime 패치 파일과 실제 실행 trace runner는 configs에 보존했다. GT는 추적 후 평가 단계에만 읽는다.
source 코드의 일부 분석 스크립트는 당시 /project 및 /out 경로를 사용한다. ZIP 단독 재평가 명령 외의 전체 분석/추적 재현은 이 서버의 데이터·캐시와 컨테이너 마운트가 필요하다.
''')
    write('08_reproducibility/environment.md', f'''# 실행 환경
현재 원래 서버의 기존 Docker 이미지 scout-mtmc/posetrack-all:ba727e6 사용. H200 MIG 환경으로 이동하지 않았다.
이미지 ID sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d.
PoseTrack 원본 commit ba727e6bc6e45e44cd6e006b73e00864892173c2. 원본 소스는 보존하고 실행 시 메모리 패치를 적용했다.
이번 Tracking은 --runtime=runc, --network=none, NVIDIA_VISIBLE_DEVICES=void, CPU 3개, 메모리 16 GiB, OPENBLAS/OMP 1 thread로 실행했다. HiGHS solver는 전 300프레임 optimal 상태를 기록했다.
Python/라이브러리 전체 버전은 dependency_versions.txt. 가중치 manifest는 기존 신경망 추론 당시 검증 기록이며 이번에 새로 추론하거나 다운로드하지 않았다.
실험 원본: {O}
이전 보존 결과: {OLD}
원본 전체 로그: raw_log_index.csv. ZIP은 검토용 핵심 사례를 포함하며 영상 전체·가중치·full_input30 특징 캐시는 포함하지 않는다.
''')
    print('Prepared', H, flush=True)

def seal():
    verification = json.loads((O / 'verification.json').read_text())
    assert verification.get('offline_bundle_metric_equivalence', {}).get('passed') is True
    put(O / 'verification.json', '08_reproducibility/verification.json')
    for name in ['metrics.json', 'protocol.json', 'per_camera_metrics.csv']:
        put(O / 'offline_eval_verification' / name, '08_reproducibility/offline_evaluation/' + name)
    put(Path(__file__), '07_source_changes/modified_files/scripts/' + Path(__file__).name)
    files = sorted(f for f in H.rglob('*') if f.is_file() and f.name not in ['FILES.csv', 'SHA256SUMS.txt'])
    manifest = [dict(file=str(f.relative_to(H)), bytes=f.stat().st_size, sha256=sha(f)) for f in files]
    with (H / 'FILES.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, ['file', 'bytes', 'sha256']); w.writeheader(); w.writerows(manifest)
    write('SHA256SUMS.txt', '\n'.join(f"{m['sha256']}  {m['file']}" for m in manifest))
    target = R / 'deliverables/PoseTrack_SCOUT_30s_followup_20261106.zip'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError('Preserving existing final ZIP: ' + str(target))
    partial = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in sorted(H.rglob('*')):
            if f.is_file():
                z.write(f, str(f.relative_to(H)))
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names))
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
        for m in manifest:
            assert hashlib.sha256(z.read(m['file'])).hexdigest() == m['sha256'], m['file']
    partial.rename(target)
    fingerprint = sha(target)
    target.with_suffix('.zip.sha256').write_text(fingerprint + '  ' + target.name + '\n')
    result = dict(path=str(target), bytes=target.stat().st_size, files=len(names),
                  sha256=fingerprint, CRC='PASS', all_manifest_hashes='PASS')
    (O / 'package_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'seal'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else seal()
