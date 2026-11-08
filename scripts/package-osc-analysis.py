#!/usr/bin/env python3
"""Bundle diagnostic evidence, controlled results and source changes for review."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

R=Path(__file__).resolve().parents[1];A=R/'outputs/osc-analysis-20261101'
NAME='OSC_SCOUT_analysis_20261101';D=R/'deliverables';D.mkdir(exist_ok=True)
summary=json.loads((A/'summary.json').read_text());assert len(summary['experiments'])==5
files=set()
def add(p):
    if p.is_symlink():raise RuntimeError('Unexpected symlink '+str(p))
    if p.is_dir():
        for c in p.iterdir():
            if c.name!='__pycache__' and not c.name.startswith('attempt_'):add(c)
    else:
        assert p.is_file(),p
        files.add(p)

for name in ('forensics5','figures','provenance','summary.json','experiment_comparison.csv','per_camera_comparison.csv',
             'tracklet_gate_comparison.csv','timestamp_skew.csv','code_findings.csv'):add(A/name)
for r in summary['experiments']:
    p=A/('s%d_th%d'%(r['duration_nominal_seconds'],r['short_track_th']))
    for name in ('tracking','evaluation','validation_scope.json','scout_to_osc_camera_map.json','track.log','evaluate.log'):add(p/name)
    for name in ('prepare.log','prepare_report.json','world_gt_comparison.json'):
        if (p/name).exists():add(p/name)
for name in ('selection.json','summary.json','input/manifest.json'):add(A/'inference30'/name)
for stage in ('detection','pose','reid'):
    for name in ('report.json','run.log'):add(A/'inference30/yachiyo'/stage/name)
for suffix in ('pdf','html'):add(R/('docs/handoff/OSC_SCOUT_원인분석_비교실험_20261101.'+suffix))

guide='''OSC 원인 분석 및 조건별 비교 — 2026-11-01
=========================================
이 ZIP은 팀 내 결과 검토용이며 다른 PC 재실행용 설치 패키지가 아닙니다.

먼저 scout-mtmc/docs/handoff/OSC_SCOUT_원인분석_비교실험_20261101.pdf 를 읽으세요.
HTML은 동일 내용의 브라우저 열람용입니다.

아래 자료의 기준 경로: scout-mtmc/outputs/osc-analysis-20261101/

1) experiment_comparison.csv: 5초 th=10/20/30/40 및 약 30초 th=40 결과.
   IDF1은 0~1 비율입니다. HOTA/DetA/AssA 빈칸은 미산출이며 0점이 아닙니다.
   Local IDSW는 카메라별 합이며 Global IDF1은 view-aware 통합 ID 대응입니다.
2) per_camera_comparison.csv: 각 설정의 카메라별 Local IDF1·IDSW 및 Global 출력.
3) tracklet_gate_comparison.csv: 각 설정·카메라·Local ID의 길이/Pose 탈락 이유.
4) forensics5/: 기존 5초 th=40에 대한 원인 분석.
   local_stage_counts.csv / local_stage_metrics.csv: raw→NMS→warp 단계 비교.
   tracklets.csv: tracklet별 길이, span, 구간, Pose 조건, 교차 탈락 사유.
   global_row_exclusions.csv: Global 미할당 4,953행과 할당 후 삭제 4행.
   idsw_events.csv: 실제 motmetrics의 ID 전환 사건.
   adjacent_gt_appearance.csv: IoU 기반 연속 GT 대응의 외형 유사도 진단.
   failure_*.jpg: 카메라별 대표 IDSW 사례 2개씩. 위/아래는 전후 시점.
   postfilter_pose_scores.json: 삭제된 4행의 17개 Keypoint confidence.
   summary.json: 카메라별 분포, 교차 조건과 원인 집계.
5) figures/: tracklet 길이, 외형 gate, 임계값 비교, timestamp 그래프와
   cam5_pose_length.png / postfilter_pose.png의 확대 Pose 사례.
6) s5_th10/, s5_th20/, s5_th30/, s5_th40/, s30_th40/:
   tracking/repeat_01, repeat_02: raw 및 fixed Local/Global JSON, 설정, 시간.
   tracking/repeat_report.json: 동일 조건 반복 결과의 최종 해시.
   evaluation/local_predictions.csv: 공통 Local CSV.
   evaluation/global_common/predictions.csv: 공통 Global CSV.
   evaluation/ground_truth.csv: 해당 구간 GT.
   evaluation/metrics_*.json: 프로토콜별 지표, 평가 보고서와 일치성 검사.
   evaluation/*_events.csv: motmetrics 사건별 원본 로그.
   evaluation/visualizations/: 각 설정 Local/Global 시각화.
   validation_scope.json / scout_to_osc_camera_map.json: 구간과 번호 대응.
7) summary.json / timestamp_skew.csv: 약 30초 구간의 실제 시간·불연속·편차.
8) provenance/:
   code_findings.csv는 기준 폴더 아래에 있으며 문제별 파일·행·함수·조건을 기록합니다.
   runner_changes.patch: 이전 실행 코드 대비 변경점.
   original_scpt.py / original_mcpt.py / original_tracking.py / original_pose.py:
     원본 모델 코드 사본(수정하지 않음). PDF의 행 번호는 이 파일과 대응합니다.
   그 외 py: 분석·실험·평가·보고서 생성 코드 사본.
   evidence.json: 원본 모델 커밋, 가중치·코드 해시.

유의사항
- 원본 모델의 인원 수 상한은 없고 objectness >=0.3은 유지했습니다.
- 5초에서는 short_track_th만 변경했습니다. Local 결과는 네 설정에서 동일합니다.
- 30초는 th=40으로 실행했고 5초 구간을 포함합니다. 독립 검증셋이 아닙니다.
- 약 30초 구간에 timestamp 불연속과 최대 2.051초 카메라 편차가 있습니다.
  World 진단은 시간 동기화가 검증된 성능으로 사용하지 마세요.
- 공간 보정·Global 재할당 OFF인 개발 실험입니다. 공식 SCOUT 벤치마크 결과가 아닙니다.
- CSV·JSON의 /work 또는 서버 경로는 당시 실행 기록이며 로컬 PC용 경로가 아닙니다.
- 전체 입력 영상·임베딩/포즈 입력 묶음·가중치·Docker 이미지는 기존 서버에 보존하며
  이 ZIP에는 결과 검토용 자료만 담았습니다.
- SCOUT 일부 이미지·GT가 포함되어 있으므로 공개 재배포는 이용 조건을 확인해야 합니다.

FILES.csv와 SHA256SUMS로 파일 목록·크기·SHA-256을 확인할 수 있습니다.
압축을 푼 최상위 폴더에서 sha256sum -c SHA256SUMS 명령으로 검사 가능합니다.
'''
archive=D/(NAME+'.zip');partial=archive.with_suffix('.zip.partial');hashes=[];listing=[]
with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):
        rel='scout-mtmc/'+p.relative_to(R).as_posix();data=p.read_bytes();sha=hashlib.sha256(data).hexdigest()
        z.writestr(NAME+'/'+rel,data);hashes.append((sha,rel));listing.append((rel,len(data),sha))
    f=io.StringIO();w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(listing)
    for rel,data in [('START_HERE.txt',guide.encode()),('FILES.csv',f.getvalue().encode())]:
        z.writestr(NAME+'/'+rel,data);hashes.append((hashlib.sha256(data).hexdigest(),rel))
    z.writestr(NAME+'/SHA256SUMS',''.join(f'{h}  {p}\n' for h,p in hashes))
with zipfile.ZipFile(partial) as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(hashes)+1
    for h,p in hashes:assert hashlib.sha256(z.read(NAME+'/'+p)).hexdigest()==h,p
partial.replace(archive);h=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(f'{h}  {archive.name}\n')
(R/'docs/handoff/OSC_ANALYSIS_START_HERE.txt').write_text(guide)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(hashes)+1,'sha256':h,'verified':True},indent=2))
