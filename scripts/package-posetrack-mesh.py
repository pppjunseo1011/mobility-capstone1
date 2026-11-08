#!/usr/bin/env python3
"""Package final research outputs with file hashes and ZIP integrity validation."""
import csv,hashlib,io,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-mesh-20261104';P=O/'reproducibility';D=R/'deliverables'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 D.mkdir(exist_ok=True)
 for run in ['B_final','C_final','C_30s']:
  assert json.loads((O/run/'summary.json').read_text())['status']=='PASS'
  assert (O/run/'metrics.json').is_file()
 sources=sorted(set(R.glob('scripts/*posetrack-mesh*'))|{R/'scripts/merge-posetrack-mesh-tail.py',R/'scripts/trace-posetrack-flat.py',R/'scripts/verify-posetrack-flat.py',R/'scripts/evaluate-uwipl-study.py',R/'scripts/scout-sample-stage.py',R/'sources.lock.json'}|set(R.glob('adapters/scout_adapter/*.py')))
 hashes={}
 for f in sources:
  if not f.is_file():continue
  rel=f.relative_to(R);dest=P/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[str(rel)]=digest(f)
 (P/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 files={}
 def add(src,name):
  assert src.is_file(),src
  assert name not in files,name
  files[name]=src
 def tree(src,prefix,exclude=()):
  for f in sorted(src.rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.name not in exclude:add(f,prefix+'/'+str(f.relative_to(src)))
 for ext in ['pdf','html']:add(R/f'docs/handoff/PoseTrack_SCOUT_전체영역_Mesh_검증_20261104.{ext}',f'00_report/PoseTrack_SCOUT_전체영역_Mesh_검증_20261104.{ext}')
 add(O/'START_HERE.md','00_report/START_HERE.md')
 for name in ['experiment_comparison.csv','per_camera_comparison.csv','findings.json','execution_audit.json','geometry_verification.json','affinity_formula_verification.json','C_solver_equivalence.json','scout_to_posetrack_camera_map.json','selected_ranges.json','timestamp_audit.csv','visual_examples.json']:
  add(O/name,'01_summary/'+name)
 for name in ['geometry_observations.csv','largest_geometry_discrepancies.csv']:add(O/name,'02_geometry/'+name)
 for window in ['full_input','full_input30']:
  for name in ['observations.json','preparation_summary.json','calibration_audit.json','input_sha256.json']:add(O/window/name,f'02_geometry/{window}/{name}')
  tree(O/window/'calibration',f'02_geometry/{window}/calibration')
 for cam in range(8):add(R/f'datasets/scout/extracted/scout/calibrations/cam_{cam}.json',f'02_geometry/original_calibration/cam_{cam}.json')
 for run in ['B_final','C_final','C_30s']:
  tree(O/run,'03_results/'+run,exclude=['geometry_details.json'])
  add(O/(run+'.log'),'03_results/'+run+'/run.log')
 tree(O/'visualizations','04_visualizations')
 tree(P,'05_reproducibility')
 add(R/'weights/posetrack/manifest.json','06_provenance/posetrack_weights_manifest_previous_verification.json')
 add(R/'outputs/osc-analysis-20261101/inference30/yachiyo/detection/report.json','06_provenance/reused_tail_yolox_report.json')
 for stage in ['detection','pose','reid']:
  add(R/f'outputs/posetrack-transfer-20261101/inference/posetrack/{stage}/report.json',f'06_provenance/original_5s_{stage}_report.json')
 for stage in ['pose','reid']:
  add(O/f'inference_s30_tail/posetrack/{stage}/report.json',f'06_provenance/tail_{stage}_report.json')
  add(O/f'inference_s30_tail/posetrack/{stage}/run.log',f'06_provenance/tail_{stage}_run.log')
 add(O/'inference_s30_tail/input/manifest.json','06_provenance/tail_input_manifest.json')
 tree(O/'A_replay','06_provenance/A_replay')
 for name in ['global_predictions.csv','summary.json','plane.json','verification.json']:
  f=R/'outputs/posetrack-flat-20261101'/name
  if f.is_file():add(f,'06_provenance/A_original_flat/'+name)
 readme='''PoseTrack SCOUT 전체 영역 / Mesh 검증 — 2026-11-04

먼저 00_report의 PDF를 열어 주세요. 자세한 설명은 START_HERE.md에 있습니다.
01_summary: A/B/C 및 30초 비교, 구간·카메라 매핑, 시간 정렬 및 수치 검증
02_geometry: 교차점·실패 상태·보정값·입력 관측 매핑(특징 벡터 제외)
03_results: B_final / C_final / C_30s의 원본 결과, 평가, 전체 내부 분석 로그
04_visualizations: 카메라별 예측, 확대 ID 연결/혼합 사례, 지면 차이
05_reproducibility: 코드, 패치 참조, 해시, 실행 명령 및 재실행 스크립트
06_provenance: A 재현 및 추가 신경망 실행 증거/기존 가중치 해시 기록

predictions_raw.csv = 원본 영상 BBox. global_predictions.csv = 왜곡 보정 BBox.
source_frame은 SCOUT 파일 번호, model_frame은 0부터 시작하는 구간 내부 번호.
local_id 공란은 미구현/미노출을 뜻하며 0으로 채우지 않았습니다.
World XY NaN / world_method=0은 위치 미확정입니다. 2D 추적 행은 유지합니다.
30초는 카메라 시간 차이 최대 2.051초가 있어 정렬 후 추가 검증이 필요합니다.
GT는 평가와 사후 그림 선정에만 사용했고 추적·기하 입력에 넣지 않았습니다.

재실행용 전체 설치 묶음은 아닙니다. 데이터셋 전체·가중치·Docker 이미지·
대용량 입력 특징 캐시는 제외했습니다. 현재 서버의 기존 프로젝트를 이용하면
05_reproducibility/source/scripts/replay-posetrack-mesh.sh로 재실행할 수 있습니다.

FILES.csv / SHA256SUMS.txt는 압축 내부 파일의 내용 검증용입니다.
실패·초기 진단 실험은 현재 서버에 보존했으며 최종 묶음에는 성공한 최종 결과만
담았습니다. B 초기 GLPK 중지와 NumPy 교환 오류는 보고서 및 패치에 설명합니다.
'''
 target=D/'PoseTrack_SCOUT_full_mesh_20261104.zip';partial=target.with_suffix('.zip.partial');manifest=[]
 with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  z.writestr('README_전달파일.txt',readme)
  manifest.append({'file':'README_전달파일.txt','bytes':len(readme.encode()),'sha256':hashlib.sha256(readme.encode()).hexdigest()})
  for i,(name,src) in enumerate(sorted(files.items())):
   h=hashlib.sha256()
   with src.open('rb') as f,z.open(name,'w',force_zip64=True) as dest:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);dest.write(block)
   manifest.append({'file':name,'bytes':src.stat().st_size,'sha256':h.hexdigest()})
   if i%100==0:print('PACK',i,'/',len(files),name,flush=True)
  out=io.StringIO();w=csv.DictWriter(out,['file','bytes','sha256']);w.writeheader();w.writerows(manifest)
  z.writestr('FILES.csv',out.getvalue());z.writestr('SHA256SUMS.txt',''.join(x['sha256']+'  '+x['file']+'\n' for x in manifest))
 partial.replace(target)
 print('Checking all ZIP CRCs...',flush=True)
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  assert len(z.namelist())==len(set(z.namelist()))
  assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
  entry_count=len(z.namelist())
 checksum=digest(target);target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n')
 result={'archive':str(target.relative_to(R)),'bytes':target.stat().st_size,'files':entry_count,'sha256':checksum,'CRC':'PASS','duplicate_paths':False,'absolute_or_parent_paths':False,'manifest_files':len(manifest),'contains_weights_or_full_dataset':False}
 (O/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
