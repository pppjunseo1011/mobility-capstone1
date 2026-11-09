#!/usr/bin/env python3
"""Report cached and new config-only UWIPL trials; no model code changes."""
import argparse,csv,hashlib,html,json,shutil,subprocess,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v4-config-20261109';V4=R/'outputs/uwipl-v4-20261108';TITLE='UWIPL_SCOUT_V4_설정조합_재평가_20261109';REP=O/'report'
NAMES=['baseline','vote_30','vote_40','vote_50','stcra_on','stcra_postvote15','vote30_stcra432','vote30_stcra432_post15','vote30_stcra432_post30']
LABELS=dict(baseline='기준: 다수결15',vote_30='다수결30',vote_40='다수결40',vote_50='다수결50',stcra_on='15 + STCRA',stcra_postvote15='15 + STCRA + 후15',vote30_stcra432='30 + STCRA',vote30_stcra432_post15='30 + STCRA + 후15',vote30_stcra432_post30='30 + STCRA + 후30')
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def source(scope,name,sec):return (V4 if name in ['baseline','vote_30','stcra_on','stcra_postvote15'] else O)/'experiments'/scope/name/f's{sec}'
def collect():
 assert read(O/'state.json')['status']=='COMPLETE';rows=[];times=[]
 for scope in ['development','holdout']:
  for name in NAMES:
   for sec in [5,30]:
    p=source(scope,name,sec);m=read(p/'metrics.json');s=m['summary'];cfg=read(p/'summary.json');cl=m['detail']['combined_2D']['CLEAR'];audit=read(p/'anchor_gt_audit.json')
    rows.append(dict(scope=scope,variant=name,**s,MOTA=cl['MOTA'],Recall=cl['CLR_Re'],Precision=cl['CLR_Pr'],anchor_gt_majority_covered_people=len({a['majority_gt'] for a in audit['per_anchor'] if a['majority_gt'] is not None}),config_vote=cfg['config']['vote'],postvote=cfg.get('postvote',False),postvote_changed_rows=cfg.get('postvote_changed_rows',0),reused=str(p).startswith(str(V4)),source=str(p)))
 for p in sorted((O/'logs').glob('*.json')):t=read(p);times.append(dict(job=p.stem,seconds=t['seconds'],returncode=t['returncode']))
 rr=[read(p)['summary'] for p in sorted((O/'reevaluation').glob('*/*/s*/*/metrics.json'))];assert len(rows)==36 and len(rr)==108
 csvout(O/'experiment_comparison.csv',rows);save(O/'experiment_comparison.json',rows);csvout(O/'reevaluation_comparison.csv',rr);save(O/'reevaluation_comparison.json',rr);csvout(O/'timing.csv',times)
 return rows,rr

def verify():
 rows,rr=collect();protected=read(O/'unchanged_code.json')
 for p,h in protected.items():assert sha(R/p)==h
 for scope in ['development','holdout']:
  assert read(O/'validation'/scope/'voting_parity.json')['status']=='PASS';assert read(O/'validation'/scope/'reevaluation.json')['status']=='PASS'
  for sec in [5,30]:assert read(O/'experiments'/scope/'vote15_stcra432'/f's{sec}'/'parity.json')['status']=='PASS'
 # Assignment identities may change; row, bbox, source frame and pose association must stay valid.
 for r in rows:
  p=Path(r['source']);original=read(V4/'experiments'/r['scope']/'baseline'/f"s{r['seconds']}"/'anchor_predictions.json');lookup={(x['camera'],x['frame'],x['local_id']):x for x in original};pred=read(p/('final_predictions.json' if (p/'final_predictions.json').exists() else 'anchor_predictions.json'))
  for x in pred:
   y=lookup[x['camera'],x['frame'],x['local_id']];assert x['bbox']==y['bbox'] and x['source_frame']==y['source_frame'] and x['raw_anchor_id']==y['raw_anchor_id']
  assert not read(p/'summary.json')['GT_used']
 provenance=dict(protected_sha256=protected,new_scripts={p.name:sha(p) for p in (R/'scripts').glob('*uwipl-v4-config*.py')},upstream_commit=subprocess.check_output(['git','-C',str(R/'repos/uwipl'),'rev-parse','HEAD'],text=True).strip(),upstream_status=subprocess.check_output(['git','-C',str(R/'repos/uwipl'),'status','--porcelain'],text=True),input_manifests={scope:sha(R/('outputs/uwipl-study-20261101/inference30/input/manifest.json' if scope=='development' else 'outputs/uwipl-v3-20261107/holdout/inference/input/manifest.json')) for scope in ['development','holdout']},V4_original_metrics={str(p.relative_to(V4)):sha(p) for p in (V4/'experiments').glob('*/*/s*/metrics.json')},existing_code_frozen=True,neural_inference_rerun=False)
 save(O/'provenance.json',provenance);save(O/'verification.json',dict(status='PASS',comparison_rows=36,reevaluation_rows=108,new_config_metric_rows=20,reused_config_metric_rows=16,model_and_existing_scripts_unchanged=True,vote15_vote30_native_parity=True,STCRA_vote15_V4_parity=True,prediction_bbox_source_frame_raw_anchor_preserved=True,GT_used_in_prediction=False,neural_inference=False,HOTA_mean_invariant_to_Identity_CLEAR_threshold=True,tiling_executed=False));print('VERIFY PASS',flush=True)

def pct(x):return '—' if x is None else f'{x*100:.2f}'
def table(rs,cols):return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for k,h,f in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(f(r.get(k)))+'</td>' for k,h,f in cols)+'</tr>' for r in rs)+'</tbody></table>'
def build():
 rows,rr=collect();REP.mkdir(exist_ok=True)
 def sel(scope,sec):return [r for r in rows if r['scope']==scope and r['seconds']==sec]
 name=lambda x:LABELS[x]
 compact=[('variant','설정',name),('HOTA_2D','HOTA',pct),('IDF1_view_aware','IDF1 view',pct),('IDSW_2D','IDSW',str)]
 full=[('variant','설정',name),('HOTA_2D','HOTA',pct),('DetA_2D','DetA',pct),('AssA_2D','AssA',pct),('IDF1_2D','IDF1 2D',pct),('IDF1_view_aware','IDF1 view',pct),('IDSW_2D','IDSW',str)]
 pages=[]
 pages.append('<h1>UWIPL V4 설정 조합 및 재평가</h1><p class="lead">2026-11-09 · 모델 코드 수정 및 새 신경망 추론 없음</p><p>V4 baseline: 넓이&gt;1,000px², 거리80, 점수&gt;0.95, 앵커11시점, 다수결15, 검출800×1440. BoT-SORT new=.7/high=.4와 기존 검출·포즈·ReID·SCT를 유지했다. 개발/별도 각각5초·30초에서 비교했다.</p><h2>개발30초</h2>'+table(sel('development',30),compact)+'<h2>별도30초</h2>'+table(sel('holdout',30),compact)+'<p>다수결50은 두30초 구간에서 기준보다 HOTA가 높고 IDSW가 적었다. 별도30초의 IDF1 view는 다수결40이50보다 높다. STCRA 조합은 개발 IDF1 view를 높였지만 별도 HOTA는 기준보다 낮아, 모든 지표에서 우세한 조합으로 해석하지 않는다.</p><p class="warn">전체 영상+타일 병합, 카메라별 타일 선택, 타일 전용 신뢰도 기준은 현재 코드의 설정 옵션이 아니므로 실행하지 않았다. 기존 모델/실험 코드를 수정하지 않는 이번 범위에서 다음 단계로 보류했다.</p>')
 for scope,label in [('development','개발'),('holdout','별도')]:
  pages.append('<h1>'+label+' 구간 · 전체 추적 지표</h1><h2>5초</h2>'+table(sel(scope,5),full)+'<h2>30초</h2>'+table(sel(scope,30),full)+'<p>비율 표는0~100, CSV/JSON은0~1. IDSW는 이전 UWIPL과 동일한 TrackEval CLEAR, IoU0.5 기준이다. HOTA·DetA·AssA는8카메라의 독립2D 시퀀스를 결합하고 IoU0.05~0.95를 평균했다. IDF1 view는 공유 Global ID 공간에서 같은 camera/time의 박스만 대응한 값이다.</p><p>다수결 길이는 고정 초가 아니라 Local tracklet 안의 배정된 관측 개수다. 마지막 짧은 블록은 앞 블록의 결정을 유지하는 기존 규칙을 그대로 사용했다. 누락 관측이 있으면30개가 정확히3초를 뜻하지 않는다.</p>')
 protocols={'height_ge50_strict':'GT≥50 / 전체 예측 유지','height_ge50_ignore':'GT≥50 / 작은 GT 대응 제외','iou_0.3':'전체 GT / IoU0.3'}
 for scope,label in [('development','개발'),('holdout','별도')]:
  subset=[]
  for sec in [5,30]:
   base=next(r for r in rows if r['scope']==scope and r['variant']=='baseline' and r['seconds']==sec);subset.append(dict(base,protocol='original',HOTA_at_0_3=None))
   subset.extend(r for r in rr if r['scope']==scope and r['variant']=='baseline' and r['seconds']==sec)
  pcols=[('seconds','초',str),('protocol','평가 규약',lambda x:'원래 전체GT/IoU0.5' if x=='original' else protocols[x]),('HOTA_2D','표준 HOTA',pct),('IDF1_2D','IDF1 2D',pct),('IDF1_view_aware','IDF1 view',pct),('IDSW_2D','IDSW',str)]
  variants=[r for r in rr if r['scope']==scope and r['seconds']==30 and r['protocol']=='iou_0.3']
  pages.append('<h1>'+label+' · V4 저장 결과 재평가</h1><p>새 추론 없이 V4의9조건×2길이×2구간을 재평가했다. 아래는 기준 설정의 프로토콜 비교이며 전체108행은 reevaluation_comparison.csv에 있다.</p>'+table(subset,pcols)+'<h2>V4의30초 결과 · IoU0.3</h2>'+table(variants,[('variant','V4 조건',str),('IDF1_2D','IDF1 2D',pct),('IDF1_view_aware','IDF1 view',pct),('IDSW_2D','IDSW',str),('HOTA_at_0_3','HOTA@0.3',pct)])+'<p>IoU0.3은 Identity/CLEAR의 대응 기준이다. 표준 HOTA는 여러 IoU를 평균하므로 원래 값과 같아야 하며 실제로 일치했다. HOTA@0.3은 곡선의 단일 지점이며 표준 HOTA와 다른 지표다.</p>')
 pages.append('<h1>높이 제한 평가의 정확한 의미</h1><p>높이50px는 각 GT 관측 박스의 높이 h≥50 조건이다. 동일 인물도 프레임에 따라 포함 여부가 달라질 수 있다. 예측 박스 자체를 높이50으로 잘라내는 실험이 아니다.</p><h2>1. Strict: 요청을 그대로 적용한 주 결과</h2><p>50px 미만 GT만 제외하고 예측은 모두 유지한다. 작은 사람에 대한 올바른 예측도 평가 GT에 대응되지 않으면 FP가 된다. 그러므로 높이가 큰 사람만 남겼다고 점수가 반드시 상승하지 않는다.</p><h2>2. Ignore: 작은 사람의 예측을 중립 처리한 보조 진단</h2><p>같은 camera/time에서 높이≥50 GT에 먼저 IoU≥0.5 최대 개수 일대일 대응을 한다. 남은 예측 중 작은 GT와 IoU≥0.5로 대응되는 것만 평가에서 제외한다. 그 밖의 미대응 예측은 FP 후보로 유지한다. 제외한 행과 대응 GT를 ignored_rows.json에 저장했다.</p><p>이 보조 규칙은 고정IoU0.5로 제외 목록을 만든 뒤 HOTA 곡선 전체에도 같은 목록을 적용한다. 공식 SCOUT의 크기별 평가 규약이라고 주장하지 않는다. 다른 규약과 섞어서 모델 개선 수치로 보고하면 안 된다.</p><p>동일 baseline의 개발30초 HOTA는 strict39.00 / ignore55.52, 별도30초는 strict40.10 / ignore53.74다. 이 차이는 모델이 달라진 효과가 아니라 평가 대상과 FP 처리 방식이 달라진 효과다.</p><h2>재평가 검증</h2><p>작은 GT 대응 예측의 제외, 큰 GT 우선 대응, strict에서 전체 예측 유지, IoU0.3/0.5에서 달라지는 ID 대응, 표준 HOTA 불변을 검사했다. 모델 GT 입력이나 파라미터 재선정은 없다.</p>')
 pages.append('<h1>실행 범위, 보류 사항 및 재현 자료</h1><p>새 설정 지표20행, 기존 결과 재사용16행, 합계36행. V4 재평가108행. 다수결15/30의 예측을 기존 결과와 비교해 정확히 일치함을 확인했고, STCRA 입력 다수결15를 다시 실행해 이전V4 결과와 정확히 일치함을 확인했다. 모델·기존 스크립트의 SHA-256도 전후 일치한다.</p><p>새로 추가한 것은 실행·평가·보고서용 스크립트뿐이다. 저장된 raw anchor 배정을 기존 다수결 규칙으로 처리하고 기존 run_stcra_scout.py와 포즈·H를 호출했다. STCRA는 거리4/3/2, confidence.65/.70/.75의3회 재할당이다. 새 학습·검출·포즈·ReID 추론, 보간, smooth_avg_tracks는 실행하지 않았다.</p><h2>타일링은 다음 단계</h2><p>현재 V4 타일 스크립트는 전체 영상 패스 없는2×2타일만 지원한다. 요청한 혼합은 전체/타일 출처 표시, 카메라별 분기, tile-only 점수 적용, 병합NMS, 새 검출 행과 임베딩의 재연결이 필요하다. 기존 코드 설정만으로 재현할 수 없어 이번 수치에는 포함하지 않았다.</p><h2>이전 한계 유지</h2><p>개발30초는 원본277~576로 최대 카메라 시각차2.051초가 남는다. 별도는V3에서 이미 사용한 timestamp 정렬 구간이며 새 블라인드 테스트가 아니다. 5초는30초 안에 포함된다. STCRA의 단일Z=0 평면·undistorted feet와 기존 좌표 단위 가정을 유지했으며 SCOUT 메시 공간 보정을 새로 검증하지 않았다.</p><h2>파일</h2><p>experiment_comparison.csv: 앵커 수, GT 커버·혼합·분절·미배정, IDF1/HOTA/IDSW 등 전체 지표. reevaluation_comparison.csv:108개 재평가 값. experiments: 전후 예측·GT audit·중복제거·STCRA 회차 기록. reevaluation: 상세 metric 및 ignored row. unchanged_code.json/provenance.json/verification.json: 코드 보존·입력·검증 근거.</p><pre>python3 scripts/run-uwipl-v4-config.py\npython3 scripts/report-uwipl-v4-config.py verify\npython3 scripts/report-uwipl-v4-config.py build</pre><p>현재 Docker 호스트·기존 캐시 경로용 명령이다. MIG 컨테이너 안에서 Docker를 중첩 실행하는 명령이 아니다. 전달 ZIP은 검토용이며 데이터셋·가중치·전체 특징 캐시를 포함하지 않는다.</p>')
 css='@page{size:A4;margin:16mm}body{font-family:"Noto Sans CJK KR",sans-serif;font-size:11px;line-height:1.65;color:#24394a}h1{font-size:22px;color:#173f5c;border-bottom:2px solid #27829f;padding-bottom:8px}h2{font-size:15px;margin-bottom:5px}.lead{font-size:14px}table{width:100%;border-collapse:collapse;font-size:9px;margin:9px 0}th,td{text-align:right;border-bottom:1px solid #d5e1e8;padding:5px 3px}th{background:#eaf2f7}th:first-child,td:first-child{text-align:left}section{break-after:page}section:last-child{break-after:auto}tr{break-inside:avoid}.warn{background:#fff0d0;padding:10px}pre{background:#eef4f7;padding:10px;white-space:pre-wrap}'
 (REP/(TITLE+'.html')).write_text('<!doctype html><meta charset="utf-8"><title>'+TITLE+'</title><style>'+css+'</style>'+''.join('<section>'+p+'</section>' for p in pages));(REP/'START_HERE.txt').write_text('UWIPL V4 설정 조합 및 재평가\nPDF: 결과와 프로토콜. experiment_comparison.csv:36행. reevaluation_comparison.csv:108행.\n모델 코드 및 기존 실험 코드는 수정하지 않았습니다. 타일 혼합은 보류했습니다.\n높이50 strict와ignore는 다른 평가 규약입니다. HOTA@0.3은 표준 HOTA와 다릅니다.\n');print('REPORT',len(pages),'pages',flush=True)

def package():
 H=O/'handoff';H.mkdir(exist_ok=True)
 def put(p,rel):q=H/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for p in REP.iterdir():
  if p.is_file():put(p,'01_report/'+p.name)
 for name in ['experiment_comparison.csv','reevaluation_comparison.csv','timing.csv','experiment_design.json','unchanged_code.json','verification.json','provenance.json','state.json']:put(O/name,'02_summary/'+name)
 for scope in ['development','holdout']:
  for name in NAMES:
   for sec in [5,30]:
    src=source(scope,name,sec)
    for p in src.iterdir():
     if p.is_file() and (p.name in ['summary.json','metrics.json','predictions.csv','anchor_gt_audit.json','duplicate_removals.json','hungarian_unassigned.json','rounds.json'] or p.name.endswith('_events.json')):put(p,f'03_experiments/{scope}/{name}/s{sec}/'+p.name)
 for folder in ['reevaluation','validation','logs']:
  for p in (O/folder).rglob('*'):
   if p.is_file() and p.suffix=='.json':put(p,'04_'+folder+'/'+str(p.relative_to(O/folder)))
 for p in (R/'scripts').glob('*uwipl-v4-config*.py'):put(p,'05_new_runners/'+p.name)
 put(R/'docs/uwipl-v4-config.md','01_report/README.md')
 files=[dict(file=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(H.rglob('*')) if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt']];csvout(H/'FILES.csv',files);(H/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in files))
 z=R/'deliverables'/(TITLE+'.zip');assert not z.exists()
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for p in H.rglob('*'):
   if p.is_file():archive.write(p,str(p.relative_to(H)))
 with zipfile.ZipFile(z) as archive:
  assert archive.testzip() is None
  for r in files:assert hashlib.sha256(archive.read(r['file'])).hexdigest()==r['sha256']
 save(O/'package_verification.json',dict(status='PASS',files=len(files)+2,bytes=z.stat().st_size,sha256=sha(z),CRC='PASS',member_hashes='PASS'));z.with_suffix('.zip.sha256').write_text(sha(z)+'  '+z.name+'\n')
 for ext in ['html','pdf']:shutil.copy2(REP/(TITLE+'.'+ext),R/'docs/handoff'/(TITLE+'.'+ext))
 print('PACKAGE',z,z.stat().st_size,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','build','package']);a=p.parse_args();globals()[a.action]()
