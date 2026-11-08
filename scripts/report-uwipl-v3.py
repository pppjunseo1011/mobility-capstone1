#!/usr/bin/env python3
"""Summarize completed V3 experiments and assemble a review-only handoff."""
import argparse,base64,collections,csv,hashlib,html,json,shutil,zipfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v3-20261107';H=O/'handoff';TITLE='UWIPL_SCOUT_V3_단계별검증_20261107'
def load(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def collect():
 rows=[]
 for prefix in ['experiments','oracles','holdout']:
  for p in sorted((O/prefix).rglob('metrics.json')):
   if not (p.parent/'summary.json').exists():continue
   rel=p.parent.relative_to(O);m=load(p)['summary'];cfg=load(p.parent/'summary.json');audit=load(p.parent/'anchor_gt_audit.json')
   m=dict(m,anchor_gt_majority_covered_people=len({a['majority_gt'] for a in audit['per_anchor'] if a['majority_gt'] is not None}))
   rows.append(dict(scope=prefix,experiment=str(rel),**cfg['config'],**m))
 rows.sort(key=lambda r:({'experiments':0,'oracles':1,'holdout':2}[r['scope']],(0 if r['experiment'].split('/')[1]=='V2' else int(r['experiment'].split('/')[1][1:])) if r['scope']=='experiments' else r['experiment'].split('/')[1],r['experiment'].split('/')[-2],r['seconds']))
 csvout(O/'experiment_comparison.csv',rows);csvout(O/'experiment_comparison_excel.csv',rows)
 with (O/'experiment_comparison_excel.csv').open('r',encoding='utf-8') as f:txt=f.read()
 (O/'experiment_comparison_excel.csv').write_text(txt,encoding='utf-8-sig')
 save(O/'experiment_comparison.json',rows);return rows
def verify(rows):
 assert len([r for r in rows if r['scope']=='experiments'])==44
 assert len([r for r in rows if r['scope']=='oracles'])==4
 assert len([r for r in rows if r['scope']=='holdout'])==6
 hist=load(O/'selection_history.json')['history'];prev=None
 for h in hist:
  allowed={'V2':set(BASE),'E1':{'samples'},'E2':{'score'},'E3':{'distance'},'E4':{'vote'},'E5':{'new','high'}}[h['family']]
  if prev:
   for r in h['ranking']:
    dif={k for k in prev if r['config'][k]!=prev[k]};assert dif<=allowed and len(dif)<=1
  ranked=sorted(h['ranking'],key=lambda r:(-r['mean_IDF1_view_aware'],-r['mean_HOTA_2D'],r['sum_IDSW'],r['config']!=prev));assert ranked[0]['config']==h['selected_config'];prev=h['selected_config']
 for row in rows:
  p=O/row['experiment'];conf=load(p/'summary.json');pred=load(p/('final_predictions.json' if (p/'final_predictions.json').exists() else 'anchor_predictions.json'));assert row['raw_rows']==len(pred)
  assert all(np.isfinite(r['bbox']).all() and min(r['bbox'][2:])>0 for r in pred)
  if not (p/'final_predictions.json').exists():assert conf['eligible_rows']==len(pred)+conf['hungarian_unassigned']
  assert row['rows_evaluated']+row['duplicate_id_rows_removed_for_eval']==len(pred)
  if row['scope']=='experiments':assert conf['GT_used'] is False
  if row['scope']=='oracles':assert conf['GT_used'] is True and conf['evaluation_only'] is True
  assert 0<=row['IDF1_view_aware']<=1 and 0<=row['HOTA_2D']<=1
  assert conf['fps']==10
 final=load(O/'final_selection.json');assert load(O/'holdout/frozen_configuration.json')['config']==final['config'];assert sorted(final['ranking'],key=lambda r:(-r['mean_IDF1_view_aware'],-r['mean_HOTA_2D'],r['sum_IDSW'],r['label']!='off'))[0]['label']==final['stcra'];holdout=load(O/'holdout/input_verification.json');assert holdout['max_skew_seconds']<=.05
 assert min(v[0] for v in holdout['source_ranges'].values())>576
 for p in [O/'pose_new/sct_keypoints.npy',O/'holdout/pose/sct_keypoints.npy']:
  a=np.load(p);assert a.ndim==3 and a.shape[1:]==(17,3) and np.isfinite(a).all()
 verification=dict(status='PASS',development_metric_rows=44,oracle_metric_rows=4,holdout_metric_rows=6,one_variable_per_candidate=True,sequential_selection_verified=True,v2_equivalence=load(O/'v2_equivalence.json'),finite_output_boxes=True,output_count_and_duplicate_accounting=True,GT_oracles_separated=True,holdout_source_disjoint=True,holdout_frozen_selection=True,holdout_max_skew_seconds=holdout['max_skew_seconds'],pose_shapes_and_finite=True)
 save(O/'verification.json',verification)
 return verification
BASE=dict(area=1000,distance=80,score=.95,samples='11',vote=5,new=.7,high=.6)
def table(rows,fields):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for k,label,fmt in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k,label,fmt in fields)+'</tr>' for r in rows)+'</tbody></table>'
def build():
 rows=collect();verification=verify(rows);rep=O/'report';rep.mkdir(exist_ok=True)
 history=load(O/'selection_history.json');final=load(O/'final_selection.json');hold=load(O/'holdout/input_verification.json')
 selection=[dict(stage=h['family'],**h['ranking'][0]) for h in history['history']];selection.append(dict(stage='E6',**final['ranking'][0]));save(O/'selected_path.json',selection)
 selectrows=[dict(stage=r['stage'],choice=r['label'],view=100*r['mean_IDF1_view_aware'],hota=100*r['mean_HOTA_2D'],idsw=r['sum_IDSW']) for r in selection]
 fields=[('experiment','실험',lambda s:str(s).replace('experiments/','').replace('holdout/','별도/').replace('oracles/','GT/')),('seconds','초',str),('anchors','앵커',str),('anchor_gt_covered_people','GT 커버',str),('HOTA_2D','HOTA',lambda x:f'{100*x:.2f}'),('DetA_2D','DetA',lambda x:f'{100*x:.2f}'),('AssA_2D','AssA',lambda x:f'{100*x:.2f}'),('IDF1_2D','IDF1 2D',lambda x:f'{100*x:.2f}'),('IDF1_view_aware','IDF1 view',lambda x:f'{100*x:.2f}'),('IDSW_2D','IDSW',str)]
 chosen=[r for r in rows if r['experiment'].startswith(('experiments/V2/','experiments/E5/high_0.4_only/','experiments/E6/4_3_2/'))]
 oracles=[r for r in rows if r['scope']=='oracles'];holdrows=[r for r in rows if r['scope']=='holdout']
 with (O/'stage_gt_recall.csv').open() as f:recall=list(csv.DictReader(f))
 recfields=[('seconds','초',str),('stage','단계',str),('rows','출력 박스',str),('gt_boxes','GT 박스',str),('matched_gt_boxes','맞춘 GT',str),('recall','재현율',lambda x:f'{100*float(x):.2f}%')]
 person=[]
 for s in [5,30]:
  a=load(O/f'diagnostics/s{s}/protocol.json');rs=load(O/f'diagnostics/s{s}/uncovered_gt_people.json');person.append(dict(seconds=s,gt=a['anchor_covered_gt']+a['uncovered_gt'],member=a['anchor_covered_gt'],majority=a['majority_covered_gt'],absent=sum(not r['appears_at_anchor_times'] for r in rs),mixed_only=a['anchor_covered_gt']-a['majority_covered_gt']))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams['font.family']='DejaVu Sans'
 fig,axes=plt.subplots(1,2,figsize=(10,3.3))
 for sec,ax in zip([5,30],axes):
  rr=[r for r in chosen if r['seconds']==sec];xx=np.arange(len(rr));ax.bar(xx-.17,[100*r['IDF1_view_aware'] for r in rr],.34,label='Global IDF1');ax.bar(xx+.17,[100*r['HOTA_2D'] for r in rr],.34,label='HOTA');ax.set_xticks(xx,['V2','E5','E6'] if [r['experiment'].split('/')[1] for r in rr]==['V2','E5','E6'] else [r['experiment'].split('/')[1] for r in rr]);ax.set_ylim(0,65);ax.set_title(f'{sec}s development');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(rep/'metric_tradeoff.png',dpi=160);plt.close(fig)
 image='data:image/png;base64,'+base64.b64encode((rep/'metric_tradeoff.png').read_bytes()).decode()
 pages=[]
 pages.append('<h1>UWIPL · SCOUT V3 실험 결과</h1><p class="lead">V2 재현 → 남은 손실 진단 → E1~E6 순차 선정 → 별도 시간 구간 검증</p><p>Sequence 1, cam_0~cam_7, 10 FPS. 개발 구간은 원본 277~326(5초), 277~576(30초). 두 구간은 중첩된 개발 데이터이며 독립 테스트가 아니다.</p><h2>선정 결과</h2><p>넓이 <b>1,000</b>, 거리 <b>80</b>, 앵커 <b>11시점</b>, 점수 <b>&gt;0.95</b> 유지. 다수결 <b>15개 관측</b>, new_track_thresh <b>0.7</b>, track_high_thresh <b>0.4</b>. E6 선정 거리 스케줄은 <b>4·3·2</b>.</p><p class="warn">선정 기준은 5초·30초 평균 IDF1 view이다. STCRA는 이 점수를 높였지만 HOTA를 낮추고 IDSW를 크게 늘렸다. 모든 지표의 최선 설정 또는 안정적인 개선으로 해석하지 않는다.</p>'+table(chosen,fields)+'<img src="'+image+'"><p>표 비율은 0~100. CSV/JSON은 0~1. GT 커버는 앵커 구성원으로 대응한 인물 수이며 대표 인물 수와 구분한다.</p>')
 pages.append('<h1>진단 A/B · 손실이 발생한 위치</h1>'+table(person,[('seconds','초',str),('gt','GT 인물',str),('member','구성원 커버',str),('majority','대표 GT 커버',str),('mixed_only','구성원에만 존재',str),('absent','대표 미포착 중 11시점에 부재',str)])+'<p>GT가 어느 앵커의 구성원에 포함됐는지와 대표 GT로 남았는지를 함께 기록했다. 대표 미포착 인물마다 각 앵커 시점의 등장 여부, IoU≥0.5 검출 존재, 넓이/점수/두 조건 탈락을 CSV·JSON으로 제공한다. 동일 인물도 시점마다 원인이 달라 단일 원인으로 강제 분류하지 않았다.</p>'+table(recall,recfields)+'<p>YOLOX 원본은 저장된 post-NMS·objectness&gt;0.1·영상 경계 clipping 및 정수화 이후 박스이다. 네트워크 logits 전체가 아니다. 실제 Re-ID/추적 입력에는 기존 objectness≥0.3 필터가 있다. 재현율은 GT 박스 기준 camera/time별 1:1 IoU≥0.5 대응이다.</p><p>앵커 배정 이후 다수결은 같은 카메라·프레임에 중복 Global ID를 만들 수 있다. 지표 계산 시 기존 평가기처럼 높은 점수 한 행을 남긴다. 원본 결과와 duplicate_removals.json을 모두 보존하며, 제거 이후 재현율도 따로 표시했다.</p>')
 pages.append('<h1>순차 실험과 선정 규칙</h1>'+table(selectrows,[('stage','단계',str),('choice','선정',str),('view','평균 IDF1 view',lambda x:f'{x:.2f}'),('hota','평균 HOTA',lambda x:f'{x:.2f}'),('idsw','합산 IDSW',str)])+'<p>매 단계는 직전 선정 설정을 기준으로 조건 하나만 변경했다. 5초·30초 평균 IDF1 view → 평균 HOTA → 합산 IDSW가 작은 순으로 선정하고 완전 동률은 변경하지 않은 설정을 우선했다. E5는 E4 기준에서 new 0.5 또는 high 0.4를 각각 단독 변경했으며 두 값을 동시에 낮추지 않았다.</p><p>11/25/50은 구간 양 끝을 포함하는 linspace 정수 인덱스이다. 매 1초는 0,10,20,… 프레임(5초 5시점, 30초 30시점)이다. 다수결은 연속 시간 간격이 아니라 원본 알고리즘의 트랙별 배정 관측 행 묶음이며, 마지막 짧은 묶음은 직전 선택을 따른다.</p><h2>E1~E3</h2>'+table([r for r in rows if r['scope']=='experiments' and r['experiment'].split('/')[1] in ['E1','E2','E3']],fields))
 pages.append('<h1>E4~E6 · 연결 안정성과 공간 보정</h1>'+table([r for r in rows if r['scope']=='experiments' and r['experiment'].split('/')[1] in ['E4','E5','E6']],fields)+'<p>E6는 STCRA 별도 코드의 3회 반복이며 confidence는 0.65/0.70/0.75로 고정했다. 보간은 실행하지 않았다. 기존 SCT 포즈와 정확히 같은 박스만 재사용하고 새 박스 8,513개에는 HRNet-W48을 추가 적용했다.</p><p>거리 2.5·2·1.5와 4·3·2는 SCOUT world 좌표를 미터로 취급한 후보이다. AIC 지도 픽셀과의 확정 환산값이 아니다. 발 좌표는 왜곡 보정 후 H⁻¹로 투영했지만 Z=0 단일 지면 평면 가정은 유지했다. 비평면 Mesh를 쓰는 PoseTrack 기하와 동일하지 않다.</p><p>new_track_thresh는 고신뢰 검출 분리 뒤 신규 생성 조건으로 적용된다. high=0.6을 유지한 new=0.5 실험이 모든 score 0.5 박스를 새 추적으로 받는 것은 아니다. 코드의 scores&gt;track_high_thresh 및 track.score&lt;new_track_thresh 조건을 함께 확인해야 한다.</p>')
 pages.append('<h1>평가 전용 GT 진단과 별도 구간</h1><h2>GT를 사용한 진단 — 정상 추론에서 제외</h2>'+table(oracles,fields)+'<p>GT_grouped_candidate_anchors: 기존 유효 신경망 앵커 후보를 GT 인물별로 묶고 미대응 후보는 제외했다. 후보에 없는 GT 인물까지 추가한 완전한 GT 앵커는 아니다. 실제 임베딩과 Hungarian·다수결은 유지했다.</p><p>두 진단은 E5 설정의 앵커 단계까지 실행했으며 STCRA는 제외했다. GT_boxes: 실제 GT 박스를 검출 대신 넣고 score=1로 설정한 뒤 원본 영상 crop에서 OSNet 5종 특징을 다시 추출했다. GT ID는 tracker에 전달하지 않았다. 영상 밖 유효 면적이 없는 3개 박스는 입력에서 제외했지만 평가 GT에는 유지했다. 둘 다 진단용 참조이며 수학적인 성능 상한을 보장하지 않는다.</p><h2>설정 고정 후 별도 구간 검증</h2>'+table(holdrows,fields)+f'<p>카메라별 원본 약 2807~3106 구간, 공통 300시점. timestamp만으로 구간을 선택했고 개발 구간과 중복되지 않는다. 2,389/2,400 슬롯을 확보했으며 최대 시각 차이는 {hold["max_skew_seconds"]*1000:.1f}ms이다. 누락 11슬롯은 관측·GT를 만들지 않는다. 비교를 위해 V2도 같은 구간에서 실행했다. 모든 선정 값은 이 구간 평가 전에 고정했으며 재튜닝하지 않았다.</p>')
 pages.append('<h1>프로토콜·검증·전달 범위</h1><p>개발 실험은 V2와 같은 원본 프레임 대응을 유지했다. 이 30초 구간에는 기존에 확인된 최대 2.051초 카메라 timestamp 차이가 남는다. 따라서 E6 공간 보정의 개발 결과에는 비동기 관측 영향이 포함될 수 있다. 별도 구간은 실제 timestamp를 0.1초 grid에 ±25ms로 대응하고 유효 관측 간 차이≤50ms를 확인했다. 이는 독립적인 카메라 하드웨어 동기화 인증은 아니다.</p><p>HOTA·DetA·AssA는 TrackEval의 2D IoU 0.05~0.95 평가, IDF1 2D·IDSW는 카메라별 시간 순서의 IoU 0.5 평가이다. IDF1 view는 카메라별 관측을 유지한 공유 Global ID 평가이며 공식 SCOUT 3D benchmark와 구분한다.</p><p>V2 구성원·ID·BBox·평가지표가 기존 결과와 일치했다. float32 배정 비용에는 1e-6 미만 반올림 차이가 있어 수치 오차로 기록했다. 모든 단계의 단일 변수 변경, 입력·출력 수, 중복 제거 수, 유한한 BBox·포즈와 별도 구간 고정 설정을 검사했다.</p><h2>ZIP 구성</h2><ul><li>요약 PDF/HTML, 전체 54행 성능표, 순차 선정 내역</li><li>GT 인물별 진단, 단계별 GT 재현율</li><li>실험별 설정·앵커 GT 분포·쪼개짐/섞임·미배정·결과 CSV·중복 제거 기록</li><li>GT 평가 전용 실험, 별도 구간 검증 및 timestamp 매핑</li><li>실행 코드·원본 소스 커밋·가중치 해시·로그·검증 기록</li></ul><p>원본 데이터 전체, 가중치와 대용량 신경망 특징 캐시는 제외한다. 재실행에 필요한 캐시는 서버 outputs/uwipl-v3-20261107 및 기존 UWIPL 폴더에 보존했다. 원본 모델 저장소·V2 실험·Docker 환경은 변경하지 않았다. 이번 실행은 기존 TITAN RTX 서버에서 수행했으며 MIG 서버 이관 후 검증과 별개다.</p>')
 selection_page,experiment_table=pages[2].split('<h2>E1~E3</h2>')
 pages[2]=selection_page+'<p>E1~E3의 전체 결과는 다음 쪽에 제시한다. 후보를 늘리거나 임계값을 낮춘다고 성능이 단조롭게 개선되지 않았다. 이번 특징 공간의 Ward 거리 기준과 샘플 수가 함께 작용하므로, 이번 개발 구간에서 선정한 값을 다른 데이터의 일반적인 최적값으로 해석하지 않는다.</p>'
 pages.insert(3,'<h1>E1~E3 · 전체 비교 결과</h1>'+experiment_table)
 style='''@page{size:A4;margin:14mm}body{font-family:"Noto Sans CJK KR",sans-serif;color:#203345;font-size:9.4pt;line-height:1.6}section{break-after:page}section:last-child{break-after:auto}h1{font-size:20pt;color:#174d66;margin:5mm 0}h2{font-size:12pt;margin:4mm 0 2mm}table{border-collapse:collapse;width:100%;font-size:7.7pt;margin:3mm 0}th{background:#23566b;color:white}td,th{padding:1.4mm 1mm;border-bottom:1px solid #ccd8df;text-align:right}td:first-child,th:first-child{text-align:left;overflow-wrap:anywhere}tr:nth-child(even){background:#f1f5f7}img{width:100%;max-height:67mm;object-fit:contain}.warn{border-left:4px solid #c27d2c;padding:3mm;background:#fff7e9}.lead{font-size:12pt}.folio{font-size:8pt;color:#587484;border-bottom:1px solid #587484;padding-bottom:2mm}p{margin:2.8mm 0}'''
 pagehtml='<!DOCTYPE html><html lang="ko"><meta charset="UTF-8"><title>UWIPL SCOUT V3</title><style>'+style+'</style><body>'+''.join(f'<section><div class="folio">UWIPL × SCOUT · 2026.11.07 · {i+1}/{len(pages)}</div>{s}</section>' for i,s in enumerate(pages))+'</body></html>'
 (rep/(TITLE+'.html')).write_text(pagehtml)
 readme=f'''# UWIPL SCOUT V3 — 2026-11-07

보고서 PDF부터 확인하세요. 44개 개발 평가 + 4개 GT 진단 평가 + 6개 별도 구간 평가를 수행했습니다.

최선값 선정 기준은 5초·30초 평균 IDF1 view입니다. 선정 설정: {final['config']}, STCRA={final['stcra']}.
STCRA는 공유 IDF1을 높였지만 HOTA와 IDSW는 악화했습니다. STCRA off 결과를 반드시 같이 검토하세요.

00_report: 보고서. 01_summary: 전체 지표·선정·검증. 02_diagnostics: 인물별 누락 및 단계 재현율.
03_experiments: V2/E1~E6 설정·앵커·GT 분포·예측·평가. 04_oracles: GT 사용 진단 전용.
05_holdout: 개발 구간과 겹치지 않는 timestamp 정렬 구간 결과. 06_source: 실행 소스.
07_reproducibility: 가중치/소스 해시·입력 검증·실행 기록.

비율 CSV/JSON은 0~1, PDF는 0~100입니다. 단계별 recall은 사람 수가 아니라 GT 박스 수 기준입니다.
앵커 구성원 GT 커버와 대표 GT 커버를 구분하며, 쪼개짐은 같은 GT가 여러 앵커에 존재하는 수, 섞임은 하나의 앵커에 여러 GT가 존재하는 수입니다.
앵커 score는 기존 입력의 objectness(b[4])이며 class score와 곱한 값이 아니다.
다수결은 트랙별 관측 행 단위이며 누락 프레임이 있으면 고정된 초 단위 구간과 다릅니다.
동일 camera/frame/ID 중복은 기존 평가 규칙에 따라 높은 점수 한 행을 남기고 제거 내역을 보존합니다.

개발 30초는 기존 V2의 시각 불일치를 유지했습니다. 별도 구간은 제공 timestamp 기준 최대42ms, 누락11/2400슬롯입니다.
STCRA H는 왜곡 보정 영상과 Z=0 단일 평면 가정이며 비평면 지면의 정확한 물리 보정을 보장하지 않습니다.
GT_boxes와 GT_grouped_candidate_anchors는 GT를 입력에 사용한 진단입니다. 정상 성능표와 합치지 마세요.

서버 원본: {O}
신규 입력/특징/포즈는 해당 폴더에 보존했습니다. ZIP에는 영상 전체·가중치·대용량 feature npy를 넣지 않았습니다.
4개 시스템 최종 비교나 새 MIG 서버 실행 검증은 수행 범위에 포함되지 않습니다.
'''
 (rep/'START_HERE.md').write_text(readme);print('REPORT HTML created',flush=True)

def package():
 assert load(O/'verification.json')['status']=='PASS';H.mkdir(exist_ok=True)
 def put(p,q):
  q=H/q;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 def tree(p,q,allow=None):
  for f in sorted(p.rglob('*')):
   if f.is_file() and (allow is None or f.name in allow):put(f,Path(q)/f.relative_to(p))
 put(O/'report/START_HERE.md','START_HERE.md');tree(O/'report','00_report')
 for name in ['experiment_comparison.csv','experiment_comparison_excel.csv','experiment_comparison.json','selected_path.json','selection_history.json','final_selection.json','verification.json','v2_equivalence.json','stage_gt_recall.csv']:
  put(O/name,'01_summary/'+name)
 tree(O/'diagnostics','02_diagnostics')
 names={'summary.json','metrics.json','anchor_gt_audit.json','anchor_members.json','hungarian_unassigned.json','predictions.csv','gt_matching_results.csv','duplicate_removals.json','rounds.json','events.json','v2_equivalence.json'}
 tree(O/'experiments','03_experiments',names);tree(O/'oracles','04_oracles',names|{'protocol.json'});tree(O/'holdout','05_holdout',names|{'frame_mapping.csv','input_verification.json','frozen_configuration.json'})
 # Final raw predictions, in addition to compact common CSVs.
 for group in ['E5/high_0.4_only','E6/4_3_2']:
  for sec in [5,30]:
   p=O/f'experiments/{group}/s{sec}';name='final_predictions.json' if group.startswith('E6') else 'anchor_predictions.json';put(p/name,f'03_experiments/{group}/s{sec}/{name}')
 for sec in [5,30]:put(O/f'holdout/final/s{sec}/final_predictions.json',f'05_holdout/final/s{sec}/final_predictions.json')
 sources=['uwipl-v3.py','prepare-uwipl-v3-holdout.py','prepare-uwipl-v3-oracle.py','select-uwipl-v3-holdout.py','report-uwipl-v3.py','uwipl-study-pose.py','scout-sample-stage.py','evaluate-uwipl-study.py','uwipl-anchor-sweep.py']
 for name in sources:put(R/'scripts'/name,'06_source/scripts/'+name)
 put(O/'run_stcra_scout.py','06_source/run_stcra_scout.py');put(O/'oracles/GT_boxes/oracle-reid-runner.py','06_source/oracle-reid-runner.py')
 tree(R/'vendor/TrackEval/trackeval','06_source/vendor/TrackEval/trackeval',{'hota.py','clear.py','identity.py','_base_metric.py','_timing.py','utils.py','__init__.py'})
 put(R/'vendor/TrackEval/LICENSE','06_source/vendor/TrackEval/LICENSE');put(R/'sources.lock.json','07_reproducibility/sources.lock.json')
 put(R/'weights/uwipl/SHA256SUMS','07_reproducibility/weights_SHA256SUMS');put(R/'weights/uwipl/manifest.json','07_reproducibility/weights_manifest.json')
 for name in ['holdout_selection.json','pose_reuse.json','stcra_homographies.json']:put(O/name,'07_reproducibility/'+name)
 for name in ['tune.log','tune-resume.log','diagnostics-v2.log','e6.log','holdout-track.log','holdout-finish.log','oracle-boxes.log']:
  if (O/name).exists():put(O/name,'07_reproducibility/logs/'+name)
 for p in [O/'pose_new/pose_report.json',O/'holdout/pose/pose_report.json',O/'holdout/inference/uwipl/detection/report.json',O/'holdout/inference/uwipl/reid/report.json',O/'oracles/GT_boxes/inference/uwipl/reid/report.json']:
  put(p,'07_reproducibility/inference_reports/'+str(p.relative_to(O)))
 tree(O/'reproducibility','07_reproducibility')
 fs=sorted(p for p in H.rglob('*') if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt']);manifest=[dict(file=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)) for p in fs]
 csvout(H/'FILES.csv',manifest);(H/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in manifest))
 target=R/'deliverables/UWIPL_SCOUT_V3_20261107.zip';assert not target.exists();partial=target.with_suffix('.zip.partial')
 with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(H.rglob('*')):
   if f.is_file():z.write(f,str(f.relative_to(H)))
 with zipfile.ZipFile(partial) as z:
  assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()))
  for m in manifest:assert hashlib.sha256(z.read(m['file'])).hexdigest()==m['sha256']
 partial.rename(target);digest=sha(target);target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n')
 for ext in ['html','pdf']:shutil.copy2(O/'report'/(TITLE+'.'+ext),R/'docs/handoff'/(TITLE+'.'+ext))
 result=dict(path=str(target),bytes=target.stat().st_size,files=len(manifest)+2,sha256=digest,CRC='PASS',all_file_hashes='PASS');save(O/'package_verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','package']);a=p.parse_args();globals()[a.mode]()
