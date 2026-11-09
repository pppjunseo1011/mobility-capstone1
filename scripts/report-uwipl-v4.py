#!/usr/bin/env python3
"""Review report for V4, keeping detector diagnostics and tracking metrics separate."""
import argparse,base64,csv,hashlib,html,json,shutil,zipfile
from pathlib import Path
ORDER={v:i for i,v in enumerate(['baseline','D1_1088x1920','D1_1280x2304','D2_tiles2x2','D3_nms0.8','vote_20','vote_30','stcra_on','stcra_postvote15'])}
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v4-20261108';TITLE='UWIPL_SCOUT_V4_검출진단_20261108'
def load(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def collect():
 metrics=[];detection=[];times=[]
 for p in sorted((O/'experiments').glob('*/*/s*/metrics.json')):
  rel=p.parent.relative_to(O/'experiments');m=load(p)['summary'];config=load(p.parent/'summary.json');audit=load(p.parent/'anchor_gt_audit.json')
  metrics.append(dict(scope=rel.parts[0],variant=rel.parts[1],**m,anchor_gt_majority_covered_people=len({a['majority_gt'] for a in audit['per_anchor'] if a['majority_gt'] is not None}),**config['config'],stcra=rel.parts[1].startswith('stcra'),postvote=config.get('postvote',False),postvote_changed_rows=config.get('postvote_changed_rows',0)))
 for p in sorted((O/'diagnostics').glob('*/*/detection_metrics.csv')):
  with p.open() as f:
   for row in csv.DictReader(f):
    for k in ['seconds','camera','gt_boxes','pred_boxes','true_positives']:row[k]=int(row[k]) if row[k] else None
    for k in ['recall','precision','iou']:row[k]=float(row[k]) if row[k] else None
    if row['precision'] is not None and row['recall'] is not None:row['F1']=2*row['precision']*row['recall']/(row['precision']+row['recall']) if row['precision']+row['recall'] else 0
    else:row['F1']=None
    detection.append(row)
 # Emit zero-denominator groups explicitly instead of silently omitting them.
 present={(r['scope'],r['variant'],r['seconds'],r['stage'],r['camera'],r['gt_height']) for r in detection}
 for scope,variant in sorted({(r['scope'],r['variant']) for r in detection}):
  for sec in [5,30]:
   for stage in ['detections','selected']:
    for cam in range(9):
     for band in ['all','lt50','50to100','ge100']:
      if (scope,variant,sec,stage,cam,band) not in present:
       detection.append(dict(scope=scope,variant=variant,seconds=sec,stage=stage,camera=cam,scout_camera='all' if cam==0 else f'cam_{cam-1}',gt_height=band,gt_boxes=0,pred_boxes=0 if band=='all' else None,true_positives=0,recall=None,precision=None,iou=.5,F1=None))
 detection.sort(key=lambda r:(r['scope'],ORDER[r['variant']],r['seconds'],r['stage'],r['camera'],['all','lt50','50to100','ge100'].index(r['gt_height'])))
 metrics.sort(key=lambda r:(r['scope'],ORDER[r['variant']],r['seconds']))
 for p in sorted((O/'logs').glob('*.status.json')):
  s=load(p);times.append(dict(job=p.name[:-12],state=s['state'],elapsed_seconds=s.get('elapsed_seconds'),returncode=s.get('returncode')))
 csvout(O/'experiment_comparison.csv',metrics);csvout(O/'detection_comparison.csv',detection);csvout(O/'timing.csv',times)
 save(O/'experiment_comparison.json',metrics);return metrics,detection,times

def verify(metrics,detection,times):
 assert len(metrics)==36,len(metrics)
 assert len(detection)==1440,len(detection)
 groups={}
 for row in detection:groups.setdefault((row['scope'],row['variant'],row['seconds'],row['stage'],row['camera']),{})[row['gt_height']]=row
 for g in groups.values():
  for key in ['gt_boxes','true_positives']:assert sum(g[b][key] for b in ['lt50','50to100','ge100'])==g['all'][key]
 assert all(t['state']=='PASS' for t in times)
 input_maps={}
 for scope in ['development','holdout']:
  source=R/('outputs/uwipl-study-20261101/inference30' if scope=='development' else 'outputs/uwipl-v3-20261107/holdout/inference')
  input_maps[scope]={(int(f['camera'][4:])+1,f['index']):f['source_frame'] for f in load(source/'input/manifest.json')['frames']}
 assert load(O/'validation/baseline_parity.json')['status']=='PASS'
 assert load(O/'validation/detection_matching.json')['status']=='PASS'
 assert load(O/'validation/training_loader.json')['status']=='PASS'
 for r in metrics:
  p=O/'experiments'/r['scope']/r['variant']/f's{r["seconds"]}';conf=load(p/'summary.json');pred=load(p/('final_predictions.json' if (p/'final_predictions.json').exists() else 'anchor_predictions.json'))
  assert len(pred)==r['raw_rows']==r['rows_evaluated']+r['duplicate_id_rows_removed_for_eval']
  assert all(x['source_frame'] is not None and input_maps[r['scope']][x['camera'],x['frame']]==x['source_frame'] for x in pred)
  assert conf['fps']==10 and conf['config']==dict(area=1000,distance=80,score=.95,samples='11',vote=r['vote'],new=.7,high=.4)
  assert all(0<=r[k]<=1 for k in ['HOTA_2D','DetA_2D','AssA_2D','IDF1_2D','IDF1_view_aware'])
  assert all(len(x['bbox'])==4 and all(float('-inf')<v<float('inf') for v in x['bbox']) and min(x['bbox'][2:])>0 for x in pred)
  if r['variant']=='baseline':assert load(p/'v3_equivalence.json')['status']=='PASS'
 for scope in ['development','holdout']:
  source=R/('outputs/uwipl-study-20261101/inference30' if scope=='development' else 'outputs/uwipl-v3-20261107/holdout/inference')
  manifest=load(source/'input/manifest.json')['frames'];mkeys=[(f['key'],f['source_frame']) for f in manifest]
  for variant in ['D1_1088x1920','D1_1280x2304','D2_tiles2x2','D3_nms0.8']:
   root=O/'inference'/scope/variant;det=load(root/'uwipl/detection/detections.json');cfg=load(root/'uwipl/detection/report.json');reid=load(root/'uwipl/reid/report.json')
   assert [(f['key'],f['source_frame']) for f in det]==mkeys
   assert cfg['status']==reid['status']=='PASS' and not cfg['gt_used'] and not reid['gt_used']
   assert cfg['native_test_conf']==.001 and cfg['objectness_filter']==.1 and cfg['downstream_objectness_filter']==.3 and cfg['max_people_per_frame'] is None
   expected={'D1_1088x1920':([1088,1920],.7,False),'D1_1280x2304':([1280,2304],.7,False),'D2_tiles2x2':([800,1440],.7,True),'D3_nms0.8':([800,1440],.8,False)}[variant]
   assert (cfg['input_size'],cfg['nms'],cfg['tiles'])==expected
   assert all(e['shape']==[cfg['selected'],512] and e['finite'] and e['nonzero'] for e in reid['embeddings']) and len(reid['embeddings'])==5
 save(O/'verification.json',dict(status='PASS',tracking_metric_rows=len(metrics),detection_metric_rows=len(detection),one_condition_at_a_time=True,baseline_V3_metrics_exact=True,all_variants_same_frame_manifest=True,embedding_shapes_and_finite=True,no_people_cap=True,gt_not_used_in_normal_inference=True,D4='not trained; conditional after D1-D3/label review'))

def table(rows,fields):
 return '<table><thead><tr>'+''.join('<th>'+label+'</th>' for _,label,_ in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k)))+'</td>' for k,_,fmt in fields)+'</tr>' for r in rows)+'</tbody></table>'
def percent(x):return '-' if x is None else f'{100*float(x):.2f}'
def number(x):return '-' if x is None else str(x)
def build():
 m,d,t=collect();verify(m,d,t);rep=O/'report';rep.mkdir(exist_ok=True)
 detvars=['baseline','D1_1088x1920','D1_1280x2304','D2_tiles2x2','D3_nms0.8']
 ranking=[]
 for variant in detvars:
  rr=[r for r in m if r['scope']=='development' and r['variant']==variant];dr=[r for r in d if r['scope']=='development' and r['variant']==variant and r['stage']=='selected' and r['camera']==0 and r['gt_height']=='all']
  ranking.append(dict(variant=variant,mean_HOTA_2D=sum(r['HOTA_2D'] for r in rr)/2,mean_IDF1_view_aware=sum(r['IDF1_view_aware'] for r in rr)/2,mean_detection_recall=sum(r['recall'] for r in dr)/2,mean_detection_precision=sum(r['precision'] for r in dr)/2,mean_detection_F1=sum(r['F1'] for r in dr)/2))
 besth=max(ranking,key=lambda x:x['mean_HOTA_2D']);bestf=max(ranking,key=lambda x:x['mean_detection_F1'])
 save(O/'development_selection.json',dict(rule='Descriptive separate maxima of mean 5s/30s HOTA and detection F1; development only; holdout never used for selection.',best_mean_HOTA=besth,best_mean_detection_F1=bestf,ranking=ranking))
 fields=[('variant','조건',str),('seconds','초',str),('HOTA_2D','HOTA',percent),('DetA_2D','DetA',percent),('AssA_2D','AssA',percent),('IDF1_2D','IDF1 2D',percent),('IDF1_view_aware','IDF1 view',percent),('IDSW_2D','IDSW',number)]
 df=[('variant','조건',str),('seconds','초',str),('gt_boxes','GT',number),('pred_boxes','검출',number),('recall','재현율',percent),('precision','정밀도',percent),('F1','F1',percent)]
 pages=[]
 pages.append('<h1>UWIPL · SCOUT V4 검출 실험</h1><p class="lead">검출 크기·타일링·NMS 단일 조건 비교 + 다수결·STCRA 후처리 진단</p><p>기준: 넓이 &gt;1,000px², 거리 80, 앵커 점수 &gt;0.95, 11시점, 다수결 15개 관측, 10 FPS. BoT-SORT new=0.7/high=0.4는 V3 E5 선택값을 유지했다. 검출 실험은 STCRA OFF이며 새로운 검출 박스로 OSNet 5종 특징과 추적을 다시 계산했다.</p><p>개발: sequence_1/cam_0~7, source 277~326 및 277~576. 별도: 카메라별 약 2807~3106, timestamp 정렬 2,389/2,400 슬롯, 최대 시각차 42ms. 별도 구간은 V3에서 이미 사용한 구간이며 새로운 블라인드 테스트는 아니다. 5초는 각 30초의 앞부분이므로 독립 표본이 아니다. 개발 30초의 기존 최대 시각차 2.051초는 유지되어 다중 뷰 해석에 제한이 있다.</p><p class="warn">GT 박스 진단의 HOTA 46.68→76.84는 5초 결과이다. GT 박스·score=1을 함께 넣은 진단이므로 검출 재현율만의 효과를 분리한 값이나 실제 검출 성능의 보장 상한은 아니다. 화면과 어긋나 보이는 GT 투영 박스도 있어 미검출 원인을 전부 검출기로 단정하지 않는다.</p><h2>개발 결과에서 별도로 고른 후보</h2><p>평균 HOTA 기준: <b>'+besth['variant']+'</b>. 평균 검출 F1 기준: <b>'+bestf['variant']+'</b>. 별도 구간 점수로 설정을 다시 고르지 않았다.</p>'+table(ranking,[('variant','조건',str),('mean_HOTA_2D','평균 HOTA',percent),('mean_IDF1_view_aware','평균 IDF1 view',percent),('mean_detection_recall','평균 재현율',percent),('mean_detection_precision','평균 정밀도',percent)])+'<p>비율 표는 0~100, CSV/JSON은 0~1. HOTA·DetA·AssA는 카메라별 2D 평가의 결합값이며 IoU 0.05~0.95 평균이다. IDF1·IDSW의 기본 IoU는 0.5다. IDF1 view는 공유 Global ID 공간의 카메라/시간 관측 평가 규약이며 공식 SCOUT 최종 벤치마크 점수로 간주하지 않는다. baseline은 기존 V3와 4개 평가 구간의 지표가 정확히 일치한다.</p>')
 chart=rep/'detector_tracking_comparison.png'
 if chart.exists():
  pages.append('<h1>검출 개선과 추적 개선은 구분</h1><img style="max-height:none;width:100%" src="data:image/png;base64,'+base64.b64encode(chart.read_bytes()).decode()+'"><p>같은 30초 구간에서 검출 지표와 추적 지표를 나란히 비교했다. 후보마다 하나의 검출 조건만 바꾸고 앵커·다수결·BoT-SORT 설정은 고정했다. 비교 기준 800은 V3의 800×1440이며, 1088/1280은 입력 높이를 줄여 표기했다.</p><p>각 지표의 값이 높아지는 방향이 같지는 않다. 검출 F1과 평균 HOTA 기준 후보를 별도로 표시하고, Global IDF1·IDSW도 함께 판단해야 한다. 별도 구간의 결과로 설정을 재선정하지 않았다.</p>')
 for scope,label in [('development','개발 구간'),('holdout','별도 구간')]:
  dr=[r for r in d if r['scope']==scope and r['stage']=='selected' and r['camera']==0 and r['gt_height']=='all']
  mr=[r for r in m if r['scope']==scope and r['variant'] in detvars]
  pages.append('<h1>'+label+' · 검출 → 전체 지표</h1>'+table(dr,df)+table(mr,fields)+'<p>입력 크기는 높이×너비. D2는 입력 800×1440을 유지한 2×2 타일 4회 추론이며, 타일은 1152×648, 원본 영상 기준 20% 겹침이다. 타일별 NMS 후 원본 좌표로 합치고 NMS 0.7을 다시 적용했다. 전체 영상 패스는 추가하지 않았다. D3는 전체 영상 800×1440에서 NMS만 0.8로 변경했다.</p><p>postprocess confidence=0.001, objectness&gt;0.1 저장, 추적 입력 objectness≥0.3, 픽셀 clipping/정수화·인원 제한 없음은 기존과 같다. IoU≥0.5 최대 대응 수를 먼저 확보하는 1:1 검출 평가이며, 추적 평가는 V3의 TrackEval 기반 구현을 그대로 사용한다. 검출 진단의 매칭은 유효 대응 수를 우선하도록 명시했으며 모든 기준·후보를 같은 방법으로 재계산했다. 이 차이로 기준 검출의 개발 5초/30초 대응 수가 과거 진단보다 1개/14개 늘어난 부분은 모델 개선이 아니다.</p>')
 for scope,label in [('development','개발'),('holdout','별도')]:
  dr=[r for r in d if r['scope']==scope and r['variant']=='baseline' and r['stage']=='selected' and r['seconds']==30 and ((r['camera']==0) or r['gt_height']=='all')]
  pages.append('<h1>'+label+' · 카메라/크기별 누락 진단</h1>'+table(dr,[('scout_camera','SCOUT 카메라',str),('camera','UWIPL 번호',str),('gt_height','GT 높이',str),('gt_boxes','GT',number),('true_positives','대응',number),('recall','재현율',percent),('precision','정밀도',percent)])+'<p>높이: lt50=&lt;50px, 50to100=50≤h&lt;100px, ge100=100px 이상. 높이별 그룹은 GT 기준이므로 대응 GT가 없는 false positive의 높이 그룹을 임의 배정하지 않았고, 높이별 정밀도는 비워 두었다. 카메라×높이 전체 교차표와 5초 결과는 detection_comparison.csv에 있다.</p><p>카메라 번호 혼동을 막기 위해 c001=cam_0, c004=cam_3, c005=cam_4를 명시했다. 요청이 원본 cam 번호를 뜻할 가능성을 고려해 cam_1·cam_5 이미지도 추가했다. 대표 장면은 누락 GT가 많은 시점 중 1초 이상 간격으로 골랐으며 일반적 장면 분포를 대표하지 않는다.</p>')
 pages.append('<h1>다수결 / STCRA 뒤 다수결</h1>'+table([dict(r,variant=('개발/' if r['scope']=='development' else '별도/')+r['variant']) for r in m if r['variant'] in ['baseline','vote_20','vote_30','stcra_on','stcra_postvote15']],fields)+'<p>vote 20/30은 V3 검출·특징·SCT를 유지하고 다수결 블록만 바꿨다. 블록은 프레임 간격이 아닌 트랙별 배정 관측 수다. STCRA ON 대조군은 V3의 거리 4/3/2, confidence 0.65/0.70/0.75, 3회 반복 결과를 재사용했다. 추가 실험은 STCRA 출력에 Local track별 15개 관측 다수결만 다시 적용했으며 삭제된 행을 복구하지 않는다. 요청의 대안 중 후처리 다수결을 실행했고 smooth_avg_tracks는 변경하지 않았다.</p><p>STCRA는 왜곡 보정 발 좌표와 Z=0 단일 평면 H를 사용하는 실험이다. Mesh 기하 검증 완료를 의미하지 않는다. 중복 Global ID 행은 기존 평가 규칙대로 처리하며 raw_rows, rows_evaluated, duplicate_id_rows_removed_for_eval을 함께 기록했다.</p>')
 with (O/'stage_gt_recall.csv').open() as f:stage_rows=list(csv.DictReader(f))
 stage_table=[]
 for scope in ['development','holdout']:
  for variant in ['baseline','D1_1088x1920','D2_tiles2x2']:
   rr=[r for r in stage_rows if r['scope']==scope and r['variant']==variant and r['seconds']=='30']
   stage_table.append(dict(scope=scope,variant=variant,**{r['stage']:float(r['recall']) for r in rr}))
 with (O/'detection_error_breakdown.csv').open() as f:errors=[r for r in csv.DictReader(f) if r['seconds']=='30' and r['variant'] in ['baseline','D2_tiles2x2']]
 pages.append('<h1>검출 이후 손실과 타일링 FP</h1><h2>30초 · 단계별 GT 박스 재현율</h2>'+table(stage_table,[('scope','구간',str),('variant','조건',str),('selected_detection','검출',percent),('botsort','BoT-SORT',percent),('anchor','앵커 배정',percent),('anchor_after_duplicate_id_removal','평가 중복 제거 후',percent)])+'<p>여기서는 ID 정답 여부와 별개로 각 단계 박스가 GT와 IoU≥0.5로 1:1 대응하는지를 계산했다. 따라서 재현율이 좋아져도 Global ID 연결 정확도가 좋아진다는 뜻은 아니다. 검출→SCT→앵커 이후에도 필터·ID 중복 처리 손실이 남는다.</p><h2>30초 · 평가상 false positive 분해</h2>'+table(errors,[('scope','구간',str),('variant','조건',str),('true_positives','TP',number),('false_positives','FP',number),('unmatched_prediction_overlaps_accounted_GT','이미 대응한 GT와 겹침',number),('unmatched_prediction_no_GT_iou_ge_0.5','어느 GT와도 IoU<0.5',number)])+'<p>별도 30초의 타일링은 기준보다 TP 854개가 늘었지만 FP 7,033개가 늘었다. 이미 대응한 GT와 겹치는 미배정 검출 증가는 513개여서, 손실을 타일 간 중복만으로 설명할 수는 없다. FP는 이 GT·IoU 규약에서의 분류이며 실제 배경 오검출인지, 박스 위치 오차인지, 주석 불일치인지까지 자동 확정한 것은 아니다.</p><p>새로 맞춘/놓친 GT의 카메라·높이별 개수, 새 검출이 넓이·점수 조건과 11개 앵커 시점을 통과하는지는 detection_gain_loss.csv에 기록했다. 이 사후 진단은 모델 입력이나 파라미터 선정에 GT를 넣지 않는다.</p>')
 labels=load(O/'label_audit/annotation_format_consistency.json');d4=load(O/'D4_feasibility.json')
 pages.append('<h1>주석 점검과 D4 진행 조건</h1><p>대표 source frame 277의 cam_0/1/3/4/5에서 MOT 박스와 배포 raw JSON projections_2d가 정확히 일치했다. 따라서 확인한 표본의 좌표 어긋남은 단순 MOT 변환 오차로 설명되지 않는다. 해당 표본의 raw 주석 시각과 카메라 timestamp 차이는 최대 약 22ms이므로 큰 위치 차이를 단순한 프레임 번호 오프셋으로 단정할 근거도 없다. 촬영 시각·주석 투영/가시성·라벨 품질의 원인은 미확정이며, 성능이 잘 나오는 방향으로 GT를 이동하거나 제외하지 않았다.</p><p>D4는 요청대로 D1~D3 결과 검토 이후의 조건부 단계이며 <b>학습을 시작하지 않았다</b>. 원래 가중치도 변경하지 않았다. 공식 UMPN scouttrain은 sequence_2의 22개 카메라와 sequence_1의 cam_8~24를 사용한다. 이번 평가의 시간·다른 카메라 누출을 더 엄격히 피하려면 sequence_2만 사용하는 학습 후보를 권장한다.</p><p>sequence_2 학습 후보 '+str(d4['candidate_images'])+'장(원본 frame 10개마다 한 장)을 확인했다. 이는 학습 완료나 공식 전체 학습 분할 그대로의 실행이 아니다. 추가로 timestamp를 중복 제거해 sequence_2 앞 75%를 학습, 뒤 15%를 내부 검증으로 나눴다. 가운데 119.6초를 비웠고 학습 18,115장·내부 검증 3,526장의 YOLOX용 COCO 주석을 준비했다. 주석이 없는 1,670장은 정답 없는 빈 이미지로 취급하지 않고 제외했다. YOLOX 원본 MOTDataset·TrainTransform으로 22카메라×2분할=44장 로딩·전처리 검증을 통과했다. 학습 주석 정합성 검토와 epoch·LR·seed 확정은 남아 있다. V4 개발/별도 sequence_1로 checkpoint를 선택하지 않는다.</p><p>출처: <a href="https://github.com/cvlab-epfl/UMPN/blob/main/datasets/factory.py">공식 UMPN datasets/factory.py</a>. 후보 목록 D4_candidate_images.csv와 D4_feasibility.json을 제공한다.</p><h2>재현 자료</h2><p>experiment_comparison.csv: 앵커 수·GT 커버·쪼개짐/섞임·미배정·전체 지표. GT 커버는 앵커 구성원으로 대응한 인물 수이며 대표 GT 커버와 별도 열로 구분했다. detection_comparison.csv: 카메라/GT 높이별 검출 지표. 각 실험 폴더: predictions.csv, anchor_gt_audit.json, summary.json, metrics.json, duplicate_removals.json. logs/*.status.json: 실제 실행 명령과 시간. 입력 manifest·가중치/소스 해시·검증 결과를 함께 보존했다.</p>')
 for scope,label in [('development','개발'),('holdout','별도')]:
  ims=[]
  for cam in [1,4,5]:
   p=next(iter(sorted((O/f'diagnostics/{scope}/baseline/missed_frames').glob(f'c{cam:03d}_*.jpg'))),None)
   if p:ims.append('<h3>'+html.escape(p.stem)+' · c%03d=cam_%d'%(cam,cam-1)+'</h3><img src="data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()+'">')
  pages.append('<h1>'+label+' · 대표 GT 시각화</h1><p>빨강=검출과 IoU≥0.5 대응되지 않은 GT, 초록=대응 GT. 박스는 원본 제공 GT이며 사람의 가시성·정확한 위치를 추가 판정한 결과가 아니다.</p>'+''.join(ims))
 css='@page{size:A4;margin:12mm}body{font-family:"Noto Sans CJK KR",sans-serif;font-size:12px;color:#172b40}section{page-break-after:always}section:last-child{page-break-after:auto}h1{font-size:23px;color:#143c64;border-bottom:2px solid #2878a8;padding-bottom:7px}h2{font-size:16px}h3{font-size:12px;margin:6px 0}p{line-height:1.65}table{border-collapse:collapse;width:100%;font-size:10px;margin:10px 0}th,td{padding:4px 3px;border-bottom:1px solid #dce4eb;text-align:right}th{background:#eaf2f8}td:first-child,th:first-child{text-align:left}.lead{font-size:13px}.warn{background:#fff2d9;padding:8px}img{display:block;max-height:218px;max-width:100%;margin:auto}'
 htmltext='<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+TITLE+'</title><style>'+css+'</style><body>'+''.join('<section>'+p+'</section>' for p in pages)+'</body></html>'
 (rep/(TITLE+'.html')).write_text(htmltext)
 lines=['# UWIPL SCOUT V4 결과','',f'개발 평균 HOTA 후보: {besth["variant"]}; 검출 F1 후보: {bestf["variant"]}.','', '기준 V3 유지. 모든 검출 조건은 독립 변경이며 STCRA OFF. D4는 조건부 검토 단계로 학습 미실행.','', '|구간|조건|초|HOTA|DetA|AssA|IDF1 2D|IDF1 view|IDSW|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in m:lines.append('|'+ '|'.join([r['scope'],r['variant'],str(r['seconds'])]+[percent(r[k]) for k in ['HOTA_2D','DetA_2D','AssA_2D','IDF1_2D','IDF1_view_aware']]+[str(r['IDSW_2D'])])+'|')
 lines+=['','개발 30초 시간 불일치, GT 투영 박스의 시각적 어긋남, 5초/30초 중첩을 함께 고려해야 한다. GT 박스 상한 진단은 score=1도 바꾸므로 검출 재현율만의 효과가 아니다.','', '자세한 프로토콜과 한계는 동명 HTML/PDF, 카메라·높이별 진단은 detection_comparison.csv를 참조한다.']
 (rep/'START_HERE.md').write_text('\n'.join(lines)+'\n');print('REPORT',rep/TITLE,flush=True)
def package():
 rep=O/'report';pdf=rep/(TITLE+'.pdf');assert pdf.exists()
 H=O/'handoff';H.mkdir(exist_ok=True)
 def put(p,rel):q=H/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for p in rep.iterdir():
  if p.is_file():put(p,'01_report/'+p.name)
 for name in ['experiment_comparison.csv','experiment_comparison.json','detection_comparison.csv','timing.csv','verification.json','development_selection.json','run_state.json','experiment_design.json','reproducibility.json','D4_feasibility.json','D4_candidate_images.csv','detection_gain_loss.csv','detection_error_breakdown.csv','stage_gt_recall.csv'] :put(O/name,'02_summary/'+name)
 for p in (O/'experiments').rglob('*'):
  if p.is_file() and p.name in ['predictions.csv','summary.json','metrics.json','anchor_gt_audit.json','duplicate_removals.json','hungarian_unassigned.json','v3_equivalence.json']:put(p,'03_experiments/'+str(p.relative_to(O/'experiments')))
 for prefix in ['diagnostics','label_audit','validation','reproducibility']:
  for p in (O/prefix).rglob('*'):
   if p.is_file():put(p,'04_diagnostics/'+prefix+'/'+str(p.relative_to(O/prefix)))
 for p in (O/'D4_prepared').rglob('*'):
  if p.is_file():put(p,'07_D4_preparation/'+str(p.relative_to(O/'D4_prepared')))
 for p in (O/'logs').glob('*.status.json'):put(p,'05_reproducibility/'+p.name)
 for p in (O/'sct').glob('*/*/config.json'):put(p,'05_reproducibility/sct/'+str(p.relative_to(O/'sct')))
 for scope in ['development','holdout']:
  p=R/f'outputs/uwipl-v3-20261107/sct/{scope}/new_0.7_high_0.4/config.json'
  if p.exists():put(p,'05_reproducibility/sct/'+scope+'_baseline_config.json')
 for p in (O/'inference').glob('*/*/uwipl/*/report.json'):put(p,'05_reproducibility/inference/'+str(p.relative_to(O/'inference')))
 for name in ['uwipl-v4.py','uwipl-v4-detect.py','run-uwipl-v4.py','report-uwipl-v4.py','audit-uwipl-v4-labels.py','analyze-uwipl-v4-losses.py','prepare-uwipl-v4-finetune.py','check-uwipl-v4-training-input.py','plot-uwipl-v4.py','finish-uwipl-v4.py','uwipl-v3.py','scout-sample-stage.py','evaluate-uwipl-study.py']:put(R/'scripts'/name,'06_source/'+name)
 for scope,path in [('development',R/'outputs/uwipl-study-20261101/inference30/input/manifest.json'),('holdout',R/'outputs/uwipl-v3-20261107/holdout/inference/input/manifest.json')]:put(path,'05_reproducibility/'+scope+'_input_manifest.json')
 sums=[dict(file=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(H.rglob('*')) if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt']]
 csvout(H/'FILES.csv',sums);(H/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['file']}\n" for r in sums))
 zpath=R/'deliverables/UWIPL_SCOUT_V4_20261108.zip';assert not zpath.exists()
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(H.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(H)))
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for r in sums:assert hashlib.sha256(z.read(r['file'])).hexdigest()==r['sha256']
 digest=sha(zpath);zpath.with_suffix('.zip.sha256').write_text(digest+'  '+zpath.name+'\n')
 for ext in ['pdf','html']:shutil.copy2(rep/(TITLE+'.'+ext),R/'docs/handoff'/(TITLE+'.'+ext))
 save(O/'package_verification.json',dict(status='PASS',file=str(zpath),bytes=zpath.stat().st_size,files=len(sums)+2,sha256=digest,CRC='PASS',all_file_hashes='PASS'));print(zpath,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','build','package']);a=p.parse_args();globals()[a.mode]()
