#!/usr/bin/env python3
"""Create six requested review tables, figures and a paginated Korean report."""
import ast,base64,collections,csv,html,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-flat-20261101';V=O/'visualizations';T=O/'tables'
STEM='PoseTrack_SCOUT_평면ROI_내부계산검증'
def load(p):return json.loads(p.read_text())
def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
def pic(p):return 'data:image/'+('png' if p.suffix=='.png' else 'jpeg')+';base64,'+base64.b64encode(p.read_bytes()).decode()
def page(i,title,body):return f'<section class="page"><div class="eyebrow">SCOUT × PoseTrack · 평면 ROI 내부 계산 검증</div><h1>{title}</h1>{body}<footer><span>개발용 5초 구간 · 원본 임계값 유지 · 성능 평가 아님</span><b>{i} / 6</b></footer></section>'
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def main():
    V.mkdir(exist_ok=True);T.mkdir(exist_ok=True)
    ev=load(O/'events.json');summary=load(O/'summary.json');prep=load(O/'preparation_summary.json');plane=load(O/'plane.json');verification=load(O/'verification.json')
    assert summary['status']==verification['status']=='PASS'
    pairs=readcsv(O/'affinity_pairs.csv');outputs=readcsv(O/'global_predictions.csv');observations=load(O/'observations.json')
    obs={f"cam{x['camera']}_row{x['camera_row']}":x for x in observations}
    reids=[e for e in ev if e['event']=='state_aware_reid'];states=[e for e in ev if e['event'] in ['view_state_assignment','view_state_detected']]
    banks=[e for e in ev if e['event']=='feature_bank'];gates=[e for e in ev if e['event']=='feature_bank_gate']
    csvout(T/'01_geometry_affinity.csv',pairs)
    csvout(T/'02_hungarian_matches.csv',[r for r in pairs if r['hungarian_selected']=='True'])
    csvout(T/'03_state_decisions.csv',states);csvout(T/'04_state_aware_reid.csv',reids)
    csvout(T/'05_feature_bank_changes.csv',banks);csvout(T/'05_feature_bank_conditions.csv',gates)
    raw=[]
    for r in outputs:
        o=obs[r['observation_id']];b=o['raw_bbox'];raw.append({'source_frame':r['source_frame'],'internal_frame':r['frame'],
          'scout_camera':'cam_'+r['camera'],'global_id':r['global_id'],'observation_id':r['observation_id'],
          'raw_x':b[0],'raw_y':b[1],'raw_w':b[2]-b[0],'raw_h':b[3]-b[1],
          'plane_x':r['plane_x'],'plane_y':r['plane_y'],'output_method_code':r['output_method_code'],
          'raw_bbox_origin':'original detection mapped by observation_id, not inverse-warped envelope'})
    csvout(T/'06_global_ids_raw_bbox.csv',raw)
    # Human-readable matrices, with row/column identity labels, in addition to complete NPZ.
    (T/'matrices').mkdir(exist_ok=True)
    for p in (O/'matrices').glob('*.npz'):
        z=np.load(p)
        if not z['final_affinity'].size:continue
        rows=[{'observation_id':str(ob),**{f'ID_{int(tid)}':float(a) for tid,a in zip(z['track_ids'],z['final_affinity'][i])}} for i,ob in enumerate(z['observation_ids'])]
        csvout(T/'matrices'/p.with_suffix('.csv').name,rows)
    active=next(r for r in pairs if r['accepted_by_score']=='True' and r['reid_weight']=='1.0' and float(r['As'])>.1)
    rejected=next(r for r in pairs if r['hungarian_selected']=='True' and r['accepted_by_score']=='False')
    ordinary=next(r for r in pairs if r['accepted_by_score']=='True' and r['reid_weight']!='1.0')
    self_checks=[e for e in ev if e['event']=='occlusion_self_similarity']
    cases={'accepted_reid_active':active,'accepted_reid_inactive':ordinary,'hungarian_proposed_but_rejected':rejected,'occlusion_self_checks':self_checks}
    js(O/'worked_examples.json',cases)
    statecount=collections.Counter(e['assigned_state'] for e in states)
    counts={'state_assignments':dict(statecount),'states_empty_camera_missing':sum(e.get('reason')=='no_detections_in_this_camera' and e['camera']<3 for e in states),
      'hungarian_proposals':sum(r['hungarian_selected']=='True' for r in pairs),'hungarian_zero_score_rejected':sum(r['hungarian_selected']=='True' and r['accepted_by_score']=='False' for r in pairs),
      'bank_changed_calls':sum(e['bank_changed'] for e in banks),'bank_vectors_written':sum(len(e['changed_slots']) for e in banks),
      'later_bank_write_conditions':sum(e['store_condition'] for e in gates),'id_switch_correction_events':sum(e['event']=='switch_view' for e in ev),
      'missing_reactivation_events':sum(e['event']=='reactivation' for e in ev)}
    js(O/'review_counts.json',counts)
    # ROI overview: all prior six-time mesh intersections, then full selected observations.
    dg=[r for r in readcsv(R/'outputs/posetrack-transfer-20261101/handoff/02_adapter/ground_geometry_diagnostic.csv') if r['mesh_x']]
    fig,axs=plt.subplots(1,2,figsize=(10,3.3));lo,hi=np.array(plane['bounds_scout_xy'])
    axs[0].scatter([float(r['mesh_x']) for r in dg],[float(r['mesh_y']) for r in dg],s=5,c='#aeb7c0',label='Prior sampled mesh intersections')
    axs[0].add_patch(plt.Rectangle(lo,*(hi-lo),fill=False,edgecolor='#b84a38',lw=2));axs[0].set_xlabel('SCOUT world X');axs[0].set_ylabel('SCOUT world Y');axs[0].set_title('Mesh-selected local ROI');axs[0].legend(fontsize=6)
    for c in sorted({r['camera'] for r in observations}):
        pts=np.array([r['mesh_world'] for r in observations if r['camera']==c]);axs[1].scatter(pts[:,0],pts[:,1],s=10,alpha=.6,label=f'cam_{c}')
    axs[1].set_xlim(lo[0]-.3,hi[0]+.3);axs[1].set_ylim(lo[1]-.3,hi[1]+.3);axs[1].set_xlabel('SCOUT world X');axs[1].set_ylabel('SCOUT world Y');axs[1].set_title('444 retained observations (not GT)');axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(V/'roi.png',dpi=160);plt.close(fig)
    frame=int(active['frame']);cam=int(active['camera']);z=np.load(O/f'matrices/frame_{frame:03d}_cam_{cam}.npz');A=z['final_affinity']
    fig,ax=plt.subplots(figsize=(8,3));im=ax.imshow(A,cmap='YlGnBu',vmin=0,aspect='auto');fig.colorbar(im,ax=ax,label='Final affinity')
    ax.set_xticks(range(A.shape[1]));ax.set_xticklabels([f'ID {x}' for x in z['track_ids']]);ax.set_yticks(range(A.shape[0]));ax.set_yticklabels(z['observation_ids'])
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):ax.text(j,i,f'{A[i,j]:.3f}',ha='center',va='center',color='white' if A[i,j]>A.max()*.6 else 'black')
    ax.scatter(z['hungarian_cols'],z['hungarian_rows'],facecolors='none',edgecolors='#e76439',s=450,lw=2)
    ax.set_title(f'Source frame {frame+277}, cam_{cam}: circles = Hungarian proposal');fig.tight_layout();fig.savefig(V/'affinity_example.png',dpi=170);plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    # Raw-image overlays use the original observed boxes, maintaining the raw/rectified split.
    for c in range(3,8):
      for f in [5,24,49]:
        im=Image.open(R/f'outputs/posetrack-transfer-20261101/inference/input/cam_{c}/frame_{f:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im)
        selected=[o for o in observations if o['camera']==c and o['frame']==f]
        ids={r['observation_id']:r['global_id'] for r in outputs if int(r['camera'])==c and int(r['frame'])==f}
        for ob in selected:
            oid=f"cam{c}_row{ob['camera_row']}";tid=ids.get(oid);b=ob['raw_bbox'];color='#f9bd38' if tid is None else ('#d9424b','#287dc0','#209a72','#9660cc','#db7a2d','#bb458d')[(int(tid)-1)%6]
            draw.rectangle(b[:4],outline=color,width=4);draw.text((b[0],max(0,b[1]-25)),f'ID {tid}' if tid else 'no output',font=font,fill=color,stroke_width=1,stroke_fill='black')
        banner=Image.new('RGB',(1920,1120),'white');banner.paste(im,(0,40));ImageDraw.Draw(banner).text((20,8),f'cam_{c} / source {f+277} / raw image, original boxes / ROI-selected predictions',font=font,fill='black')
        banner.save(V/f'cam_{c}_source_{f+277}.jpg',quality=88)
    provenance=O/'reproducibility';provenance.mkdir(exist_ok=True)
    for name in ['prepare-posetrack-flat.py','trace-posetrack-flat.py','verify-posetrack-flat.py','report-posetrack-flat.py']:
        cp(R/'scripts'/name,provenance/'scripts'/name)
    for rel in ['Tracker/PoseTracker.py','util/camera.py','Solver/bip_solver.py','aic_cpp/src/aic.cpp']:
        cp(R/'repos/posetrack/track'/rel,provenance/'upstream_track'/rel)
    for rel in ['Dockerfile','requirements.txt']:cp(R/'docker/posetrack'/rel,provenance/'docker'/rel)
    cp(R/'weights/posetrack/manifest.json',provenance/'weights_manifest.json')
    # Exact cached inference reports are evidence of the preceding stages, not fresh inference here.
    for stage in ['detection','pose','reid']:
        cp(R/f'outputs/posetrack-transfer-20261101/inference/posetrack/{stage}/report.json',provenance/f'reused_{stage}_report.json')
    code_map=[('Ab / Kalman IoU','Tracker/PoseTracker.py','compute_bboxiou_aff',625),('Aepi / Ah','Tracker/PoseTracker.py','compute_epi_homo_aff',544),
      ('실제 C++ geometry','aic_cpp/src/aic.cpp','loop_t_homo_full',423),('최종 점수 / Hungarian','Tracker/PoseTracker.py','mv_update_wo_pred',842),
      ('카메라 상태 판정','Tracker/PoseTracker.py','mv_update_wo_pred',900),('State-aware As','Tracker/PoseTracker.py','compute_reid_aff',463),
      ('가림 후 교환 / Bank 갱신','Tracker/PoseTracker.py','multi_view_3D_update',301),('Missing 재활성화','Tracker/PoseTracker.py','match_with_miss_tracks',692),
      ('Global ID 출력','Tracker/PoseTracker.py','output',985)]
    csvout(T/'code_locations.csv',[dict(zip(['topic','file','function','line'],r)) for r in code_map])
    readme='''# PoseTrack 평면 ROI 내부 계산 검증

목표: SCOUT의 작은 평면 근사 영역에서 원본 추적기를 실행하고, ID가 연결되는 수치와 조건을 확인합니다. 정확도 평가나 전체 SCOUT 보정 완료를 의미하지 않습니다.

먼저 PDF와 worked_examples.json을 보세요. CSV는 UTF-8 BOM 형식이며 Excel에서 열 수 있습니다.

## 요청한 6개 항목
1. tables/01_geometry_affinity.csv: Ab, 원시/하한 제한 Aepi·Ah, As, ReID weight, norm, 최종 점수.
2. tables/02_hungarian_matches.csv: Hungarian 배정 437쌍, score>0 수락 413쌍, score=0 거절 24쌍. tables/matrices는 행·열 이름을 붙인 최종 행렬. matrices/*.npz는 전체 구성 행렬·cost·배정 인덱스.
3. tables/03_state_decisions.csv: 실제 Detected/Occluded/Missing 대입 이벤트와 조건. state_history.csv 및 track_snapshots.json은 프레임별 상태 이력. 대입 횟수는 사람 수나 IDSW가 아닙니다.
4. tables/04_state_aware_reid.csv: 활성화 여부, 이유, max cosine, 코드에서 사용한 As=max_positive_cosine−0.5, weight. 가림 해제 시 self similarity와 ID 교환 결정은 events.json. As는 이 보고서의 코드값 표기이며 원시 cosine과 구분합니다.
5. tables/05_feature_bank_changes.csv 및 conditions.csv: 갱신 전후 count, 변경 슬롯, 조건. feature_bank_updates.npz에 실제 저장 특징(2048차원)을 보존합니다.
6. tables/06_global_ids_raw_bbox.csv: source frame/cam/observation/global ID와 원본 BBox. global_predictions.csv의 BBox는 왜곡 보정 좌표입니다. 두 파일의 좌표를 혼용하지 마세요.

## 공통 키
internal_frame 0..49 = source_frame 277..326. cam_0..cam_7이며 선택 영역에 실제 입력이 남은 카메라는 cam_3..cam_7입니다. observation_id=cam{번호}_row{원본 카메라별 검출행}. track_id와 global_id는 같은 추적 ID입니다. 카메라별 state와 전체 track state는 서로 다른 enum입니다.

## 좌표 및 제한
메시만으로 XY [32,40]×[71,79] 영역의 국소 평면을 적합했습니다. 모든 카메라는 동일 W_plane_to_scout와 P_plane=K[R|T]W, H=P_plane[:,[0,1,3]]를 사용합니다. 관절·BBox를 왜곡 보정하고, 원본 ReID 특징은 재사용했습니다. ROI/평면 오차/보정 BBox 하단 오차로 입력을 선택했으므로 이를 검출 재현율로 해석하지 마세요.
높이 0.15와 거리 0.2/1.5는 원본값을 유지하고 SCOUT 데이터 단위를 사용했습니다. 단위의 독립 실측, 발목 높이, 실제 ID 정답은 미검증입니다. 10FPS 입력에서 원본 age15프레임 등도 그대로입니다. output_method_code=1 BBox / 2 발목 / 3 3D관절이며 세계 Z가 아닙니다.
원본 target_init의 fallback에는 (x1+x1)/2 및 한쪽 Hfeet 사용이 남아 있습니다(PoseTracker.py 767~789). 이번에는 원본 분기를 관찰하기 위해 수정하지 않았습니다. 실제 실행된 경우와 개선 효과는 별도 실험 대상입니다.

## 검증
2,255개 Geometry 후보의 C++ 반환값과 별도 설명용 계산을 전부 대조했습니다. 로그를 끈 원본 추적을 동일 입력에 다시 실행해 416개 출력과 매 프레임 상태·bank count 일치를 확인했습니다. GT는 ROI 선택·추론·추적에 사용하지 않았습니다. 가림 후 self similarity 확인 4건은 모두 자기 ID 유지였으며 다른 ID 교환과 Missing 재활성화는 관찰되지 않았습니다.

## 서버에서의 실행 순서
기존 scout-mtmc/posetrack-all:ba727e6 이미지. /project=프로젝트 읽기 전용, /data=SCOUT 읽기 전용, /out=이 실험 출력. runtime=runc, network=none, user=1000:1000, CPUs=4, memory=8GB. GPU·가중치 추가 다운로드 불필요.
python /project/scripts/prepare-posetrack-flat.py
python /project/scripts/trace-posetrack-flat.py
python /project/scripts/verify-posetrack-flat.py
전체 데이터/영상/가중치 대신 검토용 그림과 선택된 444개 중간 입력만 동봉했습니다. 원본 실험과 기존 PDF/ZIP은 보존했습니다.
'''
    (O/'START_HERE.md').write_text(readme)
    pages=[]
    pages.append(page(1,'작은 평면 영역에서 추적 실행',f'''<p class="lead">Geometry Affinity, Hungarian, 상태 기반 Re-ID가 <b>어떤 값과 조건으로 ID를 연결하는지</b> 실제 SCOUT 입력으로 확인했습니다.</p>
<div class="stats"><div><strong>444개</strong><span>ROI에 남은 입력 검출</span></div><div><strong>416행 / 6 ID</strong><span>최종 추적 출력</span></div><div><strong>50프레임</strong><span>source 277~326 / 약 5초</span></div></div>
{table(['항목','범위 / 결과'],[('카메라','sequence_1 cam_0~7 중 ROI 관측이 있는 cam_3~7'),('선택 영역','SCOUT world X 32~40, Y 71~79 / 약 8×8 데이터 단위'),('선정 기준','GT 미사용. 메시 평탄도 + 다중 카메라 관측 수'),('평탄도','메시 점→적합 평면 수직거리 p95 0.0189 데이터 단위'),('실제 선택점 오차','같은 광선의 메시/평면 교점 거리 p95 0.1407, 최대 0.2112'),('입력 처리','기존 검출·포즈·ReID 재사용, 관절·BBox 왜곡 보정'),('실행 설정','원본 추적 코드·임계값 유지. CPU 컨테이너에서 실행')])}
<img style="width:100%" src="{pic(V/'roi.png')}"><p class="small">평면은 제한된 영역의 근사입니다. 원래 5,569개 중 영상 밖 58개, ROI 밖 5,067개를 제외했습니다. 이 결과는 전체 SCOUT 성능 또는 물리 보정 정확도 검증이 아닙니다.</p>'''))
    example_fields=['Ab','Aepi','Ah','As','reid_weight','norm','final_score']
    pages.append(page(2,'Geometry Affinity와 최종 점수',f'''<p>모든 후보 쌍에 대해 원본 C++가 계산한 값과 별도 설명용 계산을 대조했습니다. <b>2,255쌍 모두 일치</b>했습니다.</p>
{table(['기호','실제 코드의 계산'],[('Ab','(Kalman 예측 박스와의 IoU − 0.5) × 검출 score × [age_bbox≤15]'),('Aepi','유효 관절 광선 사이 거리 d로 1−d/0.2 계산, exp(−age) 가중 후 카메라 평균'),('Ah','평면 좌표 거리 d로 1−d/1.5 계산. 양쪽 발목 유효 시 z=0.15, 아니면 BBox 하단 z=0'),('As','활성 조건 통과 시 feature bank의 최대 양수 cosine − 0.5'),('하한 제한','Aepi와 Ah를 각각 −2.5 이상으로 제한'),('최종 점수','(Aepi+5Ab+Ah+5As×w)/(1+w), 이후 norm 보정, 음수는 0')])}
<p class="small">w=1(활성) 또는 0.00001(비활성). norm=1[Aepi≠0]+5[Ab≠0]+1[Ah≠0]. norm&gt;0이면 (5−norm)×0.1을 뺍니다. norm=7일 때는 오히려 0.2가 더해지는 원본 계산을 그대로 기록했습니다.</p>
<h2>실제 수락 사례: source {active['source_frame']}, cam_{active['camera']}</h2><p>{active['observation_id']} → Global ID {active['track_id']}</p>
{table(example_fields,[[f'{float(active[k]):.6f}' for k in example_fields]])}
<p>이 후보는 Re-ID가 활성화됐으며 최종 점수 {float(active['final_score']):.6f}로 Hungarian에서 선택되고 갱신됐습니다. 이것이 GT 기준 정답이라는 의미는 아닙니다.</p>
<h2>관측을 사용할 조건</h2><p>Aepi는 다른 카메라의 age_bbox&lt;2, 양쪽 관절 score&gt;0.7, 대상 관절 age≤1 조건을 사용합니다. Ah는 두 관측의 양쪽 발목 score&gt;0.7이면 발목을 쓰고, 그렇지 않으면 BBox를 씁니다. geometry_details.json에 카메라별 거리·유효 관절·평면 좌표를 담았습니다.</p>
<p class="small">새 ID 초기화는 별도입니다. 미매칭 검출 간 2×Aepi+Ah를 GLPK로 묶고 품질 조건을 확인합니다. 이 단계의 행렬·클러스터도 events.json에 있습니다.</p>'''))
    pages.append(page(3,'Hungarian: 배정과 수락을 분리',f'''<p>행은 검출, 열은 기존 Global ID입니다. 원본은 <code>linear_sum_assignment(−Affinity)</code>로 전체 합이 큰 배정을 찾은 뒤, <b>점수&gt;0인 배정만 갱신</b>합니다.</p>
<div class="stats"><div><strong>437쌍</strong><span>Hungarian 제안</span></div><div><strong>413쌍</strong><span>양수 점수로 수락</span></div><div><strong>24쌍</strong><span>점수 0으로 거절</span></div></div>
<img style="width:100%" src="{pic(V/'affinity_example.png')}">
<h2>배정됐지만 연결되지 않은 실제 사례</h2>
{table(['source frame','camera','검출','후보 ID','최종 점수','실제 처리'],[[rejected['source_frame'],'cam_'+rejected['camera'],rejected['observation_id'],rejected['track_id'],rejected['final_score'],'갱신 안 함']])}
<p>Hungarian의 반환 인덱스만 보면 위 사례를 연결 성공으로 오해할 수 있습니다. CSV의 <b>hungarian_selected</b>와 <b>accepted_by_score</b>를 분리했고, 실제 single_view_2D_update 호출도 별도로 기록했습니다.</p>
<h2>행렬 확인 방법</h2><p>tables/matrices의 CSV는 행·열 이름을 붙인 최종 점수입니다. matrices의 NPZ에는 Ab·Aepi·Ah·As·weight·norm·cost·선택 인덱스와 전체 행렬이 있습니다. 400개 카메라-프레임 호출 중 빈 행렬도 보존했습니다.</p>
<p class="small">원본과 같은 8개 카메라 인덱스를 유지했으므로 cam_0~2는 이 ROI에서 입력이 없습니다. Global ID 생성과 최종 출력은 초기화·3D 보정 단계를 더 거치므로 수락 413쌍과 출력 416행은 같은 개수가 아닙니다.</p>'''))
    pages.append(page(4,'상태 판정과 State-aware Re-ID',f'''{table(['상태 / 분기','실제 조건'],[('Detected','수락된 검출로 2D 갱신하거나 새 트랙을 초기화할 때'),('Occluded','Hungarian 미배정 트랙 열의 최대 IoU>0.5; 해당 검출에 배정된 다른 트랙에도 표시 가능'),('Missing (카메라별)','미배정 트랙의 최대 IoU≤0.5 또는 해당 카메라에 검출 없음'),('As 활성','검출 score≥0.88, 최근 최대10개 상태 이력에 Occluded/Missing 존재, 양수 bank cosine 존재'),('전체 트랙 Missing','update_age≥15. 카메라별 Missing과 다른 상태'),('가림 해제 후 교환','self cosine≤0.5에서 다른 후보를 비교; 다른 cosine이 self보다 크고 0.5 초과하면 switch_view')])}
<p>실제 상태 대입은 Detected <b>{statecount[1]}</b>건, Occluded <b>{statecount[2]}</b>건, Missing <b>{statecount[3]}</b>건입니다. 이는 프레임·카메라별 이벤트이며 사람 수가 아닙니다. 이 중 {counts['states_empty_camera_missing']}건의 Missing은 ROI 입력이 없는 cam_0~2에서 발생했습니다.</p>
<p>As 활성은 후보 쌍 기준 <b>1,383 / 2,255건</b>입니다. 비활성 이유는 score&lt;0.88:305건, 상태 이력 없음:39건, 최근 이력에 가림/누락 없음:528건입니다.</p>
<h2>가림 해제 뒤 실제 자기 특징 비교</h2>
{table(['source frame','camera','ID','self cosine','결과'],[[e['source_frame'],'cam_'+str(e['camera']),e['track_id'],f"{e['As_self']:.6f}",'자기 ID 유지 (>0.5)'] for e in self_checks])}
<div class="callout"><b>As 활성화와 ID 교환은 다릅니다.</b> 이번에는 위 4건이 모두 자기 ID 유지 조건을 통과했습니다. 다른 ID로 교환 및 Missing 트랙 재활성화는 0건입니다. 이 분기의 성공 사례를 검증했다고 주장하지 않습니다.</div>
<p class="small">카메라 상태 enum: Detected=1 / Occluded=2 / Missing=3. 전체 트랙 enum: Unconfirmed=1 / Confirmed=2 / Missing=3 / Deleted=4. 같은 숫자라도 의미가 다릅니다.</p>'''))
    bchange=next(e for e in banks if e['method']=='multi_view_3D_update' and e['after_count']-e['before_count']==2)
    pages.append(page(5,'Feature Bank와 최종 Global ID',f'''<p>실제 특징 저장이 일어난 함수 호출은 <b>15건</b>, 저장·변경된 슬롯은 <b>19개</b>입니다. 초기화 이후 저장 조건 통과는 10건이며, 초기 특징과 후속 갱신을 구분했습니다.</p>
{table(['후속 저장 조건','원본 값'],[('상체 관절','5·6·11·12번 score가 모두 >0.5'),('검출 신뢰도','score>0.9'),('겹침 제한','IoU>0.15인 수 <2, overlap>0.3인 수 <2'),('새 특징 여부','bank가 비었거나 최대 cosine<0.6'),('저장 위치','feat_count % bank_size; 실제 bank 크기100')])}
<p>예: source {bchange['source_frame']}, ID {bchange['track_id']}의 특징 개수가 {bchange['before_count']}→{bchange['after_count']}로 늘었습니다. CSV에 바뀐 슬롯과 NPZ 벡터 키가 연결되어 있습니다. 갱신되지 않은 호출과 조건도 남겼습니다.</p>
<img style="width:100%;max-height:71mm;object-fit:contain" src="{pic(V/'cam_7_source_282.jpg')}">
<p>숫자·색은 예측 Global ID입니다. 노란 박스는 ROI 입력에는 있으나 해당 프레임의 최종 출력에는 없는 관측입니다. GT 비교는 하지 않았습니다.</p>
<p class="small">global_predictions.csv는 왜곡 보정 BBox, tables/06_global_ids_raw_bbox.csv는 같은 observation_id에 대응하는 원본 검출 BBox입니다. plane_x/y는 선택 평면 좌표이며 W로 SCOUT 좌표계에 대응합니다. output_method_code 1/2/3은 BBox/발목/3D관절 출처 표식으로, 세계 Z가 아닙니다.</p>'''))
    pages.append(page(6,'전달 파일과 검증 범위',f'''{table(['확인할 항목','파일'],[('Ab/Aepi/Ah/As/최종 점수','tables/01_geometry_affinity.csv'),('Hungarian 실제 배정','tables/02_hungarian_matches.csv + tables/matrices/'),('상태와 조건','tables/03_state_decisions.csv + state_history.csv'),('State-aware Re-ID','tables/04_state_aware_reid.csv + events.json'),('Feature Bank 저장·갱신','tables/05_feature_bank_*.csv + feature_bank_updates.npz'),('Global ID 연결','tables/06_global_ids_raw_bbox.csv'),('작은 실제 예제','worked_examples.json + visualizations/'),('코드 위치·원본·계측 코드','tables/code_locations.csv + reproducibility/')])}
<h2>계측이 알고리즘 결과를 바꾸지 않았는가?</h2><p>로그를 끈 원본 추적을 같은 입력에 다시 실행했습니다. <b>416행의 전체 출력 값이 정확히 일치</b>했고, 50개 프레임 모두 트랙 상태·카메라 상태 이력·Feature Bank 개수가 같았습니다. 실제 코드 파일도 이미지와 서버 원본의 바이트 일치를 확인했습니다.</p>
<p>관절을 왜곡 보정한 뒤 다시 왜곡시킨 최대 픽셀 오차는 {verification['keypoint_redistortion_max_px']:.2e}px, 평면 위 P/H 투영 일치 오차는 {verification['P_H_plane_consistency_max_px']:.2e}px였습니다. 수치 일관성 검사이며 실제 사람의 위치 정확도 증명은 아닙니다.</p>
<h2>이번에 바꾸지 않은 부분</h2><p>원본의 age15·history10·관절 임계값·거리0.2/1.5·발목 높이0.15를 유지했습니다. 10FPS에 대한 시간값 보정은 하지 않았습니다. target_init의 BBox fallback 코드에 (x1+x1)/2와 한쪽 Hfeet 사용이 남아 있으며, 이를 임의 수정하지 않고 원본 계산을 기록했습니다.</p>
<div class="callout">목표인 <b>추적 실행과 내부 계산 관찰은 완료</b>했습니다. 전체 SCOUT 평면 보정, 실제 ID 정확도, 다른 ID 교환/재활성화 성공 사례, 장시간 추적 성능은 이번 결과에 포함되지 않습니다.</div>
<p class="small">소스 ba727e6bc6e45e44cd6e006b73e00864892173c2 / 기존 posetrack-all 이미지. 검출·포즈·ReID는 기존 실측 결과를 재사용했습니다. 대형 가중치·전체 데이터 대신 선택 입력과 검토 자료를 전달합니다.</p>'''))
    tree=ast.parse((R/'scripts/create-osc-handoff.py').read_text());css=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='css' for t in n.targets))
    css+='\ntable{font-size:8.3pt}td{word-break:keep-all;overflow-wrap:anywhere}h1{font-size:22pt}.page{overflow:hidden}\n'
    (R/'docs/handoff'/f'{STEM}.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PoseTrack 평면 ROI 내부 계산 검증</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>')
    print('Prepared report, tables and figures.',counts)
if __name__=='__main__':main()
