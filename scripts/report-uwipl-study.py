#!/usr/bin/env python3
"""Assemble reviewed UWIPL experiment records and a Korean PDF source."""
import ast,base64,collections,csv,hashlib,html,json,shutil,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parents[1];A=R/'outputs/uwipl-study-20261101';O=A/'handoff'
STEM='UWIPL_SCOUT_알고리즘검증_20261101'
def load(p):return json.loads(p.read_text())
def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def csvout(p,rs):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def table(h,rs):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rs)+'</tbody></table>'
def pct(v):return f'{100*float(v):.2f}%'
def pic(p):return 'data:image/'+('png' if p.suffix=='.png' else 'jpeg')+';base64,'+base64.b64encode(p.read_bytes()).decode()
def page(i,t,b):return f'<section class="page"><div class="eyebrow">SCOUT × UWIPL · 알고리즘 실행 및 단계별 관찰</div><h1>{t}</h1>{b}<footer><span>2026.11.01 · 개발 구간 실험 · 공식 SCOUT 벤치마크 아님</span><b>{i} / 9</b></footer></section>'

def main():
    metrics=readcsv(A/'experiment_comparison.csv');alg=load(A/'algorithm_summary.json');st=load(A/'stcra_summary.json')
    assert len(metrics)==32 and len(alg)==4 and len(st)==8
    mainalg={x['seconds']:x for x in alg if x['variant']=='fps10'}
    def row(sec,stage,schedule='meter_candidate',variant='fps10'):
        return next(x for x in metrics if int(x['seconds'])==sec and x['stage']==stage and x['schedule']==schedule and x['variant']==variant)
    figs=O/'05_visualizations';figs.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,sec in zip(axs,[5,30]):
        rs=readcsv(A/f'fps10/s{sec}/assignment_cost_gt.csv')
        for label,flag,col in [('Matched anchor GT','True','#267a90'),('Wrong anchor GT','False','#c15b44')]:
            values=[float(x['cost']) for x in rs if x['correct']==flag]
            if values:ax.hist(values,bins=25,alpha=.55,label=f'{label} (n={len(values)})',color=col,density=True)
        ax.set_title(f'{sec}s / raw Hungarian assignment cost');ax.set_xlabel('mean cosine distance to anchor members');ax.set_ylabel('density');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figs/'anchor_costs.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,sec in zip(axs,[5,30]):
        stages=['anchor','stcra','istcra','interpolation']
        for schedule,label in [('original_units','250 / 200 / 150'),('meter_candidate','2.5 / 2 / 1.5')]:
            ax.plot(range(4),[100*float(row(sec,s,schedule)['IDF1_view_aware']) for s in stages],marker='o',label=label)
        ax.set_xticks(range(4));ax.set_xticklabels(stages,rotation=12);ax.set_ylabel('view-aware IDF1 (%)');ax.set_title(f'{sec}s / FPS10 time profile');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figs/'ablation_idf1.png',dpi=160);plt.close(fig)
    # Representative actual anchor and post-STCRA IDs, each camera first/middle/last 5s frame.
    before=load(A/'fps10/s5/meter_candidate/anchor.json');after=load(A/'fps10/s5/meter_candidate/interpolation.json')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for c in range(1,9):
      for f in [0,24,49]:
        panels=[]
        for stage,rows in [('ANCHOR',before),('i-STCRA 2.5/2/1.5 + INTERPOLATION',after)]:
            im=Image.open(A/f'inference30/input/cam_{c-1}/frame_{f:06d}.jpg').convert('RGB');dr=ImageDraw.Draw(im)
            for r in rows:
                if r['camera']!=c or r['frame']!=f:continue
                x,y,w,h=r['bbox'];tid=r['id'];col=((tid*67)%200+40,(tid*37)%200+40,(tid*97)%200+40)
                dr.rectangle([x,y,x+w,y+h],outline=col,width=3);dr.text((x,max(0,y-20)),str(tid),font=font,fill=col,stroke_width=1,stroke_fill='black')
            im=im.resize((960,540));panel=Image.new('RGB',(960,575),'white');panel.paste(im,(0,35));ImageDraw.Draw(panel).text((10,8),f'cam_{c-1} source {277+f} | {stage}',font=font,fill='black');panels.append(panel)
        canvas=Image.new('RGB',(1920,575),'white');canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(960,0));canvas.save(figs/f'cam_{c-1}_frame{277+f}_before_after.jpg',quality=90)
    # Main, genuine event totals and GT-labelled denominators (proposals are not accepted changes).
    ev=[];rounds=[]
    for x in st:
        path=A/x['variant']/f"s{x['seconds']}"/x['schedule'];events=load(path/'events_gt_audit.json')
        for rnd in x['rounds']:
            ee=[e for e in events if e['round']==rnd['round'] and e['action']=='reassigned']
            labelled=[e for e in ee if e['correct_after'] is not None]
            round_fields={('elapsed_seconds' if k=='seconds' else k):v for k,v in rnd.items()}
            rounds.append(dict(variant=x['variant'],seconds=x['seconds'],schedule=x['schedule'],**round_fields,
                gt_labelled_reassignments=len(labelled),gt_correct_after=sum(e['correct_after'] for e in labelled),
                gt_accuracy_after=None if not labelled else sum(e['correct_after'] for e in labelled)/len(labelled)))
    csvout(A/'round_comparison.csv',rounds)
    # Collect files as review artifacts, excluding full image set and five large raw embeddings.
    top=['experiment_comparison.csv','round_comparison.csv','algorithm_summary.json','stcra_summary.json','core_summary.json',
        'camera_map.json','frame_map.json','timestamp_audit.csv','homographies.json','sources.json','feature_alignment.json',
        'pose_report.json','yolov7_load_reports.json','anchor_cost_self_test.json']
    for name in top:cp(A/name,O/'01_summary'/name)
    for name in ['image_ids.txt','execution_notes.txt']:
        cp(A/name,O/'06_reproducibility'/name)
    for name in ['core.log','pose-sct.log','stcra.log','stcra-attempt1-shape-error.log','yolov7-load.log','evaluate.log']:
        if (A/name).exists():cp(A/name,O/'06_reproducibility/logs'/name)
    for name in ['sct_keypoints.npy','pose_requests.json']:cp(A/name,O/'02_intermediate'/name)
    for variant in ['original_frames','fps10']:
        for p in (A/variant).rglob('*'):
            if p.is_file():cp(p,O/'03_experiments'/variant/p.relative_to(A/variant))
    for p in (A/'evaluation').rglob('*'):
        if p.is_file():cp(p,O/'04_evaluation'/p.relative_to(A/'evaluation'))
    for p in (A/'instrumented').rglob('*'):
        if p.is_file():cp(p,O/'06_reproducibility/instrumented'/p.relative_to(A/'instrumented'))
    for name in ['uwipl-study-core.py','uwipl-study-pose.py','uwipl-study-stcra.py','evaluate-uwipl-study.py','uwipl-check-yolov7.py','report-uwipl-study.py','scout-sample-stage.py','package-uwipl-study.py']:
        cp(R/'scripts'/name,O/'06_reproducibility/scripts'/name)
    for p in (R/'docker/uwipl').iterdir():
        if p.is_file():cp(p,O/'06_reproducibility/docker'/p.name)
    for p in (R/'logs').glob('uwipl-*-gpu-20261025*.log'):cp(p,O/'06_reproducibility/reused_environment_logs'/p.name)
    for p in (R/'logs').glob('uwipl-*-pip-freeze.txt'):cp(p,O/'06_reproducibility/reused_environment_logs'/p.name)
    for name in ['report.json','run.log']:cp(A/'inference30/uwipl/reid'/name,O/'06_reproducibility/reid'/name)
    for name in ['report.json','run.log']:cp(A/'inference30/uwipl/detection'/name,O/'06_reproducibility/reused_detection'/name)
    cp(A/'inference30/reuse.json',O/'06_reproducibility/reused_detection/reuse.json')
    cp(R/'weights/uwipl/manifest.json',O/'06_reproducibility/weights_manifest.json')
    cp(R/'vendor/TrackEval/LICENSE',O/'06_reproducibility/TrackEval/LICENSE')
    for rel in ['trackeval/metrics/hota.py','trackeval/metrics/identity.py','trackeval/metrics/clear.py','trackeval/metrics/_base_metric.py','trackeval/_timing.py','trackeval/utils.py']:
        cp(R/'vendor/TrackEval'/rel,O/'06_reproducibility/TrackEval'/rel)
    parameters=[('실사 입력 경로','S001, 카메라7대','SCOUT sequence_1 cam_0~7 → S001/c001~c008','원본 AIC 보정 사용 안 함'),
        ('다수결: 논문/일반 경로','15프레임','15 → 5프레임','30fps의0.5초를10fps의5프레임으로'),
        ('다수결: S001 코드','200/300/150/300/300/500/150','직접 적용 안 함; 별도 기록','AIC 카메라별 수동값'),
        ('BoT-SORT 유지','fps=30, track_buffer=30','fps=10, track_buffer=30 → 실제10프레임','int(fps/30×buffer)'),
        ('짧은 track 제외','10개 관측','4개 관측','ceil(10×10/30); 정렬된 관측수 기준'),
        ('보간','n_min=5, n_dti=20','n_min=2, n_dti=7','실제 조건 n_frame>n_min, 1<gap<n_dti'),
        ('보간 위치 거리','200 image pixels','200 image pixels 유지','STCRA world 거리와 별개'),
        ('i-STCRA 거리','250/200/150 지도 단위','동일 숫자 / 2.5·2·1.5 후보','정확한 AIC→m 변환 아님'),
        ('i-STCRA 신뢰도','0.65/0.7/0.75','동일','3회 누적 실행'),
        ('smooth_avg_tracks','window_size=60','호출 안 함','원본 main도 호출 안 함'),
        ('카메라 가중치','AIC7대 수동값','8대 모두1.0','SCOUT 시야 면적 근거 없음')]
    csvout(O/'01_summary/parameter_changes.csv',[dict(zip(['parameter','upstream','scout','reason'],x)) for x in parameters])
    write(O/'START_HERE.md',f'''# UWIPL × SCOUT 알고리즘 검증 인계

`{STEM}.pdf`와 `01_summary/experiment_comparison.csv`를 먼저 보세요.

- 실행: sequence_1 / cam_0~7 / source277~326(약5초),277~576(약30초). 60초는 미실행.
- 원본 프레임값 프로필과10fps 보정 프로필, 원본 거리250/200/150과 후보2.5/2/1.5, 앵커→STCRA→i-STCRA→보간32개 결과 행을 기록했습니다. 앵커는 거리 설정 사이에서 동일 결과를 재사용합니다.
- 검출은 동일 YOLOX의 기존30초 결과를 재사용했습니다. OSNet5개는 새로 생성했고 HRNet-W48은 SCT 박스에 실행했습니다. HigherHRNet이라는 논문 명칭과 실제 배포 모델 차이를 기록했습니다.
- STCRA는 독립 단계로 원본 id_reassignment를 호출했습니다. H=K[r1 r2 T], 카메라8개, 균등 가중치. raw 발 좌표만 undistort한 뒤 H 역투영합니다. 원본 저장소는 변경하지 않았습니다.
- 2.5/2/1.5는 미터 좌표계 해석하의 후보 거리입니다. AIC 지도250/200/150을 정확히 환산한 값이 아닙니다. 단일 Z=0 평면은 SCOUT의 실제 비평면 지면을 근사하므로 결과를 보정 정확성 검증으로 쓰지 마세요.
- 30초 구간300개 frame index 중29개는 카메라간 시차가50ms를 넘고 최대2.051초입니다. 같은 index를 동시점으로 둔 진단 실험입니다. GT ID/좌표는 추론에 쓰지 않았고 평가에만 썼습니다.
- HOTA/IDF1_2D는 카메라별 표준2D TrackEval, IDSW는 카메라별 시간축 합계입니다. view_aware는 글로벌ID를 공유하되 카메라별 관측을 분리한 자체2D 프로토콜입니다. 공식 SCOUT3D 평가가 아닙니다.
- 평가 시 같은 camera/frame/id 중 점수가 가장 높은1개만 남겼습니다. 제거 내역과 원시 결과를 모두 보존했습니다. 보간 원본 코드의 중복 전체삭제는 별도 내역입니다.
- 앵커 비용 정답 여부와 재배정 정답률은 GT와IoU≥0.5 대응 후 앵커 구성원 다수 GT로 사후 판정합니다. 미대응은 정답률 분모에서 제외하며 분모를 제공합니다.

## 폴더

01_summary: 설정·매핑·표 / 02_intermediate: 실제SCT 포즈 / 03_experiments: 프로필·구간·거리·단계별 JSON·CSV·MOT·이벤트 / 04_evaluation: 평가방법·테스트 / 05_visualizations: 비교 그림24장 및 도표 / 06_reproducibility: 스크립트·diff·환경·가중치hash·로그.

전체 SCOUT, 대형 가중치, Docker 이미지와 5종 전체 임베딩은 서버에 보존하고 ZIP에는 포함하지 않았습니다. 별도 PC 재실행보다 검토·분석을 위한 자료입니다. ZIP의 FILES.csv와 SHA256SUMS.txt로 검증할 수 있습니다.
''')
    pages=[]
    pages.append(page(1,'요청 항목 실행 결과',f'''
<p class="lead">UWIPL의 <b>앵커 → STCRA → i-STCRA → 보간</b>을 SCOUT8카메라의5초·30초 구간에서 실행했습니다. 모델 간 순위를 매기기보다 입력 조건과 실제 재배정 동작을 확인했습니다.</p>
<div class="stats"><div><strong>2,400장</strong><span>30초 ×8대 / 5초는 앞50프레임</span></div><div><strong>37,023×2,560</strong><span>OSNet5종을 연결한 특징</span></div><div><strong>32개 결과 행</strong><span>시간2 × 구간2 × 거리2 × 단계4</span></div></div>
{table(['요청','결과'],[('가중치 로드','YOLOX·OSNet5종·HRNet-W48 정상. YOLOv7 5종도1장 추론 PASS'),('SCT 후 포즈','실제 추적 박스35042개에17관절 추론; 중복 계산은 재사용'),('H 및 STCRA 연결','SCOUT H를 S001의 c001~c008에 연결, STCRA 별도 호출'),('앵커 수','5초10개 / GT43명, 30초7개 / GT76명'),('식(2)~(5)','발 선택·투영오차·3대 이상 관측·이상치·재배정/삭제·GT 정답률 기록'),('단계별 지표','IDF1·IDSW·HOTA, 원시/평가용 출력 행 수 기록'),('기존 결과 보존','원본 코드·OSC 산출물 유지, 기존 동일 검출·환경 검증 재사용')])}
<div class="callout"><b>실행 성공과 개선은 다릅니다.</b> 10 FPS·5초의 거리 후보 설정에서 앵커→i-STCRA의 HOTA 2D는{pct(row(5,'anchor')['HOTA_2D'])}→{pct(row(5,'istcra')['HOTA_2D'])}, IDSW는{row(5,'anchor')['IDSW_2D']}→{row(5,'istcra')['IDSW_2D']}였습니다. 이상치를 더 찾았지만 추적 개선으로 이어지지는 않았습니다.</div>
<p class="small">SCOUT 지면은 Z=0 근사입니다. 2.5/2/1.5는 검증된 AIC지도→m 변환이 아닌 거리 후보입니다. 30초에는 카메라 시차가 커지는 구간이 있어5초와 분리해야 합니다.</p>
<p class="small">32개는 결과표의 행 수입니다. 동일 앵커 결과를 두 거리 설정에서 재사용하며, i-STCRA의 첫 회차가 STCRA 결과입니다. 모든 조합을 처음부터 중복 추론하지 않았습니다.</p>'''))
    pages.append(page(2,'실행 조건과 시간 단위 설정',f'''
<p>SCOUT의 cam_0~7을 담당자 표현의 카메라1~8로 대응했습니다. 원본277부터 내부frame0을 부여하고, 픽셀 크기는1920×1080을 유지했습니다.</p>
{table(['항목','원본/참조값','이번 적용'],[(x[0],x[1],x[2]) for x in parameters])}
<h2>원본과 수정 프로필의 의미</h2><p><b>original_frames</b>는 논문/일반 경로의 다수결15와 유지30·보간20을 실제10fps 영상에 그대로 적용한 대조군입니다. <b>fps10</b>은 각각5·10·7로 바꾼 프로필입니다. S001 수동 카메라 설정을 그대로 복제한 원본 재현은 아닙니다. 시간 프로필 비교는 여러 시간값을 함께 바꾸므로 개별 파라미터의 인과효과로 해석하지 않습니다.</p>
<h2>기존 결과 재사용 근거</h2><p>기존 OSC와 UWIPL의 YOLOX 가중치·experiment config·preprocessing이 같고 이전16프레임 출력도 정확히 일치했습니다. 검출30초를 재사용하고 UWIPL 고유 OSNet5종은 새로 계산했습니다. 이후SCT 박스에 포즈를 재실행했습니다.</p>
<p class="small">논문은 HigherHRNet이라고 서술하지만 공식 README·배포 설정·가중치는 top-down HRNet-W48입니다. 본 실행도 실제 배포 경로를 사용했습니다. 원본 source8e23a69 / TrackEval12c8791.</p>'''))
    pages.append(page(3,'앵커와 다수결 동작',f'''
{table(['10 FPS 프로필','5초','30초'],[(label,mainalg[5][key],mainalg[30][key]) for label,key in [('앵커 수','anchors'),('GT 고유 인원','gt_unique_people'),('앵커 후보에서 GT 대응된 인원','gt_people_covered_by_anchor_candidates'),('앵커 관측 수','anchor_observations'),('다수결로 ID가 바뀐 행','majority_changed')]])}
<h2>앵커가 적은 이유를 볼 수 있는 조건</h2><p>클립 내11개 시점을 균등 선택하고, 원본 S001의 박스면적&gt;10,000px²·score&gt;0.95 조건을 유지했습니다. 5개 특징을 연결한 raw Euclidean 거리80의 Ward clustering으로 앵커 수를 추정했습니다. GT 인원수를 클러스터 수로 주지 않았습니다.</p>
<img class="plot" style="width:100%" src="{pic(figs/'anchor_costs.png')}">
<p>배정 비용은 앵커 구성원과의 평균 cosine distance입니다. 원본 get_box_dist와 벡터화 계산의 실제 특징 소입력 값 일치를 확인했습니다. Hungarian 후보와 다수결 후ID를 함께 기록했습니다.</p>
<div class="callout">정답/오답은 추론 후에만 판단했습니다. IoU≥0.5로GT와 대응시키고, 각 앵커 구성원에 가장 많이 대응된GT ID를 해당 앵커의 의미로 사용합니다. GT 미대응 관측은 unknown입니다. 이 분포는 독립 평가세트의 보정 성능이 아닙니다.</div>
<p class="small">짧은 clip에서 AIC의 frame4999·9999 등을 조회하면 앵커가 없으므로 관측된11개 시점으로 바꿨습니다. 카메라별 ROI, 특정 인물8번 처리,7대 전용 배열도 SCOUT에 복사하지 않았습니다.</p>'''))
    wr=[]
    for sec in [5,30]:
        a=mainalg[sec];e=a['matched_world_error_dataset_units'];total=a['ankle_used']+a['bbox_fallback']
        wr.append([f'{sec}초',f"{a['ankle_used']}/{total} ({pct(a['ankle_used']/total)})",f"{a['bbox_fallback']}/{total}",e['count'],f"{e['mean']:.3f}",f"{e['median']:.3f}",f"{e['p95']:.3f}"])
    pages.append(page(4,'발 좌표 · 투영 · 거리 기준',f'''
<h2>식(2): 발목 또는 박스 아래쪽 선택</h2><p>양쪽 발목 score가 모두0.5보다 크면 두 발목 평균을 사용합니다. 그렇지 않으면 <code>(x+0.5w, y+0.95h)</code>로 대체합니다. 실제SCT 박스의 포즈로 계산했습니다.</p>
{table(['구간','발목 통과','박스 대체','GT대응','평균 m*','중앙 m*','p95 m*'],wr)}
<h2>식(3): SCOUT 보정값 적용</h2><p>각 카메라에 <code>H=K[r1 r2 T]</code>를 생성했습니다. 방향은 지면world XY→화면이며 역행렬로 위치를 구합니다. 원본 raw 영상의 왜곡을 고려해 선택한 발점만 undistort한 뒤 H⁻¹을 적용했습니다. BBox와 평가 영상은 raw 좌표를 유지합니다.</p>
<div class="callout">* SCOUT 논문은 mesh를 metric coordinate system으로 설명합니다. 표는1 dataset unit=1m로 해석한 수치이며 별도 실측으로 단위를 재검증하지 않았습니다. 오차는IoU로 대응된 사람GT의XY와 비교했습니다. Z=0 근사·가림·발점 추정 오류를 함께 포함합니다.</div>
<h2>두 거리 설정</h2>
{table(['설정','회차1/2/3','의미'],[('original_units','250 / 200 / 150','원본 숫자를 SCOUT 좌표에 그대로 적용'),('meter_candidate','2.5 / 2.0 / 1.5','미터 좌표 해석의 작은 거리 후보; GT로 최적화 안 함'),('confidence','0.65 / 0.70 / 0.75','양쪽 설정에서 동일')])}
<p>원본 AIC 지도 픽셀당 몇m인지 근거가 없어250→2.5가 정확한 환산이라고 주장할 수 없습니다. 두 결과는 <b>거리 민감도 실험</b>입니다. 비평면 지면을 단일H로 표현한 오차도 그대로 남습니다.</p>
<p class="small">평균 오차 분모는GT와IoU≥0.5로 대응된 앵커 출력 관측입니다. 미검출·미할당 관측까지 포함한 전체 위치 정확도는 아닙니다. CSV에는 각 행의GT ID·IoU·좌표·오차가 있습니다.</p>'''))
    rt=[]
    for x in rounds:
        if x['variant']!='fps10':continue
        rt.append([str(x['seconds'])+'s','원본' if x['schedule']=='original_units' else '후보',x['round'],x['outlier_observations'],x['actual_reassigned'],x['actual_deleted'],f"{x['gt_correct_after']}/{x['gt_labelled_reassignments']}",'-' if x['gt_accuracy_after'] is None else pct(x['gt_accuracy_after'])])
    pages.append(page(5,'STCRA 실제 재배정·삭제 내역',f'''
<p><b>run_tracking.py 밖의 별도 단계</b>로 STCRA/run_stcra.py의 id_reassignment를 호출했습니다. S001에SCOUT의8개H를 주입하고, AIC7대의 시야면적 가중치 대신 균등1.0을 사용했습니다. 출력은프로필/구간/거리별 폴더로 분리했습니다.</p>
{table(['구간','거리','회차','이상치','재배정','삭제','GT정답/판정','정답률'],rt)}
<h2>식(4)와 식(5)의 관찰 기준</h2><p>같은 ID·프레임에 3개 이상의 카메라 관측이 있어야 이상치 검사가 작동합니다. 해당 조건을 만족하는 ID가 하나 이상 있는 frame 비율과, 전체 ID-frame 그룹 중 3대 이상인 그룹 수를 회차별 CSV에 분리 기록했습니다. 이는 ID가 GT 기준으로 맞다는 뜻은 아닙니다. 식(5)의 후보 confidence와 실제 변경도 구분해, 원본 corrected_list뿐 아니라 동일 관측의 입력·출력을 비교했습니다.</p>
<p>첫 회차의 3대 이상 관측 비율은 5초50/50프레임, 30초300/300프레임으로 모두100%입니다. ID-frame 그룹도 각각500/500,2100/2100개가 조건을 만족했습니다. 적은 수의 앵커 ID에 여러 사람이 섞여도 이 조건을 통과할 수 있으므로, 실제 동일인 관측 여부와 구분해야 합니다.</p>
<p class="small">삭제에는 중복 억제와 재배정 실패 후 남지 않은 관측이 포함됩니다. 보간의 중복 ID 전체 삭제는 interpolation_changes.json에 별도로 기록했습니다.</p>
<div class="callout">정답률은 GT 판정 가능한 실제 재배정 중 새ID의 앵커 대표GT와 해당관측GT가 일치한 비율입니다. unknown은 분모에서 제외합니다. 재배정0건 또는 판정0건은0%가 아닌 미산출로 표시합니다.</div>'''))
    for pi,sec in [(6,5),(7,30)]:
        rs=[]
        for schedule in ['original_units','meter_candidate']:
          for stage in ['anchor','stcra','istcra','interpolation']:
            x=row(sec,stage,schedule);rs.append(['원본' if schedule=='original_units' else '후보',stage,pct(x['IDF1_2D']),x['IDSW_2D'],pct(x['HOTA_2D']),pct(x['IDF1_view_aware']),x['raw_rows'],x['duplicate_id_rows_removed_for_eval']])
        caveat='50개 frame index 모두 최대카메라 시차50ms 이하이며 전체최대0.032초입니다.' if sec==5 else '<b>300개 index 중29개에서50ms 초과, 최대2.051초의 카메라시차가 있습니다.</b> 같은번호를동시점으로 두고 실행한 진단결과이며, 공간재배정 성능의정식 검증으로 사용할 수 없습니다.'
        pages.append(page(pi,f'{sec}초 단계별 비교',f'''
<p>{'원본277~326, 카메라당50장' if sec==5 else '원본277~576, 카메라당300장'} / 아래는fps10 시간프로필입니다. {caveat}</p>
{table(['거리','단계','IDF1 2D','IDSW','HOTA 2D','IDF1 view','원시행','평가중복제외'],rs)}
<img style="width:100%" src="{pic(figs/'ablation_idf1.png')}">
<h2>지표 해석</h2><p>HOTA·IDF1 2D는 카메라별 시퀀스를 공식TrackEval로 계산한 뒤 결합했습니다. IDSW는 카메라별 시간축 전환의합계입니다. IDF1 view는 동일Global ID를 공유하면서 서로다른카메라의 박스끼리 매칭되지 않게한 관측단위 진단프로토콜입니다.</p>
<p class="small">동일 camera/frame/id가 여러행이면 평가에서최고점수1개를남겼고 제거행을별도기록했습니다. 원시출력은보존했습니다. HOTA는IoU0.05~0.95, IDF1·IDSW는IoU0.5 기준입니다. 공식SCOUT3D benchmark나 다른모델과의순위표가 아닙니다.</p>'''))
    cmp=[]
    for sec in [5,30]:
      for stage in ['anchor','stcra','istcra','interpolation']:
        b=row(sec,stage,variant='original_frames');a=row(sec,stage)
        cmp.append([f'{sec}s',stage,pct(b['IDF1_view_aware']),pct(a['IDF1_view_aware']),b['IDSW_2D'],a['IDSW_2D']])
    pages.append(page(8,'시간 설정 변화와 대표 화면',f'''
{table(['구간','단계','원본프레임 IDF1 view','10fps IDF1 view','원본 IDSW','10fps IDSW'],cmp)}
<p class="small">거리후보2.5/2/1.5를 고정했습니다. 시간값을 묶어바꾼 비교이며개별파라미터 ablation이 아닙니다. 원본다수결은논문/일반경로15를뜻합니다. S001 수동150~500은parameter_changes.csv에별도보존했습니다.</p>
<img style="width:100%" src="{pic(figs/'cam_0_frame277_before_after.jpg')}">
<img style="width:100%" src="{pic(figs/'cam_5_frame301_before_after.jpg')}">
<p>왼쪽은 앵커 배정, 오른쪽은 작은 거리후보의i-STCRA+보간입니다. 색과숫자는예측Global ID이며GT가아닙니다. 8카메라에서3개시점씩총24개 비교그림을첨부했습니다.</p>'''))
    pages.append(page(9,'전달 자료와 다음 판단',f'''
{table(['폴더','내용'],[('01_summary','32행 지표표, 회차별통계, 카메라·프레임매핑, H, 설정변경'),('02_intermediate','35042개 SCT박스의 Pose17×3 및행대응'),('03_experiments','앵커구성원/비용, feet16열, 라운드별JSON, 재배정이벤트, GT오차, 단계별CSV·MOT'),('04_evaluation','평가프로토콜·의미있는소입력검증'),('05_visualizations','비용분포, 단계별지표, 전후비교24장'),('06_reproducibility','환경재사용로그, 가중치목록, 코드·설정·변경diff·실패수정로그')])}
<h2>원본에서 일반화한 부분</h2><p>7대전용값·특정인물8번 처리·c005 ROI를제거하고8대를균등처리했습니다. 관측된11개시점으로앵커를만들고, 특징행의시점대응과마지막frame Hungarian 처리를명시했습니다. 원본run_stcra.py의계산함수를사용했고cam_weight만8대로수정한diff를보존했습니다. 초기array/list입력형식오류는수정후재실행했고실패로그도첨부했습니다.</p>
<h2>다음 단계에서 우선 검토할 사항</h2><ol><li>앵커 후보의크기·신뢰도조건과raw거리80이먼거리SCOUT인물을누락시키는지 분석합니다. GT인원수를정답으로넣어앵커수를맞추지는않습니다.</li><li>동기화가확인된시점으로30초구간을재구성하고, 비평면mesh기반위치와단일H를별도비교합니다.</li><li>AIC지도단위와미터사이의근거를확보하거나SCOUT용거리기준을독립개발구간에서정합니다.</li><li>평가전중복ID처리정책과공식SCOUT3D프로토콜을확정한후장기성능을측정합니다.</li></ol>
<p class="small">참조: UWIPL 논문 arxiv.org/abs/2304.09471 및공식저장소8e23a69. SCOUT metric좌표/동기화 설명: arxiv.org/html/2507.08494v1 §4. TrackEval:12c8791b303e0a0b50f753af204249e622d0281a. 전체데이터·대형가중치·전체임베딩은서버에보존하고ZIP에는검토자료를담았습니다.</p>'''))
    tree=ast.parse((R/'scripts/create-osc-handoff.py').read_text())
    css=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='css' for t in n.targets))
    css+='\ntable{font-size:8pt} td{word-break:keep-all;overflow-wrap:anywhere} .page{overflow:hidden} h1{font-size:22pt} ol{padding-left:6mm}\n'
    write(R/'docs/handoff'/f'{STEM}.html','<!DOCTYPE html><html lang="ko"><meta charset="utf-8"><title>UWIPL SCOUT 알고리즘 검증</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>')
    print('Report and handoff prepared:',O)

if __name__=='__main__':main()
