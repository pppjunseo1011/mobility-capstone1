#!/usr/bin/env python3
"""Create a readable partial-stage handoff from executed PoseTrack audits."""
import ast
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/posetrack-transfer-20261101'
INF=RUN/'inference'
OUT=RUN/'handoff'
STEM='PoseTrack_SCOUT_실행검증_인계서_20261101'

def load(p): return json.loads(p.read_text())
def write(p,s): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def js(p,s): write(p,json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cp(a,b): b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def pic(p):return 'data:image/'+('png' if p.suffix=='.png' else 'jpeg')+';base64,'+base64.b64encode(p.read_bytes()).decode()
def page(n,title,body):return f'<section class="page"><div class="eyebrow">PoseTrack × SCOUT · 담당자 요청서 v3 회신</div><h1>{title}</h1>{body}<footer><span>2026.11.01 · 실행 준비 및 모듈 검증 · 성능 벤치마크 아님</span><b>{n} / 7</b></footer></section>'

def main():
    audit=load(OUT/'02_adapter/audit_summary.json')
    alignment=csvread(OUT/'03_intermediate/alignment_audit.csv')
    cal=csvread(OUT/'02_adapter/calibration_audit.csv')
    assert load(INF/'summary.json')['status']=='PASS'
    assert audit['module_alignment']=='PASS' and audit['native_tracking_readiness']=='BLOCKED'
    assert (OUT/'01_environment/environment_smoke.exit').read_text().strip()=='0'
    assert (OUT/'01_environment/pip_check.exit').read_text().strip()=='0'
    lup=load(OUT/'01_environment/lup_load.json')
    assert lup['status']=='PASS'
    # Evidence is copied, not replaced by prose summaries.
    for stage in ['detection','pose','reid']:
        for name in ['report.json','run.log']:
            cp(INF/f'posetrack/{stage}/{name}',OUT/f'03_intermediate/logs/{stage}_{name}')
    for src,dst in [('posetrack/detection/detections.json','03_intermediate/detections.json'),
                    ('posetrack/pose/poses.json','03_intermediate/poses.json'),
                    ('posetrack/reid/aic24-rows.json','03_intermediate/embedding_rows.json'),
                    ('summary.json','06_reproducibility/inference_summary.json'),
                    ('selection.json','06_reproducibility/selection.json'),
                    ('input/manifest.json','06_reproducibility/input_manifest.json')]:cp(INF/src,OUT/dst)
    for p in (INF/'runner-history').rglob('*'):
        if p.is_file():cp(p,OUT/'06_reproducibility/runner-history'/p.relative_to(INF/'runner-history'))
    for name in ['infer-scout-sample.py','scout-sample-stage.py','audit-posetrack-transfer.py',
                 'check-posetrack-lup.py','create-posetrack-transfer-report.py','download-posetrack-weights.py',
                 'render-posetrack-transfer-videos.py','package-posetrack-transfer.py']:
        cp(ROOT/'scripts'/name,OUT/'06_reproducibility/scripts'/name)
    for p in (ROOT/'docker/posetrack').iterdir():
        if p.is_file():cp(p,OUT/'06_reproducibility/docker/posetrack'/p.name)
    for p in (ROOT/'adapters/scout_adapter').rglob('*.py'):
        cp(p,OUT/'06_reproducibility/adapters/scout_adapter'/p.relative_to(ROOT/'adapters/scout_adapter'))
    cp(ROOT/'sources.lock.json',OUT/'06_reproducibility/sources.lock.json')
    write(OUT/'06_reproducibility/upstream_changes.diff',subprocess.check_output(['git','diff'],cwd=ROOT/'repos/posetrack',text=True))
    # Independent hash recomputation of all four weights.
    weights=load(ROOT/'weights/posetrack/manifest.json')['files']+[load(ROOT/'weights/posetrack/lup-manifest.json')]
    wr=[]
    for w in weights:
        f=ROOT/'weights/posetrack'/w['filename'];digest=sha(f)
        assert digest==w['sha256'] and f.stat().st_size==w['size_bytes']
        wr.append({'filename':f.name,'server_path':str(f),'bytes':f.stat().st_size,'sha256':digest,
                   'source_url':w['source_url'],'load_status':'PASS',
                   'use':'training_initializer_only' if f.name.startswith('lup') else 'inference',
                   'publisher_checksum_verified':False})
    csvwrite(OUT/'01_environment/weights_manifest.csv',wr)
    status=[('서버·Docker','PASS','01_environment/environment_host.txt','GPU 1 / 기존 이미지 재사용'),
        ('CUDA·확장·Solver import 및 작은 연산','PASS','01_environment/environment_smoke.log','Torch/CUDA, aic_cpp bbox 연산, GLPK, FAISS'),
        ('pip check','PASS','01_environment/pip_check.txt','의존성 충돌 없음'),
        ('공식 가중치 4종','PASS','01_environment/weights_manifest.csv','3종 추론 + LUP 학습 초기값 로드'),
        ('YOLOX 실제 추론','PASS','03_intermediate/logs/detection_report.json','400장 / 원시 6412 / 선택 5569'),
        ('HRNet 실제 추론','PASS','03_intermediate/logs/pose_report.json','5569 × 17 × 3'),
        ('MGN ReID 실제 추론','PASS','03_intermediate/logs/reid_report.json','5569 × 2048 / OSC 특징 사용 안 함'),
        ('카메라·프레임·시간 매핑','PASS','02_adapter/frame_map.json','원본277~326 → 내부0~49 / 최대 시차0.032초'),
        ('동일 검출 행 정합성','PASS','03_intermediate/alignment_audit.csv','행·BBox·관절·특징·유한값 검사'),
        ('물리적 P/H 보정 검증','BLOCKED','02_adapter/calibration_audit.csv','왜곡 좌표·비평면 지면·물리 단위 검증 필요'),
        ('1~3카메라 / 8카메라 Tracking','BLOCKED','04_tracking/status.json','검증된 calibration.json 필요'),
        ('Geometry·Matching·State·Bank 실제 추적 로그','BLOCKED','05_analysis/requested_artifacts.csv','추적 미실행; 분기 미발생과 구분')]
    csvwrite(OUT/'00_preflight_status.csv',[dict(zip(['item','status','evidence','detail'],x)) for x in status])
    blocked={'status':'BLOCKED','tracking_executed':False,'global_id_output_created':False,
        'reason':audit['blockers'],'not_observed_used':False,
        'next_step':'Undistort image/keypoint/bbox coordinates consistently; validate common metric ground representation, then run 1 camera and 2-3 cameras with instrumented tracker.'}
    js(OUT/'04_tracking/status.json',blocked)
    tracefiles=['geometry_trace.csv','affinity_snapshot.npz','matching_trace.csv','hungarian_trace.json',
                'state_history.csv','reid_activation_trace.csv','feature_bank_trace.csv','target_event_trace.csv']
    csvwrite(OUT/'05_analysis/requested_artifacts.csv',[{'requested_file':f,'status':'BLOCKED',
        'file_created':False,'reason':'Verified native P/H not available; tracker not executed. No fabricated events or empty placeholder arrays.'} for f in tracefiles])
    # Actual input prerequisite counts, explicitly not runtime state activation.
    poses=load(INF/'posetrack/pose/poses.json');gates=[]
    for c in range(8):
        ps=[p for p in poses if p['camera']==f'cam_{c}']
        b=np.concatenate([np.array(p['bbox_xyxy_score']).reshape(-1,5) for p in ps])
        k=np.concatenate([np.array(p['keypoints']).reshape(-1,17,3) for p in ps])
        gates.append({'camera':f'cam_{c}','detections':len(b),'confidence_ge_088':int((b[:,4]>=.88).sum()),
            'confidence_gt_09':int((b[:,4]>.9).sum()),'feet_both_gt_07':int((k[:,[15,16],2]>.7).all(1).sum()),
            'bank_input_quality_only':int(((b[:,4]>.9)&(k[:,[5,6,11,12],2]>.5).all(1)).sum()),
            'single_init_input_quality_only':int(((b[:,4]>.9)&(k[:,[5,6,11,12,13,14,15,16],2]>.5).all(1)).sum()),
            'meaning':'Input predicates only; no history, overlap, matching or bank-event decision.'})
    csvwrite(OUT/'05_analysis/input_gate_counts.csv',gates)
    flow=[
        ('Camera.__init__','track/util/camera.py',12,'P=world→undistorted image; H=ground XY→image; Hfeet[:,-1]+=0.15*P[:,2]','왜곡 없는 픽셀·공통 좌표계·단위 필요'),
        ('PoseTracker.compute_bboxiou_aff','track/Tracker/PoseTracker.py',625,'Kalman predicted bbox; aff_box=(IoU-0.5)*(age<=15)*detection_score','IoU, bbox age, score 원값 수집'),
        ('PoseTracker.compute_epi_homo_aff','track/Tracker/PoseTracker.py',544,'aic_cpp.loop_t_homo_full(...) followed immediately by continue','이후 Python 대체 루프는 실행 안 됨'),
        ('loop_t_homo_full','track/aic_cpp/src/aic.cpp',423,'skip same view or age_bbox>=2; joint scores>0.7, age_2D<=1; weighted 1-distance/0.2; feet or box 1-distance/1.5','C++ 내부값 로깅 필요; 환경 smoke는 이 함수의 실제 SCOUT 실행 아님'),
        ('PoseTracker.mv_update_wo_pred','track/Tracker/PoseTracker.py',842,'epi/homo clamp>=-2.5; weighted sum (1,5,1,5)/(1+reid_weight); norm penalty; negative→0; Hungarian(-aff_final)','선택됐어도 aff_final<=0이면 update 거부'),
        ('unmatched per-view state','track/Tracker/PoseTracker.py',896,'unselected target: colmax IoU>0.5 → Occluded else Missing; history max10','전체 TrackState와 별도; Hungarian-selected zero-affinity pair도 구분'),
        ('PoseTracker.compute_reid_aff','track/Tracker/PoseTracker.py',463,'score>=0.88 + nonempty Occluded/Missing history + positive bank cosine; weight=1, score=max(cos)-0.5','기본 weight=1e-5; confidence 통과만으로 활성화가 아님'),
        ('PoseTrack.multi_view_3D_update','track/Tracker/PoseTracker.py',392,'upper joints[5,6,11,12]>0.5; bbox>0.9; count(IoU>0.15)<2; count(overlap>0.3)<2; empty bank or maxsim<0.6','feat_count 및 index modulo bank_size 전후 기록'),
        ('PoseTracker.target_init','track/Tracker/PoseTracker.py',725,'cross-view GLPK on 2*epi+homo; quality/overlap gates; single_view_init or multi_view_init','reid_sim_temp는 계산하지만 초기 그룹 affinity에는 더하지 않음'),
        ('PoseTracker.match_with_miss_tracks','track/Tracker/PoseTracker.py',692,'missing bank max cosine>0.5 → reactivate; otherwise append new','새 ID 후보와 최종 재활성 ID 분리'),
        ('overall state/output','track/Tracker/PoseTracker.py',956,'Unconfirmed unmatched→Deleted; update_age>=15→Missing; output Confirmed & update_age==1','17/10열 입력과 출력10열 camera/id/frame/bbox/worldXY/source 구분'),
        ('static candidate bug A','track/Tracker/PoseTracker.py',767,'target_init fallback uses (bbox[0]+bbox[0])/2, not (x1+x2)/2','정적 확인만; 수정·성능 영향 실험 안 함'),
        ('static candidate bug B','track/Tracker/PoseTracker.py',789,'sample_b box fallback uses homo_feet_inv while sample_a uses homo_inv','좌우 fallback 높이 불일치; 실제 발생 빈도 미측정'),
        ('bank-size caveat','track/Tracker/PoseTracker.py',65,'PoseTrack.bank_size=100; PoseTracker.bank_size=30 (line459), missing-match slicing uses tracker bank_size','장기 실행 bank slice 검토 필요; 재현된 실패로 단정하지 않음')]
    csvwrite(OUT/'05_analysis/code_flow.csv',[{'function':f,'file':p,'line':n,'condition':c,'interpretation':d,
        'evidence_kind':'STATIC_SOURCE_REVIEW','runtime_status':'BLOCKED'} for f,p,n,c,d in flow])
    write(OUT/'05_analysis/READ_ME.txt','code_flow.csv는 원본 소스 정적 분석입니다. 실제 Global ID 결정 trace가 아닙니다.\ninput_gate_counts.csv는 실제 추론 입력의 일부 품질 조건만 계산했습니다. 상태 이력·IoU·bank 중복 판정은 포함하지 않습니다.\nNOT_OBSERVED는 추적을 실행하고도 이벤트가 없을 때 사용합니다. 이번에는 추적을 시작하지 않았으므로 BLOCKED입니다.\n')
    # Representative detection/pose stills and all-frame pose videos are separate.
    for c in range(8):
        for i in [0,25,49]:
            for stage in ['detection','pose']:
                cp(INF/f'posetrack/{stage}/cam_{c}_{i:06d}.jpg',OUT/f'03_intermediate/visualizations/{stage}/cam_{c}_source_{277+i}.jpg')
    figdir=OUT/'03_intermediate/visualizations';figdir.mkdir(parents=True,exist_ok=True)
    rows=load(INF/'posetrack/reid/aic24-rows.json')
    ft=np.load(INF/'posetrack/reid/aic24.npy',allow_pickle=False)
    ii=[i for i,r in enumerate(rows) if r['source_frame']==277 and r['camera'] in ['cam_0','cam_1']]
    sub=ft[ii];sub=sub/np.linalg.norm(sub,axis=1,keepdims=True);cos=sub@sub.T
    np.savez_compressed(OUT/'03_intermediate/reid_cosine_example.npz',cosine=cos,embedding_rows=np.array(ii))
    labels=[rows[i]['camera']+':'+str(rows[i]['box_index']) for i in ii]
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(cos,vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(len(labels)));ax.set_yticks(range(len(labels)));ax.set_xticklabels(labels,rotation=90,fontsize=7);ax.set_yticklabels(labels,fontsize=7)
    ax.set_title('Actual PoseTrack ReID cosine / source frame 277\ncam: detection index; NOT Global ID or GT matching',fontsize=11)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'reid_cosine.png',dpi=140);plt.close(fig)
    contact=Image.new('RGB',(1440,870),'white');draw=ImageDraw.Draw(contact)
    for k,c in enumerate([0,3,5,7]):
        im=Image.open(INF/f'posetrack/pose/cam_{c}_000000.jpg');im.thumbnail((720,405))
        x=(k%2)*720;y=(k//2)*435;contact.paste(im,(x,y+25));draw.text((x+8,y+5),f'cam_{c} | source 277 | BBox + HRNet-W32 pose (no ID)',fill='black')
    contact.save(figdir/'pose_contact.jpg',quality=92)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(8)
    axes[0].bar(x,[float(r['distortion_shift_px_p95']) for r in cal],color='#267c98');axes[0].set_title('Raw vs undistorted pixel shift (p95)');axes[0].set_ylabel('pixels')
    axes[1].bar(x-.18,[float(r['z0_vs_mesh_world_error_p95']) for r in cal],.36,label='z=0 plane',color='#8eabb8')
    axes[1].bar(x+.18,[float(r['pca_vs_mesh_world_error_p95']) for r in cal],.36,label='mesh PCA plane',color='#ba684a')
    axes[1].set_title('Plane-ray vs mesh-ray distance (p95)');axes[1].set_ylabel('dataset units (not verified meters)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x);ax.set_xticklabels([f'cam_{i}' for i in x],rotation=30);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'calibration_diagnostic.png',dpi=150);plt.close(fig)
    # Concrete run record: no native tracker launch is implied.
    command='python3 scripts/infer-scout-sample.py --data datasets/scout/extracted/scout --adapter-export outputs/scout-first8-20s-v03-20261027/sequence1-posetrack --output outputs/posetrack-transfer-20261101/inference --models posetrack --gpu-posetrack 1 --gpu-others 0 --frames-per-camera 50 --max-people-per-frame 0'
    write(OUT/'06_reproducibility/commands.txt',f'''Working directory: {ROOT}
Image: scout-mtmc/posetrack-all:ba727e6
Image ID: sha256:da99db46dd7d4cfbbd936899c7f3941f065df1725132d62417d0d5194d05bf3d
Upstream master commit: ba727e6bc6e45e44cd6e006b73e00864892173c2

Executed host inference orchestrator:
{command}

Executed in the existing image (mount project /project read-only, SCOUT /data read-only, handoff /out read-write):
python /opt/scout/smoke.py --device cuda
python -m pip check
python -m pip freeze
python /project/scripts/audit-posetrack-transfer.py --data /data --inference /project/outputs/posetrack-transfer-20261101/inference --out /out
python /check.py   # mounted scripts/check-posetrack-lup.py; /weights read-only; /out=01_environment

No upstream source edits, no native tracker execution, no training, no metrics/ablation.
Module runner uses existing JPEGs and original model APIs rather than original hardcoded shell/video loops.
Every selected detection is stably sorted by objectness, identically for Pose/ReID; threshold>=0.3, cap=None.
Raw detector filter objectness>0.1, NMS0.7, exp.test_conf0.001, input800x1440.
Native output arrays retain raw features; run_tracking_batch.py L2-normalizes on read.
Full SCOUT, large checkpoints and Docker layers are not bundled. Input manifest includes source image checksums.
''')
    js(OUT/'06_reproducibility/config.json',{'sequence':'sequence_1','cameras':[f'cam_{i}' for i in range(8)],
        'source_start_inclusive':277,'source_end_inclusive':326,'internal_start':0,'internal_end':49,
        'nominal_fps':10,'actual_span_seconds':[4.9,4.9],'no_artificial_topk':True,'gpu':1,
        'gt_used':False,'native_tracking_run':False,'report_scope':'actual inference modules + calibration audit + static source review'})
    # Actual timestamp spans override nominal descriptions.
    fm=load(OUT/'02_adapter/frame_map.json');conf=load(OUT/'06_reproducibility/config.json')
    conf['actual_span_seconds']={f'cam_{i}':max(x['timestamp_seconds'] for x in fm if x['camera']==f'cam_{i}')-min(x['timestamp_seconds'] for x in fm if x['camera']==f'cam_{i}') for i in range(8)}
    js(OUT/'06_reproducibility/config.json',conf)
    write(OUT/'START_HERE.md',f'''# PoseTrack × SCOUT 담당자 인계 · 2026-11-01

**완료: 환경·공식 가중치 4종·400장 검출/Pose/ReID 추론·5569행 대응 검사. 미완료: 검증된 P/H, Global Tracking, 실제 ID 결정 trace.**

1. 동봉 PDF `{STEM}.pdf`를 먼저 읽어 주세요.
2. `00_preflight_status.csv`에서 PASS/BLOCKED와 근거 파일을 확인하세요.
3. `03_intermediate/result/`에 원본 추적기의 행 구조에 맞춘 카메라별 Detection 7열, Pose 57열, ReID N×2048을 넣었습니다. 같은 행이 같은 검출입니다. raw features이며 원본 loader가 L2 정규화합니다.
4. `02_adapter/frame_map.json`: sequence_1, cam_0~7, 원본277~326 → 내부0~49. 사람 수 제한 없음, objectness≥0.3은 유지. 각 카메라 50장, 최대 시차0.032초.
5. `02_adapter/calibration_audit.csv`: 실제 검출 하단 665개를 표본으로 검사했습니다. 메시 교차597개. 단일 평면 오차와 왜곡 때문에 native calibration.json은 만들지 않았습니다. `candidate_matrices_DO_NOT_USE`는 수학적 진단용으로 추적에 넣으면 안 됩니다.
6. `04_tracking/status.json` 및 `05_analysis/requested_artifacts.csv`: 실제 추적을 시작하지 않았으므로 요청한 8종 trace는 BLOCKED입니다. 임의 ID·가짜 로그·빈 affinity를 만들지 않았습니다.
7. `05_analysis/code_flow.csv`는 정적 코드 분석이며 실제 분기 실행 결과가 아닙니다. `input_gate_counts.csv`는 실제 입력의 일부 품질 조건만 계산했습니다.
8. `06_reproducibility/`에는 실행 명령, 설정, 소스 커밋, 원본 변경 diff(없음), 사용 runner와 검사 스크립트가 있습니다.

## 담당자에게 필요한 다음 결정

- raw 픽셀 왜곡을 제거하고 BBox/관절/투영행렬을 같은 좌표계로 맞출지 결정합니다.
- 메시 기반 지면 연산을 적용할지, 검증된 평면 ROI만 허용할지 정합니다. 카메라별 임의 H를 만들면 공통 world 좌표가 보장되지 않습니다.
- 물리 단위를 확인한 후 feet 높이0.15, epipolar0.2, homography1.5 임계값을 검토합니다.
- 이후 1카메라→2~3카메라 추적을 실행하여 Hungarian 선택/실제 update, 상태, bank, new/reactivate를 계측합니다.

## 가중치 및 변경 범위

YOLOX·HRNet-W32·aic24.pkl은 실제 추론에 사용했고 LUP는 학습 초기값입니다. LUP 공식 파일(170531587 bytes)을 추가 확보했으며 실제 `reid_train.sh`의 R101_reprod 설정과520개 키가 맞았습니다. 대안 IBN 설정은 키가 달라 실패했지만 사용 설정이 아니며 `lup_ibn_alternative_probe.json`에 보존했습니다. 기존 모델·OSC 산출물·가중치는 덮어쓰지 않았습니다.

본 묶음은 담당자 검토용입니다. SCOUT 전체 데이터·대형 가중치·Docker 이미지는 포함하지 않습니다. SHA256SUMS.txt와 FILES.csv로 무결성을 검사할 수 있습니다.
''')
    # Seven-page Korean report with bounded content.
    pages=[]
    pages.append(page(1,'실행 결과와 현재 도달 단계',f'''
<p class="lead">PoseTrack 전용 환경에서 <b>8카메라 · 400프레임</b>을 검출 → 자세 추정 → ReID 특징 추출까지 실행했습니다. <b>Global ID 추적은 보정 검증 단계에서 보류</b>했습니다.</p>
<div class="stats"><div><strong>5,569개</strong><span>검출–관절–특징 대응 통과</span></div><div><strong>2,048차원</strong><span>PoseTrack 고유 ReID 특징</span></div><div><strong>BLOCKED</strong><span>검증된 P/H 및 추적 trace</span></div></div>
<div class="callout">요청서 v3의 원칙에 따라 형상만 맞춘 보정행렬이나 GT 기반 임의 ID를 만들지 않았습니다. 이번 결과는 실제 모듈 추론 성공과 원인 확인까지이며, 전체 추적 성공 또는 성능 검증을 뜻하지 않습니다.</div>
{table(['항목','상태','결과'],[(x[0],x[1],x[3]) for x in status])}
<p class="small">환경 smoke의 PASS와 실제 모델 추론 PASS는 별도 로그로 보존했습니다. HOTA·AssA·IDF1, 모델 순위와 ablation은 이번 요청 범위에 포함하지 않았습니다.</p>'''))
    pages.append(page(2,'환경 · 가중치 · 실행 범위',f'''
<h2>기존 PoseTrack 전용 Docker 재사용</h2>
{table(['구분','확인 결과'],[('GPU','NVIDIA TITAN RTX 24GB × 2 / 본 실행 GPU 1'),('모델 환경','Python3.9.23 / torch2.0.1+cu118 / torchvision0.15.2+cu118'),('주요 패키지','mmcv2.0.1 / mmpose1.3.1 / mmengine0.10.4 / numpy1.23.5'),('확장 검사','aic_cpp·GLPK·lap·FAISS·torchvision/MMCV 작은 연산 PASS'),('소스','PoseTrack master / ba727e6bc6e45e44cd6e006b73e00864892173c2'),('이미지','scout-mtmc/posetrack-all:ba727e6')])}
<h2>공식 가중치 4종 확인</h2>
{table(['가중치','역할','이번 확인'],[('bytetrack_x_mot17.pth.tar','YOLOX 검출','894개 키 일치 +400장 추론'),('HRNet-W32 …81c58e40.pth','17개 관절','1756개 키 일치 +5569명 추론'),('aic24.pkl','MGN ReID 추론','직렬화 모델 로드 +5569×2048 특징'),('lup_moco_r101.pth','학습 초기화','공식 파일 추가 확보 / R101_reprod 520개 키 일치')])}
<p>각 파일의 크기·SHA-256·공식 URL·서버 경로를 함께 제공합니다. SHA-256은 서버 파일을 식별하는 자체 계산값이며 배포자가 제공한 공식 체크섬과 대조한 값은 아닙니다.</p>
<div class="callout">LUP는 aic24.pkl 추론의 대체 가중치가 아닙니다. 별도 IBN 설정으로 검사하면 키가 맞지 않았고, 실제 학습 스크립트의 R101_reprod 설정에서는 모두 일치했습니다. 학습은 시작하지 않았습니다.</div>
<h2>실행 방식 및 범위</h2><p>원본 모델 API와 가중치를 기존 JPEG 프레임에 적용했습니다. 원본 shell의 고정 GPU·경로·video loop를 그대로 실행한 결과는 아닙니다. 원본 저장소는 수정하지 않았으며 별도 runner 및 검사 스크립트를 첨부했습니다.</p>'''))
    pages.append(page(3,'입력·중간 출력 대응 검사',f'''
<p>sequence_1 / cam_0~7 / 원본 <b>277~326</b>을 사용했습니다. 내부 frame은 <b>0~49</b>, 카메라 폴더는 camera_0001~0008입니다. nominal10fps의 약5초 구간이며 첫·끝 timestamp 차이는 약4.9초입니다.</p>
{table(['SCOUT → PoseTrack','프레임','검출=Pose=ReID 행','최대 인원/장'],[(r['camera']+' → '+r['native_camera'],r['frames'],r['rows'],r['max_people_per_frame']) for r in alignment])}
<h2>같은 행이 같은 검출인지 확인</h2><p>7열 Detection, 57열 Pose 및2048차원 특징을 카메라·내부 frame·검출 순번으로 연결했습니다. 5,569행의 BBox, 관절 배열, 특징 배열 순서와 유한값 검사를 통과했습니다. 프레임당 상위10명 제한은 없으며 최대24명까지 처리됐습니다.</p>
{table(['형식','구성'],[('Detection 7열','frame, class=1, x1,y1,x2,y2,objectness'),('Pose 57열','frame + BBox5 + COCO17개 (x,y,score)'),('ReID N×2048','검출 행 순서의 raw 특징; 원본 loader에서 L2 정규화'),('row_correspondence.csv','source frame ↔ 내부 frame ↔ native행 ↔ embedding행')])}
<h2>필터와 동기화</h2><p>검출 objectness&gt;0.1 이후6,412개, 후속 입력 objectness≥0.3 이후5,569개입니다. NMS0.7, 검출 입력800×1440, 정수 BBox 및 영상 범위 clipping을 기록했습니다. 같은 내부 프레임의 카메라 간 최대 timestamp 차이는0.032초입니다.</p>
<p class="small">GT 사람 ID·세계 좌표를 추론 입력으로 사용하지 않았습니다. 03_intermediate/result는 행 형식 검증을 마친 모듈 출력이며 검증된 calibration.json 없이 바로 추적할 수 있는 완성 데이터셋은 아닙니다.</p>'''))
    pages.append(page(4,'실제 출력 시각화',f'''
<p>아래는 원본 frame277의 검출과 HRNet-W32 관절입니다. BBox 옆 숫자는 검출 순번이며 Local/Global ID가 아닙니다.</p>
<img style="width:100%;max-height:90mm;object-fit:contain" src="{pic(figdir/'pose_contact.jpg')}">
<h2>전달하는 시각화</h2><p>카메라별 frame277·302·326의 검출/Pose48장, Pose 연속 영상8개, 실제 ReID cosine 행렬을 제공합니다. 영상은 전체50프레임을10fps로 표시합니다.</p>
<img style="width:65%;max-height:70mm;object-fit:contain;display:block;margin:auto" src="{pic(figdir/'reid_cosine.png')}">
<p class="small">ReID cosine은 동일인물 확정값이 아닙니다. cam_0·cam_1의 frame277 전체 검출을 비교한 예시이며, 같은 행·열의 자기 유사도 대각선은 약1입니다. 신원 판단이나 GT matching을 수행하지 않았습니다.</p>'''))
    pages.append(page(5,'보정 검증: 추적을 보류한 이유',f'''
<p>SCOUT는 K/R/T/dist와 <b>비평면 지면 메시</b>를 제공합니다. PoseTrack은 렌즈 왜곡 항이 없는 P와 <b>하나의 지면 평면을 가정한 H</b>를 사용합니다. 3×4·3×3 모양 및 역행렬 존재만으로 두 기하가 같아지지 않습니다.</p>
<img style="width:100%" src="{pic(figdir/'calibration_diagnostic.png')}">
{table(['카메라','표본/메시 교차','왜곡 이동 p95(px)','PCA 평면 오차 p95*'],[(r['camera'],r['sampled_bbox_points']+'/'+r['mesh_hits'],f"{float(r['distortion_shift_px_p95']):.2f}",f"{float(r['pca_vs_mesh_world_error_p95']):.3f}") for r in cal])}
<p class="small">* 실제 검출의 BBox 하단 중심을 왜곡 보정한 ray와 지면 메시의 교점, 동일 ray와 공통 PCA 평면의 교점 사이3D 거리입니다. 단위는 dataset unit이며 독립 검증된 meter가 아닙니다. 사람 위치 GT 오차도 아닙니다.</p>
<h2>검사 범위와 결론</h2><p>내부 frame0·10·20·30·40·49의 모든 검출 하단665개 중597개가 메시와 교차했습니다. 나머지는 화면 경계/교차 실패로 CSV에 이유를 남겼습니다. 행렬 rank와 가역성은 만족했지만, 후보 평면의 p95 오차가2.568~11.811단위이고 raw 영상에는 왜곡이 남아 있습니다.</p>
<div class="callout">후보 P/H는 <b>candidate_matrices_DO_NOT_USE</b>로만 제공합니다. native calibration.json은 생성하지 않았습니다. 메시 ray의 작은 재투영 오차는 수학적 자기 일관성이며 보정의 물리 정확성이나 사람 발 위치 정확성을 증명하지 않습니다.</div>'''))
    pages.append(page(6,'ID 결정 코드: 확인한 조건',f'''
<p><b>아래는 소스 정적 분석입니다.</b> 실제 SCOUT 추적 분기·Global ID 이벤트 로그로 해석하면 안 됩니다. 정확한 파일·행 번호와 수집할 값은 code_flow.csv에 정리했습니다.</p>
{table(['지점','원본 조건과 해석'],[
('BBox affinity · L625','Kalman 예측 IoU, age≤15, detection score를 적용: (IoU−0.5)×age mask×score.'),
('Geometry · L544 / C++ L423','Python은 loop_t_homo_full 호출 뒤 continue. 실제 연산은 C++이며 같은 view/오래된 bbox는 제외.'),
('Matching · L842','epi/homo≥−2.5 clamp → 가중합(1,5,1,5)/(1+ReID weight) → penalty → 음수0 → Hungarian.'),
('실제 update · L926','Hungarian에 선택돼도 최종 affinity≤0이면 update하지 않음.'),
('Per-view 상태 · L896','미선택 target의 maxIoU>0.5면 Occluded, 아니면 Missing. 전체 TrackState와 별도.'),
('ReID · L463','score≥0.88, 상태 이력에 Occluded/Missing, 양의 bank cosine 필요. maxcos−0.5, weight=1; 기본1e−5.'),
('Feature bank · L392','상체관절>0.5, bbox>0.9, overlap/IoU 조건을 통과하고 중복 유사도<0.6일 때 저장.'),
('초기화·복구 · L725/L692','미매칭 검출은 cross-view GLPK 그룹화 후 초기화. missing bank cosine>0.5면 기존 ID 재활성화.')])}
<h2>추가로 확인된 코드 검토 지점</h2><p>target_init의 BBox fallback에서 <code>(x1+x1)/2</code>를 사용하는 구문, 두 검출의 fallback에서 H/Hfeet를 다르게 쓰는 구문이 있습니다. PoseTrack bank 크기100과 PoseTracker 크기30도 혼재합니다. 원본은 수정하지 않았고, 실제 추적 영향이나 실패 원인으로 확정하지 않았습니다.</p>
<p class="small">input_gate_counts.csv는 실제 검출 score·관절 품질에 해당하는 일부 조건만 센 자료입니다. 활성화 횟수·bank 저장 횟수·ID 생성 횟수로 쓰면 안 됩니다.</p>'''))
    pages.append(page(7,'전달 파일과 다음 실행 순서',f'''
{table(['폴더','포함 자료'],[('01_environment','호스트/GPU/Docker, import·확장 검사, pip freeze/check, 가중치4종 hash·로드'),('02_adapter','camera/frame/timestamp map, 원본 보정, 진단 P/H, 665개 지면검사 CSV'),('03_intermediate','카메라별 det7 / pose57 / ReID2048, 행 대응표, 원시 JSON, 시각화·로그'),('04_tracking','BLOCKED 상태와 원인. Global ID 결과 없음'),('05_analysis','요청 trace8종 상태표, 정적 코드 흐름, 실제 입력 품질 조건 개수'),('06_reproducibility','실행 명령·설정·소스 커밋·diff·runner·Dockerfile·검사 코드')])}
<h2>요청한 trace8종은 왜 없는가?</h2><p>geometry_trace, affinity_snapshot, matching_trace, hungarian_trace, state_history, reid_activation_trace, feature_bank_trace, target_event_trace는 모두 <b>실제 tracker 실행 이후</b> 의미가 있습니다. 이번에는 보정에서 막혀 BLOCKED로 기록했으며 빈 파일이나 가상 이벤트로 대체하지 않았습니다.</p>
<h2>다음 단계</h2><ol><li>raw 영상·BBox·관절을 왜곡 없는 좌표계로 통일합니다.</li><li>메시 기반 지면 계산을 연결하거나 실제 검증된 평면 ROI로 실험 범위를 정합니다. 공통 world축·물리 단위와 feet0.15/거리1.5/epipolar0.2를 확인합니다.</li><li>1카메라 후2~3카메라로 추적을 실행하고 위8종 로그를 계측합니다. 원본과 계측본은 분리하고 diff를 남깁니다.</li><li>정상 매칭·미매칭·가림·복구 대표사례를 골라 실제 값으로 함수 분기를 설명합니다. 성능 평가는 이후 별도 단계로 진행합니다.</li></ol>
<div class="callout">담당자가 지금 할 수 있는 분석: 입력 형식·행 대응 확인, 실제 관절/특징 품질 검토, 단일 평면이 맞지 않는 수치 확인, 추적 코드 계측 지점 결정. 아직 할 수 없는 분석: 실제 Global ID 선택 이유·state 변화·bank 저장/복구 이벤트 해석.</div>
<p class="small">ZIP에는 PDF와 검토용 산출물이 들어 있습니다. 전체 SCOUT·대형 가중치·Docker 레이어는 제외했습니다. FILES.csv 및 SHA256SUMS.txt로 파일 무결성을 확인할 수 있습니다.</p>'''))
    tree=ast.parse((ROOT/'scripts/create-osc-handoff.py').read_text())
    css=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='css' for t in n.targets))
    css+='\ntable {font-size:8.5pt} td {word-break:keep-all;overflow-wrap:anywhere} .page {overflow:hidden;} code {font-size:8pt} ol {padding-left:6mm}\n'
    dest=ROOT/'docs/handoff'/f'{STEM}.html'
    write(dest,'<!DOCTYPE html><html lang="ko"><meta charset="utf-8"><title>PoseTrack SCOUT 실행 검증 인계</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>')
    print(dest)
    print('PASS: handoff data prepared; PDF/video/ZIP verification still required')


if __name__=='__main__':main()
