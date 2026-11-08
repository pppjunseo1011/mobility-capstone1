#!/usr/bin/env python3
"""Document one-factor anchor experiments; preserve original evaluation fields."""
import ast,base64,collections,csv,hashlib,html,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-anchor-sweep-20261102';B=R/'outputs/uwipl-study-20261101';V=O/'visualizations'
STEM='UWIPL_SCOUT_앵커_단일변수실험_20261102'
def load(p):return json.loads(p.read_text())
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rs):
    with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def table(h,rs):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rs)+'</tbody></table>'
def pct(x):return f'{100*float(x):.2f}%'
def pic(p):return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
def page(i,t,b):return f'<section class="page"><div class="eyebrow">SCOUT × UWIPL · 앵커 단계 단일 변수 실험</div><h1>{t}</h1>{b}<footer><span>2026.11.02 · 10 FPS · 개발 구간 평가 · STCRA 미실행</span><b>{i} / 5</b></footer></section>'
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    V.mkdir(exist_ok=True);rows=list(csv.DictReader((O/'experiment_comparison.csv').open()));choice=load(O/'selection.json')
    assert len(rows)==16 and choice['selected_area']==1000 and choice['selected_distance']==40
    def row(s,a,d):return next(r for r in rows if int(r['seconds'])==s and int(r['area_threshold'])==a and int(r['distance_threshold'])==d)
    stats=load(O/'candidate_statistics.json');stat=lambda s:next(r for r in stats if r['seconds']==s and r['area_threshold']==10000 and r['distance_threshold']==80)
    cfgrows=[];fragment=[];compact=[]
    for r in rows:
        w=O/'experiments'/r['config_id']/('s'+r['seconds']);cfg=load(w/'summary.json');audit=load(w/'anchor_gt_audit.json')
        assert cfg['fps_argument']==10 and cfg['effective_lost_buffer']==10 and cfg['majority_vote_window']==5 and cfg['short_track_min_observations']==4 and cfg['score_threshold_strict']==.95 and not cfg['stcra_run']
        assert int(r['raw_rows'])+int(r['hungarian_unassigned'])==int(r['eligible_rows'])
        gtanchors=collections.defaultdict(list)
        for aid,votes in audit['member_vote_counts'].items():
            for gt,count in votes.items():
                if count:gtanchors[gt].append(int(aid))
        for gt,aids in gtanchors.items():fragment.append({'config_id':r['config_id'],'seconds':int(r['seconds']),'gt_id':gt,'anchor_count':len(aids),'anchor_ids':json.dumps(aids)})
        compact.append({'config_id':r['config_id'],'seconds':int(r['seconds']),'anchors':int(r['anchors']),
            'gt_covered':len(gtanchors),'gt_with_multiple_anchors':sum(len(x)>1 for x in gtanchors.values()),
            'mean_anchors_per_covered_gt':float(np.mean([len(x) for x in gtanchors.values()])),
            'max_anchors_per_gt':max(map(len,gtanchors.values())),'max_gt_people_in_one_anchor':int(r['max_gt_people_in_one_anchor']),
            'weighted_purity_matched_only':float(r['anchor_gt_weighted_purity']),
            'matched_members':int(r['anchor_gt_matched_members']),'unmatched_members':int(r['anchor_observations'])-int(r['anchor_gt_matched_members'])})
        cfgrows.append({'config_id':r['config_id'],'seconds':int(r['seconds']),'family':r['family'],'area_strict_gt':int(r['area_threshold']),
          'raw_Ward_distance':int(r['distance_threshold']),'score_strict_gt':.95,'fps_argument':10,'lost_buffer':10,'vote_window':5,'short_track_min':4,'anchor_time_samples':11,'STCRA_run':False})
    csvout(O/'parameter_comparison.csv',cfgrows);csvout(O/'anchor_gt_fragmentation.csv',fragment);csvout(O/'anchor_gt_summary.csv',compact)
    for s in [5,30]:
        work=O/f'experiments/area_10000_distance_80/s{s}'
        assert load(work/'baseline_equivalence.json')['status']=='PASS' and load(work/'baseline_metric_equivalence.json')['status']=='PASS'
    original30=load(O/'experiments/area_10000_distance_80/s30/anchor_gt_audit.json')
    assert len(original30['member_vote_counts']['1'])==13
    findings={'full5s_area_only_pass_ratio':stat(5)['full_area_pass']/stat(5)['full_detections'],
      'full5s_area_and_score_pass_ratio':stat(5)['full_joint_pass_ratio'],'full5s_median_area':stat(5)['full_bbox_area_median'],
      'actual_anchor_11_times_5s_pass_ratio':stat(5)['sampled_joint_pass_ratio'],'baseline30_anchor1_gt_people':13,
      'selection':choice,'one_factor_integrity':'PASS all fixed parameters asserted; baseline inference and metrics exactly match previous',
      'combined_worse_than_area_only_both_clips':all(float(row(s,1000,40)['IDF1_view_aware'])<float(row(s,1000,80)['IDF1_view_aware']) for s in [5,30]),
      'STCRA_run':False,'validation_scope':'development clips, not independent test'}
    js(O/'findings.json',findings)
    for family,values in [('area',[10000,5000,2500,1000]),('distance',[80,60,40,20])]:
        fig,axs=plt.subplots(1,2,figsize=(10,3.4))
        for ax,s in zip(axs,[5,30]):
            rr=[row(s,v,80) if family=='area' else row(s,10000,v) for v in values]
            for metric,label,marker in [('IDF1_view_aware','view-aware IDF1','o'),('IDF1_2D','2D IDF1','s'),('HOTA_2D','2D HOTA','^')]:
                ax.plot(range(4),[100*float(r[metric]) for r in rr],label=label,marker=marker)
            ax.set_xticks(range(4));ax.set_xticklabels(values);ax.set_xlabel('Area threshold (px^2)' if family=='area' else 'Raw Ward distance threshold')
            ax.set_ylabel('score (%)');ax.set_title(f'{s}s / other settings fixed');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(V/f'{family}_ablation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.3))
    variants=[(10000,80,'baseline'),(1000,80,'area only'),(10000,40,'distance only'),(1000,40,'combined')]
    for ax,s in zip(axs,[5,30]):
        rr=[row(s,a,d) for a,d,_ in variants];ax.bar(range(4),[100*float(x['IDF1_view_aware']) for x in rr],color=['#8797a6','#2a8a8a','#6886ae','#c48458'])
        ax.set_xticks(range(4));ax.set_xticklabels([v[2] for v in variants]);ax.set_ylabel('view-aware IDF1 (%)');ax.set_title(f'{s}s / selected combination')
        for i,x in enumerate(rr):ax.text(i,100*float(x['IDF1_view_aware'])+.5,f"{100*float(x['IDF1_view_aware']):.2f}",ha='center',fontsize=9)
        ax.set_ylim(0,55)
    fig.tight_layout();fig.savefig(V/'combined_comparison.png',dpi=160);plt.close(fig)
    evidence=O/'reproducibility';evidence.mkdir(exist_ok=True)
    for name in ['uwipl-anchor-sweep.py','report-uwipl-anchor-sweep.py','evaluate-uwipl-study.py','uwipl-study-core.py']:
        cp(R/'scripts'/name,evidence/'scripts'/name)
    for name in ['sources.json','image_ids.txt','timestamp_audit.csv']:cp(B/name,evidence/name)
    for name in ['protocol.json','self_test.json','view_protocol_self_test.json']:cp(B/'evaluation'/name,evidence/'previous_evaluation'/name)
    cp(R/'weights/uwipl/manifest.json',evidence/'weights_manifest.json')
    for kind in ['detection','reid']:cp(B/f'inference30/uwipl/{kind}/report.json',evidence/f'reused_{kind}_report.json')
    provenance_files=[B/'fps10/sct.json',B/'inference30/uwipl/detection/detections.json',R/'outputs/osc-analysis-20261101/s30_th40/gt_selected.json',
       *sorted((B/'inference30/uwipl/reid').glob('*.npy'))]
    js(evidence/'input_hashes.json',[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in provenance_files])
    # Keep original comparison schema plus a Korean-friendly BOM copy.
    ordered=[]
    for s in [5,30]:
        for a,d in [(10000,80),(5000,80),(2500,80),(1000,80),(10000,60),(10000,40),(10000,20),(1000,40)]:ordered.append(row(s,a,d))
    csvout(O/'experiment_comparison_excel.csv',ordered)
    pages=[]
    pages.append(page(1,'앵커 원인 확인과 실험 결론',f'''<p class="lead">넓이 또는 거리 기준만 하나씩 바꿔 <b>5초·30초의 앵커 단계</b>를 비교했습니다. STCRA·포즈·보간은 실행하지 않았습니다.</p>
<div class="stats"><div><strong>16개 결과</strong><span>단일 변수 14 + 조합 2</span></div><div><strong>동일 입력</strong><span>기존 검출·ReID·10 FPS 설정</span></div><div><strong>기준 재현 PASS</strong><span>앵커·배정·지표·GT 분포 일치</span></div></div>
<h2>요청하신 두 원인의 재확인</h2>
{table(['항목','실제 확인값'],[('5초 넓이만 >10,000','515 / 5,569 = 9.25%'),('5초 넓이와 score>0.95 동시 통과','450 / 5,569 = 8.08%'),('5초 박스 넓이 중앙값','1,254px² (기존 selected 검출 기준)'),('실제 11개 앵커 시점의 통과','95 / 1,212 = 7.84%'),('30초 기존 앵커 1번','GT 13명이 함께 포함됨; member_vote_counts로 재확인')])}
<h2>관찰 결과</h2><p>넓이만1,000으로 낮추면 view-aware IDF1은 5초 <b>14.50→42.39%</b>, 30초 <b>11.35→45.98%</b>로 올랐습니다. 미배정도 양쪽 모두0이 됐습니다. 다만30초 IDSW는359→719로 늘었습니다.</p>
<p>거리만 줄이는 실험에서는 평균 view-aware IDF1 기준40이 선택됐습니다. 그러나 두 값을 합친 <b>넓이1,000·거리40</b>은 넓이만 변경한 결과보다 낮았습니다. 앵커가5초116개·30초193개로 늘면서 같은GT가 여러 앵커로 나뉘는 현상을 함께 확인했습니다.</p>
<div class="callout">앵커 혼합 감소·미배정 감소·ID 연결 안정성은 서로 다른 지표입니다. 한 앵커에 한 사람만 남아도 같은 사람이 여러 앵커로 쪼개지면 IDF1이 낮아질 수 있습니다.</div>'''))
    for pi,family,values,title in [(2,'area',[10000,5000,2500,1000],'넓이 기준만 변경'),(3,'distance',[80,60,40,20],'거리 기준만 변경')]:
        rr=[]
        for s in [5,30]:
            for v in values:
                r=row(s,v,80) if family=='area' else row(s,10000,v)
                rr.append([f'{s}s',v,r['anchors'],r['hungarian_unassigned'],pct(r['IDF1_2D']),pct(r['IDF1_view_aware']),pct(r['HOTA_2D']),r['IDSW_2D']])
        desc='거리80·score>0.95를 고정하고 넓이만 바꿨습니다.' if family=='area' else '넓이>10,000px²·score>0.95를 고정하고 거리만 바꿨습니다. 거리는 정규화 전5개 OSNet 특징을 연결한2560차원 공간의 Ward 병합 거리이며 미터가 아닙니다.'
        pages.append(page(pi,title,f'''<p>{desc} 다수결5프레임·최소트랙4개 관측·유지10프레임·동일한11개 앵커 시점을 유지했습니다.</p>
{table(['구간','넓이' if family=='area' else '거리','앵커','미배정','IDF1 2D','IDF1 view','HOTA 2D','IDSW'],rr)}
<img style="width:100%" src="{pic(V/f'{family}_ablation.png')}">
<p>{'선정값은1,000입니다. 5초만 보면2,500의 view-aware IDF1이43.41%로 더 높지만, 5초·30초를 같은 비중으로 평균하면1,000이44.18%로 가장 높습니다.' if family=='area' else '선정값은40입니다. 거리20은 앵커가5초69개·30초88개로 증가했고, GT 대응된 구성원의 순도는100%였지만 IDF1은40보다 낮았습니다. 순도만으로 최적값을 고르지 않았습니다.'}</p>
<p class="small">미배정은 짧은 트랙 제외 후 Hungarian에서 앵커 열 수 부족으로 배정받지 못한 관측 수입니다. 평가에서 중복ID를 제거한 수와 다릅니다. 각 원시 목록을 별도로 보존했습니다.</p>'''))
    rr=[]
    for s in [5,30]:
        for a,d,label in variants:
            r=row(s,a,d);rr.append([f'{s}s',label,f'{a}/{d}',r['anchors'],pct(r['IDF1_view_aware']),pct(r['HOTA_2D']),r['IDSW_2D']])
    pages.append(page(4,'선정값 조합: 자동으로 더 좋아지지 않음',f'''<p>단일 변수별로 <b>5초·30초 평균 view-aware IDF1</b>이 가장 높은 값을 선택했습니다. 동률이면 평균 HOTA 2D, IDSW 합계를 순서대로 사용하도록 사전 고정했습니다.</p>
{table(['구간','설정','넓이/거리','앵커','IDF1 view','HOTA 2D','IDSW'],rr)}
<img style="width:100%" src="{pic(V/'combined_comparison.png')}">
<p>조합1,000/40의 view-aware IDF1은5초24.96%,30초25.41%였습니다. 기존 기준보다는 높지만 넓이만1,000으로 바꾼42.39%·45.98%보다 낮습니다. 이번 범위에서는 <b>1,000/80을 다음 검토 기준으로 삼을 근거</b>가 더 강합니다.</p>
<div class="callout">두 단일 변수 최적값을 합친 설정을 추가로 시험한 결과입니다. 전체 조합을 탐색한 최적값이 아니며, 같은 개발 구간의GT로 선택했으므로 독립 평가 성능으로 보고하면 안 됩니다.</div>'''))
    gtrows=[]
    for s in [5,30]:
        for a,d,label in variants:
            q=next(q for q in compact if q['seconds']==s and q['config_id']==f'area_{a}_distance_{d}')
            gtrows.append([f'{s}s',label,q['gt_covered'],pct(q['weighted_purity_matched_only']),q['max_gt_people_in_one_anchor'],q['gt_with_multiple_anchors'],f"{q['mean_anchors_per_covered_gt']:.2f}"])
    pages.append(page(5,'GT 분포와 전달 자료',f'''{table(['구간','설정','GT포함 인원','가중 순도','최대 혼합 인원','여러앵커 GT수','평균 앵커/GT'],gtrows)}
<p class="small">GT 대응은 이전과 같은 camera/frame별 IoU≥0.5 일대일 매칭입니다. 순도는 GT 대응된 후보만의 다수GT 비율이며, 미대응 수는 별도 기록했습니다. 앵커 번호는 설정마다 다시 부여되므로 서로 다른 설정의1번끼리 같은 집단이라고 비교하지 않습니다.</p>
{table(['자료','파일'],[('전체16행 지표','experiment_comparison.csv / Excel용 별도 CSV'),('설정·선정 근거','parameter_comparison.csv / selection.json'),('각 설정의 GT 분포','experiments/<설정>/s5 또는 s30/anchor_gt_audit.json'),('GT 혼합·분할 상세','anchor_gt_summary.csv / anchor_gt_fragmentation.csv'),('미배정·예측·평가 제외','hungarian_unassigned.json / anchor_predictions.csv / anchor_eval_duplicate_removals.json'),('원본 재현·코드·입력hash','baseline_*equivalence.json / reproducibility/')])}
<h2>해석 범위</h2><p>HOTA/IDF1 2D는 카메라별 TrackEval 결합, IDSW는 카메라별 시간축 합계입니다. view-aware는 카메라 간 박스 매칭을 막고 Global ID를 공유하는 기존 진단 프로토콜이며 공식 SCOUT3D 평가가 아닙니다.</p>
<p>기존과 같이 평가 때만 camera/frame/id 중 최고 점수1개를 남겼습니다. 원시 출력은 유지했습니다. 30초 구간의29개 index에서 카메라 시차가50ms를 넘고 최대2.051초인 제한도 그대로입니다. 공간 STCRA를 실행하지 않았어도 이 구간을 엄밀한 동시점 평가로 해석하지 않습니다.</p>
<p class="small">기존 검출·5종ReID를 재사용했습니다. 이전SCT JSON에 특징이 없어CPU SCT를 한 번 재생해 복원했고, 33,142개 bbox·score와 트랙 분할이 이전과 일치함을 확인했습니다. 대형 특징 캐시는 서버에 두고 ZIP에는 검토 결과를 담았습니다.</p>'''))
    tree=ast.parse((R/'scripts/create-osc-handoff.py').read_text());css=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='css' for t in n.targets))
    css+='\ntable{font-size:8pt}td{word-break:keep-all;overflow-wrap:anywhere}h1{font-size:22pt}.page{overflow:hidden}\n'
    (R/'docs/handoff'/f'{STEM}.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>UWIPL 앵커 단일 변수 실험</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>')
    (O/'START_HERE.md').write_text('''# UWIPL 앵커 단일 변수 실험 · 2026-11-02

PDF와 experiment_comparison_excel.csv를 먼저 보세요. 이전 experiment_comparison.csv의 지표 열을 유지했습니다.

- sequence_1 cam_0~7, 5초 source277~326 / 30초277~576, 10FPS. STCRA·포즈·보간 미실행.
- 넓이10,000/5,000/2,500/1,000 (거리80 고정), 거리80/60/40/20 (넓이10,000 고정). 중복 기준은 한 번만 계산하여14행, 선정값 조합2행으로 총16행입니다.
- score>0.95, 다수결5, 짧은트랙4관측, lost buffer10, 기존11개 앵커 시점·검출·ReID·평가를 고정했습니다.
- 기준10,000/80의 앵커·예측·미배정과 지표·GT 분포는 기존 결과와 정확히 같습니다.
- 선정 기준: 5초/30초 평균 view-aware IDF1 → 평균 HOTA2D → 작은 IDSW 합계. 넓이1,000, 거리40이 각각 선택됐으나 조합1,000/40은 넓이만 바꾼1,000/80보다 낮았습니다. 개발구간GT를 사용한 선택이며 독립 테스트 결과가 아닙니다.

## 파일
experiments/area_<넓이>_distance_<거리>/s5 또는 s30 안에 다음을 제공합니다.
- summary.json, candidate_filter_stats.json: 설정·필터 분모·통과 수.
- anchor_members.json, anchor_gt_audit.json, anchor_member_gt_matches.json: 후보와GT 분포/행 대응. GT는 앵커 생성 후 사후 매칭에만 사용.
- hungarian_unassigned.json: 실제 미배정 행.
- anchor_predictions.json/csv, anchor_metrics.json: 원시 예측과 이전 프로토콜 지표.
- anchor_eval_duplicate_removals.json: 평가에서만 제거한 동일 camera/frame/id 중복.

parameter_comparison.csv, selection.json, anchor_gt_summary.csv, anchor_gt_fragmentation.csv, findings.json과 visualizations/를 같이 보세요. 클러스터 번호는 설정 간 의미가 유지되지 않습니다. 순도100%가 같은 사람의 여러앵커 분할을 막아 주지는 않습니다.

## 재사용 및 제한
검출·임베딩 추론은 다시 하지 않았습니다. 이전SCT JSON에 현재 특징이 없어 기존이미지 CPU SCT를 한 번 재생하고 모든33142행 bbox·score 및 트랙분할 일치를 검증해 특징을 캐시했습니다. STCRA는 호출하지 않았습니다. 처음 평가에서 GT미대응 앵커의 빈 사전 키가 기준 비교를 방해했으며 처리만 수정했습니다. 완료 지표는 재사용했고 실패로그도 보존했습니다.
평가모듈은 기존 evaluate-uwipl-study.py를 그대로 불러옵니다. 2D TrackEval과 자체 view-aware 지표를 구분하며, 원시 결과를 보존하고 평가에서만 최고점수 중복제거를 합니다. 30초 동기화 제한은 이전과 같습니다.

## 실행 계약
기존 scout-mtmc/uwipl-botsort:8e23a69 이미지, CPU4/메모리8GB, runc, network=none, uid1000.
/project=프로젝트(ro), /previous=outputs/uwipl-study-20261101(ro), /inference=이전inference30(ro), /out=이번실험(rw).
cache → generate → evaluate → choose → generate --combined → evaluate 순서로 scripts/uwipl-anchor-sweep.py 실행.
evaluate에는 /vendor=vendor(ro), /gt.json=outputs/osc-analysis-20261101/s30_th40/gt_selected.json(ro)이 추가됩니다. generate는GT를 읽지 않습니다. 가중치·전체SCOUT·sct_current_features.npy는 ZIP에서 제외했습니다.
''')
    print('Report ready',findings['combined_worse_than_area_only_both_clips'])
if __name__=='__main__':main()
