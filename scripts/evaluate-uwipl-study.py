#!/usr/bin/env python3
"""GT-only evaluation after inference; official TrackEval metric kernels.

2D: eight independent camera sequences. View-aware: camera/frame observation
steps with shared global identity namespace, no cross-camera box matching.
IDSW is always summed over temporal per-camera sequences, not view steps.
"""
import collections,csv,importlib,json,sys,types
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load official metric kernels only, avoiding optional dataset-reader packages.
for name,path in [('trackeval','/vendor/TrackEval/trackeval'),('trackeval.metrics','/vendor/TrackEval/trackeval/metrics')]:
    pkg=types.ModuleType(name);pkg.__path__=[path];sys.modules[name]=pkg
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity
from trackeval.metrics.clear import CLEAR

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def csvout(p,rs):
    if not rs:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def iou(a,b):
    a=np.array(a,float).reshape(-1,4);b=np.array(b,float).reshape(-1,4)
    xx=np.maximum(a[:,None,:2],b[None,:,:2]);yy=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    inter=np.maximum(yy-xx,0).prod(2);den=a[:,2:].prod(1)[:,None]+b[:,2:].prod(1)[None,:]-inter
    return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)
def gt_matches(rows,truth):
    groups=collections.defaultdict(list);result={}
    for i,r in enumerate(rows):groups[(r['camera'],r['frame'])].append((i,r))
    for key,rs in groups.items():
        gt=truth.get(key,[]);cost=iou([r['bbox'] for _,r in rs],[r['bbox'] for r in gt]);ri,ci=linear_sum_assignment(-cost)
        for i,j in zip(ri,ci):
            if cost[i,j]>=.5:result[rs[i][0]]={'gt':gt[j],'iou':float(cost[i,j])}
    return result
def dedup(rows):
    keep={};removed=[]
    for i,r in enumerate(rows):
        k=(r['camera'],r['frame'],r['id'])
        if k in keep:
            prev=keep[k]
            if r.get('score',-1)>prev.get('score',-1):removed.append(prev);keep[k]=r
            else:removed.append(r)
        else:keep[k]=r
    return list(keep.values()),removed
def metric_data(rows,truth,keys):
    pred=collections.defaultdict(list)
    for r in rows:pred[(r['camera'],r['frame'])].append(r)
    gs=sorted({r['id'] for k in keys for r in truth.get(k,[])});ps=sorted({r['id'] for k in keys for r in pred.get(k,[])})
    gm={g:i for i,g in enumerate(gs)};pm={g:i for i,g in enumerate(ps)}
    gd=[];pd=[];sim=[]
    for k in keys:
        g=truth.get(k,[]);p=pred.get(k,[])
        assert len({r['id'] for r in p})==len(p) and len({r['id'] for r in g})==len(g)
        gd.append(np.array([gm[r['id']] for r in g],int));pd.append(np.array([pm[r['id']] for r in p],int))
        sim.append(iou([r['bbox'] for r in g],[r['bbox'] for r in p]))
    return {'num_timesteps':len(keys),'num_gt_ids':len(gs),'num_tracker_ids':len(ps),
            'num_gt_dets':sum(map(len,gd)),'num_tracker_dets':sum(map(len,pd)),
            'gt_ids':gd,'tracker_ids':pd,'similarity_scores':sim}
def metric_suite(rows,truth,n):
    rows,removed=dedup(rows);metrics=[HOTA(),Identity({'PRINT_CONFIG':False}),CLEAR({'PRINT_CONFIG':False})]
    allres=[{} for _ in metrics]
    for cam in range(1,9):
        data=metric_data(rows,truth,[(cam,f) for f in range(n)])
        for m,r in zip(metrics,allres):r[cam]=m.eval_sequence(data)
    combined=[m.combine_sequences(r) for m,r in zip(metrics,allres)]
    viewdata=metric_data(rows,truth,[(cam,f) for f in range(n) for cam in range(1,9)])
    vh=metrics[0].eval_sequence(viewdata);vi=metrics[1].eval_sequence(viewdata)
    h,idr,cl=combined
    summary={'rows_evaluated':len(rows),'duplicate_id_rows_removed_for_eval':len(removed),
        'IDF1_2D':float(idr['IDF1']),'IDSW_2D':int(cl['IDSW']),'HOTA_2D':float(np.mean(h['HOTA'])),
        'DetA_2D':float(np.mean(h['DetA'])),'AssA_2D':float(np.mean(h['AssA'])),
        'IDF1_view_aware':float(vi['IDF1']),'HOTA_view_aware':float(np.mean(vh['HOTA'])),
        'global_ids':len({r['id'] for r in rows})}
    return summary,{'combined_2D':dict(zip(['HOTA','Identity','CLEAR'],combined)),
                    'HOTA_view_aware':vh,'Identity_view_aware':vi},removed
def stats(x):
    a=np.asarray(x,float)
    return {'count':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,
        'p95':float(np.percentile(a,95)) if len(a) else None,'max':float(a.max()) if len(a) else None}

def main():
    out=Path('/out');gt=json.loads(Path('/gt.json').read_text());truth=collections.defaultdict(list)
    for r in gt:
        cam=int(r['camera_id'].split('_')[1])+1;f=r['source_frame']-277
        truth[(cam,f)].append({'id':r['track_id'],'bbox':r['bbox_xywh'],'world':r['world_xyz']})
    # Semantic tests including shared identity across two views.
    testgt={(1,0):[{'id':1,'bbox':[0,0,10,10]}],(1,1):[{'id':1,'bbox':[0,0,10,10]}]}
    testpred=[{'camera':1,'frame':i,'id':i+1,'bbox':[0,0,10,10]} for i in range(2)]
    td=metric_data(testpred,testgt,[(1,0),(1,1)])
    assert np.allclose(HOTA().eval_sequence(td)['HOTA'],np.sqrt(.5))
    assert Identity({'PRINT_CONFIG':False}).eval_sequence(td)['IDF1']==.5
    assert CLEAR({'PRINT_CONFIG':False}).eval_sequence(td)['IDSW']==1
    for r in testpred:r['id']=1
    assert np.allclose(HOTA().eval_sequence(metric_data(testpred,testgt,[(1,0),(1,1)]))['HOTA'],1)
    assert iou([[0,0,2,2]],[[10,10,2,2]])[0,0]==0
    save(out/'evaluation/self_test.json',{'status':'PASS','tests':['perfect HOTA1','ID split HOTA sqrt(0.5), IDF1 .5, IDSW1','nonoverlap IoU0']})
    summary=[];alg=[];all_events=[]
    for variant in ['original_frames','fps10']:
      for seconds,n in [(5,50),(30,300)]:
        work=out/variant/f's{seconds}';pred=json.loads((work/'anchor_predictions.json').read_text())
        cfg=json.loads((work/'summary.json').read_text());members=json.loads((work/'anchor_members.json').read_text())
        am=gt_matches(members,truth);votes=collections.defaultdict(collections.Counter)
        for i,pair in am.items():votes[members[i]['anchor_id']][pair['gt']['id']]+=1
        identity={k:v.most_common(1)[0][0] for k,v in votes.items()}
        save(work/'anchor_gt_audit.json',{'anchor_majority_gt_map_evaluation_only':identity,
            'member_vote_counts':votes,'GT_not_used_for_anchors':True})
        matches=gt_matches(pred,truth);pose_gt={pred[i]['pose_row']:p for i,p in matches.items()}
        costs=[]
        for i,r in enumerate(pred):
            pair=matches.get(i);target=identity.get(r['raw_anchor_id']);correct=None
            if pair is not None and target is not None:correct=target==pair['gt']['id']
            costs.append({'camera':r['camera'],'source_frame':277+r['frame'],'pose_row':r['pose_row'],
                'raw_anchor_id':r['raw_anchor_id'],'voted_id':r['id'],'cost':r['assignment_cost'],
                'matched_gt':None if pair is None else pair['gt']['id'],'anchor_majority_gt':target,
                'correct':correct,'evaluation_only':True})
        csvout(work/'assignment_cost_gt.csv',costs)
        feet=json.loads((work/'foot_projection.json').read_text());errors=[]
        for r in feet:
            pair=pose_gt.get(r['pose_row'])
            if pair is None or pair['gt']['world'] is None:continue
            gtxy=pair['gt']['world'][:2];err=float(np.linalg.norm(np.array(r['world_xy'])-gtxy))
            errors.append({'pose_row':r['pose_row'],'camera':r['camera'],'source_frame':r['source_frame'],
                'gt_id':pair['gt']['id'],'iou':pair['iou'],'ankle_used':r['ankle_confidence_pass'],
                'pred_world_x':r['world_xy'][0],'pred_world_y':r['world_xy'][1],
                'gt_world_x':gtxy[0],'gt_world_y':gtxy[1],'error_xy_dataset_units':err,
                'error_m_if_one_unit_is_one_meter':err,'metric_unit_independently_verified':False})
        csvout(work/'projection_vs_gt.csv',errors)
        visible=collections.defaultdict(set)
        for (cam,f),rs in truth.items():
            if f<n:
                for r in rs:visible[(r['id'],f)].add(cam)
        gtids={r['id'] for (c,f),rs in truth.items() if f<n for r in rs}
        report={'variant':variant,'seconds':seconds,'anchors':cfg['anchors'],'gt_unique_people':len(gtids),
            'gt_people_covered_by_anchor_candidates':len({p['gt']['id'] for p in am.values()}),
            'anchor_observations':len(members),'majority_changed':cfg['majority_changed_rows'],
            'gt_frames_with_any_3view_id':len({f for (tid,f),vs in visible.items() if len(vs)>=3}),
            'gt_frame_ratio_3views':len({f for (tid,f),vs in visible.items() if len(vs)>=3})/n,
            'ankle_used':sum(r['ankle_confidence_pass'] for r in feet),'bbox_fallback':sum(not r['ankle_confidence_pass'] for r in feet),
            'matched_world_error_dataset_units':stats([r['error_xy_dataset_units'] for r in errors]),
            'cost_correct':stats([r['cost'] for r in costs if r['correct'] is True]),
            'cost_incorrect':stats([r['cost'] for r in costs if r['correct'] is False]),
            'cost_unlabelled':sum(r['correct'] is None for r in costs)}
        alg.append(report);save(work/'algorithm_audit.json',report)
        for schedule in ['original_units','meter_candidate']:
            path=work/schedule
            for stage in ['anchor','stcra','istcra','interpolation']:
                rows=json.loads((path/f'{stage}.json').read_text());metrics,detail,removed=metric_suite(rows,truth,n)
                metrics.update(variant=variant,seconds=seconds,schedule=schedule,stage=stage,raw_rows=len(rows))
                summary.append(metrics);save(path/f'{stage}_metrics.json',{'summary':metrics,'detail':detail})
                save(path/f'{stage}_eval_duplicate_removals.json',removed)
                csvout(path/f'{stage}_predictions.csv',[{'sequence':'sequence_1','camera':f"cam_{r['camera']-1}",
                    'frame':r['frame']+277,'model_frame':r['frame'],'id':r['id'],'x':r['bbox'][0],'y':r['bbox'][1],
                    'width':r['bbox'][2],'height':r['bbox'][3],'score':r.get('score',-1)} for r in rows])
                print('METRICS',variant,seconds,schedule,stage,metrics['IDF1_view_aware'],metrics['HOTA_2D'],flush=True)
            events=json.loads((path/'events.json').read_text())
            for e in events:
                match=pose_gt.get(e['pose_row']);gtid=None if match is None else match['gt']['id']
                e.update(gt_id=gtid,old_majority_gt=identity.get(e['old_id']),new_majority_gt=identity.get(e['new_id']))
                e['correct_after']=None if gtid is None or e['new_majority_gt'] is None else gtid==e['new_majority_gt']
                e['correct_before']=None if gtid is None or e['old_majority_gt'] is None else gtid==e['old_majority_gt']
                all_events.append(dict(e,variant=variant,seconds=seconds,schedule=schedule))
            save(path/'events_gt_audit.json',events)
    csvout(out/'experiment_comparison.csv',summary);save(out/'algorithm_summary.json',alg)
    csvout(out/'reassignment_gt_events.csv',all_events)
    save(out/'evaluation/protocol.json',{
        'gt':'SCOUT provided MOT annotations, all valid positive-area boxes in selected first 8 cameras; no size/confidence GT filtering',
        'IDF1_2D_HOTA_2D':'Official TrackEval kernels, independent camera sequences combined by metric rules',
        'IDSW_2D':'sum of temporal per-camera CLEAR IDSW at IoU0.5',
        'view_aware':'shared global IDs; separate camera/frame observation steps ensure boxes never match across cameras; not official SCOUT 3D benchmark',
        'HOTA_thresholds':list(np.arange(.05,.99,.05)),
        'duplicates':'Metrics only: highest-score row retained per camera/frame/id; all removed rows separately recorded. Raw outputs preserved.',
        'anchor_cost_gt':'GT is matched IoU>=.5 after inference; correctness uses anchor-member majority GT; unknowns excluded',
        'projection':'XY distance versus matched person GT; Z=0 H; conditional meter column requires unit verification',
        'scope':'development algorithm experiments, no model ranking or test-set claim'})

if __name__=='__main__':main()
