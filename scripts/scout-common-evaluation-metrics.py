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

# Standalone common metric kernels extracted from the existing evaluator.
# Same equations; only locate the bundled official TrackEval source portably.
TRACK_ROOT=Path('/vendor/TrackEval/trackeval')
if not TRACK_ROOT.is_dir():TRACK_ROOT=Path(__file__).resolve().parents[1]/'vendor/TrackEval/trackeval'
# Load official metric kernels only, avoiding optional dataset-reader packages.
for name,path in [('trackeval',str(TRACK_ROOT)),('trackeval.metrics',str(TRACK_ROOT/'metrics'))]:
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
