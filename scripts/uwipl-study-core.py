#!/usr/bin/env python3
"""SCOUT adaptation of native UWIPL BoTSORT / anchor clustering. No GT input.

Explicit adaptations: eight uniform cameras, observed anchor frames, aligned
current features instead of fragile feature-history indexing, flush final frame.
"""
import argparse, collections, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0,'/opt/uwipl/BoT-SORT')
from tracker.bot_sort import BoTSORT

def save(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
def mod(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    inf=Path('/inference');out=Path('/out')
    ds=json.loads((inf/'uwipl/detection/detections.json').read_text())
    rr=inf/'uwipl/reid'
    names=['osnet_ms_m_c','osnet_ibn_ms_m_c','osnet_ain_ms_m_c',
        'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0',
        'osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0']
    # Runner filenames are truncated at the first period; record exact paths.
    paths=[next(rr.glob(n+'*.npy')) for n in names]
    rowmaps=[json.loads(p.with_name(p.stem+'-rows.json').read_text()) for p in paths]
    assert all(r==rowmaps[0] for r in rowmaps)
    features=np.concatenate([np.load(p,allow_pickle=False) for p in paths],axis=1)
    assert features.shape[1]==2560 and np.isfinite(features).all()
    bykey={};offset=0
    for d in ds:
        n=len(d['selected']);bykey[(d['camera'],d['index'])]=(d,features[offset:offset+n].copy(),offset)
        for j in range(n):assert rowmaps[0][offset+j]['key']==d['key'] and rowmaps[0][offset+j]['box_index']==j
        offset+=n
    assert offset==len(features)
    native=mod('/opt/uwipl/BoT-SORT/tools/aic_track.py','native_aic')
    pool=[];pool_index={};all_summary=[]
    for variant,fps,vote,short,imin,igap in [('original_frames',30,15,10,5,20),('fps10',10,5,4,2,7)]:
        tracks=[];cfg=native.make_parser().parse_args(['/unused'])
        cfg.fps=fps;cfg.track_low_thresh=.1;cfg.nms_thres=1;cfg.aspect_ratio_thresh=10
        # Fixed cameras; native BoTSORT already disables GMC.
        for cam in range(8):
            tracker=BoTSORT(cfg,frame_rate=fps)
            for frame in range(300):
                d,fts,_=bykey[(f'cam_{cam}',frame)]
                det=np.asarray(d['selected'],dtype=float).reshape(-1,7)[:,:5]
                online=tracker.update(det.copy(),fts.copy())
                for t in online:
                    b=t.tlwh.copy()
                    if b[2]*b[3]<=cfg.min_box_area:continue
                    feat=t.curr_feat.copy()
                    assert feat.shape==(2560,) and np.isfinite(feat).all()
                    coords=[float(x) for x in b];k=(cam,frame,*[round(x,3) for x in coords])
                    if k not in pool_index:
                        pool_index[k]=len(pool);pool.append({'row':len(pool),'camera':f'cam_{cam}',
                            'camera_id':cam+1,'frame':frame,'source_frame':277+frame,
                            'bbox_xywh':coords,'score':float(t.score),'input_file':d['file']})
                    tracks.append({'camera':cam+1,'frame':frame,'source_frame':277+frame,
                        'local_id':int(t.track_id),'bbox':coords,'score':float(t.score),
                        'pose_row':pool_index[k],'feature':feat})
            print('SCT',variant,cam,len(tracks),flush=True)
        save(out/f'{variant}/sct.json',[{k:v for k,v in x.items() if k!='feature'} for x in tracks])
        for seconds,nframes in [(5,50),(30,300)]:
            work=out/f'{variant}/s{seconds}';work.mkdir(parents=True,exist_ok=True)
            cur=[r for r in tracks if r['frame']<nframes]
            anchor_frames=sorted(set(np.linspace(0,nframes-1,11,dtype=int).tolist()))
            anchor_rows=[];anchor_feat=[]
            for cam in range(8):
                for frame in anchor_frames:
                    d,ft,off=bykey[(f'cam_{cam}',frame)]
                    for j,b in enumerate(d['selected']):
                        if (b[2]-b[0])*(b[3]-b[1])>10000 and b[4]>.95:
                            anchor_rows.append({'camera':cam+1,'frame':frame,'source_frame':277+frame,
                                'detection_index':j,'feature_row':off+j,'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]]})
                            anchor_feat.append(ft[j])
            af=np.asarray(anchor_feat)
            assert len(af)>=2,'No eligible anchor observations; do not fill using GT'
            estimate=AgglomerativeClustering(distance_threshold=80,n_clusters=None).fit(af)
            count=int(estimate.labels_.max()+1)
            labels=AgglomerativeClustering(n_clusters=count).fit(af).labels_
            anf=af/np.linalg.norm(af,axis=1,keepdims=True)
            centers=np.stack([anf[labels==k].mean(axis=0) for k in range(count)])
            for r,l in zip(anchor_rows,labels):r['anchor_id']=int(l+1)
            save(work/'anchor_members.json',anchor_rows)
            np.save(work/'anchor_mean_unit_features.npy',centers)
            lens=collections.Counter((r['camera'],r['local_id']) for r in cur)
            eligible=[dict(r) for r in cur if lens[r['camera'],r['local_id']]>=short]
            groups=collections.defaultdict(list)
            for r in eligible:groups[(r['camera'],r['frame'])].append(r)
            assigned=[];unassigned=[]
            for key,rs in sorted(groups.items()):
                feat=np.stack([r['feature'] for r in rs]);feat/=np.linalg.norm(feat,axis=1,keepdims=True)
                cost=1-feat@centers.T
                # Mean cosine distance equals the native average of per-anchor-sample cosine distances.
                ri,ci=linear_sum_assignment(cost)
                for i,j in zip(ri,ci):
                    row={k:v for k,v in rs[i].items() if k!='feature'}
                    row.update(raw_anchor_id=int(j+1),id=int(j+1),assignment_cost=float(cost[i,j]))
                    assigned.append(row)
                for i in set(range(len(rs)))-set(ri):unassigned.append({k:v for k,v in rs[i].items() if k!='feature'})
            ts=collections.defaultdict(list)
            for r in assigned:ts[(r['camera'],r['local_id'])].append(r)
            changed=0
            for rs in ts.values():
                rs.sort(key=lambda r:r['frame']);last=None
                for start in range(0,len(rs),vote):
                    chunk=rs[start:start+vote]
                    choice=last if len(chunk)<vote and last is not None else collections.Counter(r['raw_anchor_id'] for r in chunk).most_common(1)[0][0]
                    for r in chunk:r['id']=choice;changed+=int(choice!=r['raw_anchor_id'])
                    last=choice
            assigned.sort(key=lambda r:(r['camera'],r['frame'],r['local_id']))
            save(work/'anchor_predictions.json',assigned);save(work/'hungarian_unassigned.json',unassigned)
            summary={'seconds_nominal':seconds,'frames_per_camera':nframes,'variant':variant,
                'fps_argument':fps,'effective_lost_buffer':int(fps/30*cfg.track_buffer),
                'majority_vote_window':vote,'short_track_min_observations':short,'interpolation_n_min':imin,'interpolation_n_dti':igap,
                'anchor_frames':anchor_frames,'anchor_observations':len(af),'anchors':count,'sct_rows':len(cur),
                'eligible_rows':len(eligible),'hungarian_unassigned':len(unassigned),'anchor_rows':len(assigned),
                'majority_changed_rows':changed,'gt_used':False,'clustering_raw_euclidean_threshold':80,
                'upstream_S001_vote_windows':[200,300,150,300,300,500,150],
                'adaptations':['uniform cameras low threshold0.1; no AIC c005 ROI or special person8',
                    '11 observed frames replace absent AIC anchor indices','explicit aligned current features and final-frame flush',
                    'paper/generic vote15 baseline; no S001 hand-tuned vote windows applied to SCOUT',
                    'no padded artificial IDs or forced assignment when tracks exceed anchors']}
            save(work/'summary.json',summary);all_summary.append(summary)
            print('ANCHORS',variant,seconds,count,'rows',len(assigned),'vote changes',changed,flush=True)
    save(out/'pose_requests.json',pool);save(out/'core_summary.json',all_summary)
    save(out/'feature_alignment.json',{'status':'PASS','rows':offset,'dimensions':2560,'models':[p.name for p in paths],
        'gt_used':False,'pose_unique_requests':len(pool),'raw_finite':True})

if __name__=='__main__':main()
