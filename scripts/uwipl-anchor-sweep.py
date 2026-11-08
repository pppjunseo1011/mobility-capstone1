#!/usr/bin/env python3
"""One-factor UWIPL anchor ablation; no STCRA, pose, or model inference.

cache/generate do not consume GT. evaluate consumes GT only after predictions.
choose selects development parameters, explicitly using the measured GT metrics.
"""
import argparse,collections,csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
O=Path('/out');B=Path('/previous');I=Path('/inference')
NAMES=['osnet_ms_m_c','osnet_ibn_ms_m_c','osnet_ain_ms_m_c','osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0','osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0']
def load(p):return json.loads(p.read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def csvout(p,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
def mod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def inputs():
    ds=load(I/'uwipl/detection/detections.json');rr=I/'uwipl/reid';paths=[next(rr.glob(n+'*.npy')) for n in NAMES]
    rowmaps=[load(p.with_name(p.stem+'-rows.json')) for p in paths];assert all(x==rowmaps[0] for x in rowmaps)
    features=np.concatenate([np.load(p,allow_pickle=False) for p in paths],axis=1);assert features.shape==(37023,2560) and np.isfinite(features).all()
    keys={};offset=0
    for d in ds:
        n=len(d['selected']);keys[d['camera'],d['index']]=(d,features[offset:offset+n],offset)
        for j in range(n):assert rowmaps[0][offset+j]['key']==d['key'] and rowmaps[0][offset+j]['box_index']==j
        offset+=n
    assert offset==len(features)
    return ds,features,keys
def cache():
    sys.path.insert(0,'/opt/uwipl/BoT-SORT');from tracker.bot_sort import BoTSORT
    ds,features,keys=inputs();native=mod('/opt/uwipl/BoT-SORT/tools/aic_track.py','native_aic_sweep')
    cfg=native.make_parser().parse_args(['/unused']);cfg.fps=10;cfg.track_low_thresh=.1;cfg.nms_thres=1;cfg.aspect_ratio_thresh=10
    old=load(B/'fps10/sct.json');byframe=collections.defaultdict(list)
    for i,r in enumerate(old):byframe[r['camera']-1,r['frame']].append((i,r))
    cached=np.zeros((len(old),2560),dtype=np.float32);used=set();identity_map={};reverse={};start=time.monotonic()
    for cam in range(8):
        tracker=BoTSORT(cfg,frame_rate=10)
        for frame in range(300):
            d,fts,_=keys[f'cam_{cam}',frame];det=np.asarray(d['selected'],float).reshape(-1,7)[:,:5]
            online=[t for t in tracker.update(det.copy(),fts.copy()) if t.tlwh[2]*t.tlwh[3]>cfg.min_box_area]
            expected=byframe[cam,frame];assert len(online)==len(expected),(cam,frame,len(online),len(expected))
            for t,(i,r) in zip(online,expected):
                np.testing.assert_array_equal(t.tlwh,r['bbox']);assert float(t.score)==r['score'];used.add(i)
                nk=(cam,int(t.track_id));ok=(cam,r['local_id'])
                assert identity_map.setdefault(nk,ok)==ok and reverse.setdefault(ok,nk)==nk
                cached[i]=t.curr_feat
        print('SCT_CACHE',cam,len(used),flush=True)
    assert len(used)==len(old) and np.isfinite(cached).all()
    np.save(O/'sct_current_features.npy',cached);save(O/'sct_rows.json',old)
    save(O/'cache_verification.json',{'status':'PASS','rows':len(old),'shape':list(cached.shape),'bbox_and_score':'exactly equal to prior fps10 SCT for every row',
      'track_partitions':'bijective native ID mapping to previous local IDs','saved_ids':'prior IDs retained','gt_used':False,
      'why_replayed':'previous SCT JSON omitted current features; detector/ReID were NOT rerun','elapsed_seconds':time.monotonic()-start})
def configs():return [(10000,80,'baseline')]+[(a,80,'area') for a in [5000,2500,1000]]+[(10000,d,'distance') for d in [60,40,20]]
def key(area,distance):return f'area_{area}_distance_{distance}'
def generate(combo=False):
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import AgglomerativeClustering
    ds,features,bykey=inputs();tracks=load(O/'sct_rows.json');sctfeat=np.load(O/'sct_current_features.npy',mmap_mode='r')
    assert len(tracks)==len(sctfeat)
    conf=configs() if not combo else [(load(O/'selection.json')['selected_area'],load(O/'selection.json')['selected_distance'],'combined')]
    candidate_stats=[]
    for seconds,n in [(5,50),(30,300)]:
      frames=sorted(set(np.linspace(0,n-1,11,dtype=int).tolist()));cur=[(i,r) for i,r in enumerate(tracks) if r['frame']<n]
      lens=collections.Counter((r['camera'],r['local_id']) for _,r in cur);eligible=[(i,r) for i,r in cur if lens[r['camera'],r['local_id']]>=4]
      groups=collections.defaultdict(list)
      for i,r in eligible:groups[r['camera'],r['frame']].append((i,r))
      for area,distance,family in conf:
        dest=O/'experiments'/key(area,distance)/f's{seconds}'
        if combo and (dest/'summary.json').exists():
            print('REUSE_COMBINED',area,distance,seconds,flush=True);continue
        start=time.monotonic();members=[];afeat=[];areas=[];scores=[];sampled_areas=[];sampled_scores=[]
        for cam in range(8):
          for frame in range(n):
            d,ft,off=bykey[f'cam_{cam}',frame]
            for j,b in enumerate(d['selected']):
              size=(b[2]-b[0])*(b[3]-b[1]);areas.append(size);scores.append(b[4])
              if frame not in frames:continue
              sampled_areas.append(size);sampled_scores.append(b[4])
              if size>area and b[4]>.95:
                members.append({'camera':cam+1,'frame':frame,'source_frame':277+frame,'detection_index':j,'feature_row':off+j,
                    'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]]});afeat.append(ft[j])
        af=np.asarray(afeat);assert len(af)>=2
        estimate=AgglomerativeClustering(distance_threshold=distance,n_clusters=None).fit(af);count=int(estimate.labels_.max()+1)
        labels=AgglomerativeClustering(n_clusters=count).fit(af).labels_;anf=af/np.linalg.norm(af,axis=1,keepdims=True)
        centers=np.stack([anf[labels==k].mean(0) for k in range(count)])
        for r,l in zip(members,labels):r['anchor_id']=int(l+1)
        assigned=[];unassigned=[]
        for _,irs in sorted(groups.items()):
            feat=np.stack([sctfeat[i] for i,r in irs]);feat/=np.linalg.norm(feat,axis=1,keepdims=True)
            cost=1-feat@centers.T;ri,ci=linear_sum_assignment(cost)
            for i,j in zip(ri,ci):
                r=dict(irs[i][1]);r.update(raw_anchor_id=int(j+1),id=int(j+1),assignment_cost=float(cost[i,j]));assigned.append(r)
            for i in set(range(len(irs)))-set(ri):unassigned.append(dict(irs[i][1]))
        ts=collections.defaultdict(list)
        for r in assigned:ts[r['camera'],r['local_id']].append(r)
        changed=0
        for rs in ts.values():
            rs.sort(key=lambda r:r['frame']);last=None
            for start_index in range(0,len(rs),5):
                chunk=rs[start_index:start_index+5]
                choice=last if len(chunk)<5 and last is not None else collections.Counter(r['raw_anchor_id'] for r in chunk).most_common(1)[0][0]
                for r in chunk:r['id']=choice;changed+=int(choice!=r['raw_anchor_id'])
                last=choice
        assigned.sort(key=lambda r:(r['camera'],r['frame'],r['local_id']))
        assert len(assigned)+len(unassigned)==len(eligible)
        save(dest/'anchor_members.json',members);save(dest/'anchor_predictions.json',assigned);save(dest/'hungarian_unassigned.json',unassigned)
        np.save(dest/'anchor_mean_unit_features.npy',centers)
        ar=np.array(areas);sc=np.array(scores);sa=np.array(sampled_areas);ss=np.array(sampled_scores)
        stat={'seconds':seconds,'area_threshold':area,'distance_threshold':distance,'full_detections':len(ar),'full_bbox_area_median':float(np.median(ar)),
           'full_area_pass':int((ar>area).sum()),'full_score_pass':int((sc>.95).sum()),'full_joint_pass':int(((ar>area)&(sc>.95)).sum()),
           'full_joint_pass_ratio':float(((ar>area)&(sc>.95)).mean()),'sampled_detections':len(sa),'sampled_bbox_area_median':float(np.median(sa)),
           'sampled_joint_pass':len(members),'sampled_joint_pass_ratio':len(members)/len(sa)}
        candidate_stats.append(stat);save(dest/'candidate_filter_stats.json',stat)
        cfg={'config_id':key(area,distance),'family':family,'variant':'fps10','seconds_nominal':seconds,'frames_per_camera':n,
          'fps_argument':10,'effective_lost_buffer':10,'majority_vote_window':5,'short_track_min_observations':4,'score_threshold_strict':.95,
          'area_threshold_strict':area,'clustering_raw_euclidean_threshold':distance,'cluster_method':'Ward on raw concatenated 2560d features; native two-pass count then fit',
          'anchor_frames':frames,'anchor_observations':len(members),'anchors':count,'sct_rows':len(cur),'eligible_rows':len(eligible),
          'hungarian_unassigned':len(unassigned),'anchor_rows':len(assigned),'majority_changed_rows':changed,'gt_used':False,'stcra_run':False,
          'elapsed_seconds':time.monotonic()-start}
        save(dest/'summary.json',cfg)
        if area==10000 and distance==80:
            ref=B/f'fps10/s{seconds}'
            assert members==load(ref/'anchor_members.json'),'baseline member mismatch'
            assert assigned==load(ref/'anchor_predictions.json'),'baseline prediction mismatch'
            assert unassigned==load(ref/'hungarian_unassigned.json'),'baseline unassigned mismatch'
            save(dest/'baseline_equivalence.json',{'status':'PASS','members_predictions_unassigned':'exact JSON equality to previous fps10 run'})
        print('ANCHOR',seconds,area,distance,'members',len(members),'anchors',count,'unassigned',len(unassigned),flush=True)
    save(O/('combined_candidate_statistics.json' if combo else 'candidate_statistics.json'),candidate_stats)
def evaluate():
    ev=mod('/project/scripts/evaluate-uwipl-study.py','previous_evaluator');truth=collections.defaultdict(list)
    for r in load(Path('/gt.json')):truth[int(r['camera_id'].split('_')[1])+1,r['source_frame']-277].append({'id':r['track_id'],'bbox':r['bbox_xywh'],'world':r['world_xyz']})
    summary=[]
    for work in sorted((O/'experiments').glob('*/s*')):
      cfg=load(work/'summary.json');seconds=cfg['seconds_nominal'];n=cfg['frames_per_camera']
      if (work/'anchor_metrics.json').exists() and (cfg['config_id']!=key(10000,80) or (work/'baseline_metric_equivalence.json').exists()):
          summary.append(load(work/'anchor_metrics.json')['summary']);continue
      pred=load(work/'anchor_predictions.json');members=load(work/'anchor_members.json');am=ev.gt_matches(members,truth)
      votes=collections.defaultdict(collections.Counter)
      for i,pair in am.items():votes[members[i]['anchor_id']][pair['gt']['id']]+=1
      majority={k:v.most_common(1)[0][0] for k,v in votes.items()};sizes=collections.Counter(m['anchor_id'] for m in members)
      per_anchor=[]
      for k in sorted(sizes):
        vc=votes.get(k,collections.Counter());labelled=sum(vc.values());per_anchor.append({'anchor_id':k,'members':sizes[k],'matched_members':labelled,'unmatched_members':sizes[k]-labelled,
          'distinct_gt_people':len(vc),'majority_gt':majority.get(k),'majority_count':max(vc.values()) if vc else 0,
          'purity_matched_only':max(vc.values())/labelled if labelled else None,'gt_distribution':dict(vc)})
      audit={'anchor_majority_gt_map_evaluation_only':majority,'member_vote_counts':votes,'GT_not_used_for_anchors':True,
          'per_anchor':per_anchor,'match_protocol':'one-to-one camera/frame IoU>=0.5, same as previous; unmatched excluded from purity',
          'GT_used_for_parameter_selection':True,'development_only':True}
      save(work/'anchor_gt_audit.json',audit)
      membership=[]
      for i,m in enumerate(members):
        p=am.get(i);membership.append(dict(m,matched_gt=None if p is None else p['gt']['id'],gt_iou=None if p is None else p['iou']))
      save(work/'anchor_member_gt_matches.json',membership)
      metrics,detail,removed=ev.metric_suite(pred,truth,n)
      gtids={r['id'] for (c,f),rs in truth.items() if f<n for r in rs};labelled=sum(sum(v.values()) for v in votes.values())
      metrics.update(config_id=cfg['config_id'],family=cfg['family'],variant='fps10',seconds=seconds,schedule='not_run',stage='anchor',
        area_threshold=cfg['area_threshold_strict'],distance_threshold=cfg['clustering_raw_euclidean_threshold'],raw_rows=len(pred),
        anchors=cfg['anchors'],anchor_observations=len(members),hungarian_unassigned=cfg['hungarian_unassigned'],eligible_rows=cfg['eligible_rows'],
        gt_unique_people=len(gtids),anchor_gt_covered_people=len({p['gt']['id'] for p in am.values()}),anchor_gt_matched_members=labelled,
        anchor_gt_weighted_purity=sum(max(v.values()) for v in votes.values() if v)/labelled if labelled else None,
        anchors_with_multiple_gt=sum(len(v)>1 for v in votes.values()),max_gt_people_in_one_anchor=max(map(len,votes.values()),default=0),
        majority_changed_rows=cfg['majority_changed_rows'])
      save(work/'anchor_metrics.json',{'summary':metrics,'detail':detail});save(work/'anchor_eval_duplicate_removals.json',removed)
      csvout(work/'anchor_predictions.csv',[{'sequence':'sequence_1','camera':f"cam_{r['camera']-1}",'frame':277+r['frame'],'model_frame':r['frame'],
        'id':r['id'],'x':r['bbox'][0],'y':r['bbox'][1],'width':r['bbox'][2],'height':r['bbox'][3],'score':r['score']} for r in pred])
      if cfg['config_id']==key(10000,80):
        old=load(B/f'fps10/s{seconds}/meter_candidate/anchor_metrics.json')['summary']
        for k in ['IDF1_2D','HOTA_2D','IDSW_2D','IDF1_view_aware','HOTA_view_aware','rows_evaluated','duplicate_id_rows_removed_for_eval']:
            assert metrics[k]==old[k],(k,metrics[k],old[k])
        previous=load(B/f'fps10/s{seconds}/anchor_gt_audit.json')
        assert clean(votes)==previous['member_vote_counts']
        save(work/'baseline_metric_equivalence.json',{'status':'PASS','metrics_and_gt_distribution':'exact match to previous'})
      summary.append(metrics);print('METRICS',seconds,cfg['config_id'],metrics['IDF1_view_aware'],metrics['HOTA_2D'],metrics['IDSW_2D'],flush=True)
    if (O/'selection.json').exists():
        choice=load(O/'selection.json');combo=key(choice['selected_area'],choice['selected_distance'])
        base=[r for r in summary if r['family']!='combined']
        combined=[dict(r,family='combined',reused_existing_config=any(r['config_id']==q['config_id'] for q in base)) for r in summary if r['config_id']==combo]
        summary=base+combined
    csvout(O/'experiment_comparison.csv',summary)
def choose():
    with (O/'experiment_comparison.csv').open() as f:rs=list(csv.DictReader(f))
    rankings={}
    for family,col,values in [('area','area_threshold',[10000,5000,2500,1000]),('distance','distance_threshold',[80,60,40,20])]:
      rank=[]
      for value in values:
        subset=[r for r in rs if int(r[col])==value and int(r['distance_threshold' if family=='area' else 'area_threshold'])==(80 if family=='area' else 10000)]
        assert len(subset)==2
        rank.append({'value':value,'mean_IDF1_view_aware':float(np.mean([float(r['IDF1_view_aware']) for r in subset])),
          'mean_HOTA_2D':float(np.mean([float(r['HOTA_2D']) for r in subset])),'sum_IDSW_2D':sum(int(r['IDSW_2D']) for r in subset)})
      rank.sort(key=lambda x:(-x['mean_IDF1_view_aware'],-x['mean_HOTA_2D'],x['sum_IDSW_2D'],-x['value']));rankings[family]=rank
    save(O/'selection.json',{'selected_area':rankings['area'][0]['value'],'selected_distance':rankings['distance'][0]['value'],
      'rule':'equal-weight mean view-aware IDF1 across 5s and 30s; ties mean HOTA2D, smaller summed IDSW, then larger original-like threshold',
      'rankings':rankings,'GT_used_for_selection':True,'scope':'development tuning; these same clips are not independent test data'})
    print('SELECTED',load(O/'selection.json'),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cache','generate','evaluate','choose']);ap.add_argument('--combined',action='store_true');a=ap.parse_args()
    if a.mode=='generate':generate(a.combined)
    else:globals()[a.mode]()
