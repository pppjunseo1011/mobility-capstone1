#!/usr/bin/env python3
"""Run native STCRA separately, with trace instrumentation and explicit SCOUT H.

GT is deliberately absent. Raw-image BBoxes are retained; only the chosen foot
point is undistorted before H inverse. Z=0 remains an experimental assumption.
"""
import collections,copy,csv,difflib,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def mod(p,n):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def flat(ts):
    return {int(loc[7]):{'camera':int(cam[1:]),'id':int(tid),'frame':int(fid),'pose_row':int(loc[7]),
        'bbox':[float(x) for x in loc[:4]],'score':float(loc[6])}
        for cam,tr in ts.items() for tid,fr in tr.items() for fid,ls in fr.items() for loc in ls}

def main():
    out=Path('/out');repo=Path('/opt/uwipl');kp=np.load(out/'sct_keypoints.npy',allow_pickle=False)
    sys.path.insert(0,str(repo/'STCRA'))
    source=(repo/'STCRA/run_stcra.py').read_text()
    start=source.index('    cam_weight = {');end=source.index('    avg_tracks =',start)
    patched=source[:start]+"    cam_weight = {f'c{i:03d}': 1.0 for i in range(1,9)}  # SCOUT: no AIC view-area priors\n"+source[end:]
    p=out/'instrumented/run_stcra_scout.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(patched)
    (p.parent/'stcra.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='upstream/STCRA/run_stcra.py',tofile='instrumented/run_stcra_scout.py')))
    st=mod(p,'stcra_scout');fu=mod(repo/'STCRA/file_utils.py','native_feet');H={};cal={};geom=[]
    for cam in range(1,9):
        c=json.loads(Path(f'/data/calibrations/cam_{cam-1}.json').read_text());cal[cam]=c
        K,R,T=[np.array(c[k],float) for k in ['K','R','T']]
        h=K@np.column_stack([R[:,:2],T.reshape(3)])
        assert np.linalg.matrix_rank(h)==3
        H[f'c{cam:03d}']=h
        geom.append({'camera':cam,'scout_camera':f'cam_{cam-1}','H_world_to_undistorted_image':h.tolist(),
            'det':float(np.linalg.det(h)),'condition':float(np.linalg.cond(h)),
            'plane':'Z=0 in provided SCOUT world coordinates','physical_calibration_validated':False})
    st.cam_utils.HOMO={'S001':H}
    save(out/'homographies.json',geom)
    summary=[]
    for variant in ['original_frames','fps10']:
      for seconds in [5,30]:
        work=out/variant/f's{seconds}';conf=json.loads((work/'summary.json').read_text())
        pred=json.loads((work/'anchor_predictions.json').read_text());base={r['pose_row']:r for r in pred}
        feet=work/'feet';feet.mkdir(exist_ok=True);ts={};foot_records=[]
        for cam in range(1,9):
            rs=[r for r in pred if r['camera']==cam];fp=feet/f'S001_c{cam:03d}.txt'
            with fp.open('w') as f:
                for r in rs:
                    ankles=kp[r['pose_row'],[15,16]].ravel().tolist()
                    f.write(','.join(map(str,[r['frame'],r['id'],*r['bbox'],r['score'],-1,-1,-1,*ankles]))+'\n')
            camts=fu.load_tracking(str(fp),enable_feet=True) if rs else collections.defaultdict(dict)
            lookup=collections.defaultdict(list)
            for r in rs:lookup[(r['frame'],r['id'],tuple(r['bbox']))].append(r['pose_row'])
            for tid,tr in camts.items():
              for frame,locs in tr.items():
                for loc in locs:
                    row=lookup[(frame,tid,tuple(loc[:4]))].pop(0);c=cal[cam];K=np.array(c['K'],float);dist=np.array(c['dist'],float)
                    raw=np.array(loc[4:6],float)
                    und=cv2.undistortPointsIter(raw.reshape(1,1,2),K,dist,None,K,
                        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2)
                    assert np.isfinite(und).all()
                    loc[4:6]=und.tolist();loc.append(row)
                    w=st.cam_utils.image2world(und.tolist(),H[f'c{cam:03d}'])
                    assert np.isfinite(w).all()
                    foot_records.append({'pose_row':row,'camera':cam,'frame':frame,'source_frame':277+frame,
                        'ankle_confidence_pass':bool((kp[row,[15,16],2]>.5).all()),
                        'raw_foot_xy':raw.tolist(),'undistorted_foot_xy':und.tolist(),'world_xy':list(w),
                        'bbox':loc[:4],'global_id_anchor':tid})
            ts[f'c{cam:03d}']=camts
        save(work/'foot_projection.json',foot_records)
        for schedule,thresholds in [('original_units',[250.,200.,150.]),('meter_candidate',[2.5,2.,1.5])]:
            dest=work/schedule;dest.mkdir(exist_ok=True);state=copy.deepcopy(ts)
            save(dest/'anchor.json',pred)
            rounds=[];all_events=[]
            for i,(threshold,confidence) in enumerate(zip(thresholds,[.65,.7,.75]),1):
                before=flat(state);captured={}
                def trace(frame,event,arg):
                    if frame.f_code.co_name=='id_reassignment' and frame.f_code.co_filename==str(p):
                        if event=='return':
                            for k in ['reassign_duplicate_ids','reassign_outliers','case1_cnt','case2_cnt','cnt_all','cnt_same']:
                                captured[k]=copy.deepcopy(frame.f_locals.get(k))
                        return trace
                start=time.monotonic();sys.settrace(trace)
                try:state,proposals,avg=st.id_reassignment(state,dis_thr=threshold,conf_thr=confidence)
                finally:sys.settrace(None)
                after=flat(state);events=[]
                for key,r in before.items():
                    a=after.get(key)
                    if a is None or a['id']!=r['id']:
                        events.append({'round':i,'pose_row':key,'camera':r['camera'],'frame':r['frame'],
                            'source_frame':277+r['frame'],'old_id':r['id'],'new_id':None if a is None else a['id'],
                            'action':'deleted' if a is None else 'reassigned'})
                views=collections.defaultdict(set)
                for r in before.values():views[(r['id'],r['frame'])].add(r['camera'])
                ge3=[(tid,f) for (tid,f),v in views.items() if len(v)>=3]
                rec={'round':i,'distance_threshold':threshold,'confidence_threshold':confidence,'input_rows':len(before),
                    'output_rows':len(after),'outlier_observations':len(captured['reassign_outliers']),
                    'duplicate_reassign_candidates':len(captured['reassign_duplicate_ids']),
                    'proposals':len(proposals),'actual_reassigned':sum(e['action']=='reassigned' for e in events),
                    'actual_deleted':sum(e['action']=='deleted' for e in events),
                    'id_frame_groups':len(views),'id_frame_groups_3plus_views':len(ge3),
                    'frames_with_any_3view_id':len({f for _,f in ge3}),
                    'frame_ratio_3plus_views':len({f for _,f in ge3})/conf['frames_per_camera'],
                    'seconds':time.monotonic()-start}
                rounds.append(rec);all_events.extend(events)
                save(dest/f'round_{i}_proposals.json',proposals)
                save(dest/f'round_{i}_outliers.json',captured['reassign_outliers'])
                save(dest/f'round_{i}_avg_tracks.json',avg)
                save(dest/f'round_{i}.json',list(after.values()))
                if i==1:save(dest/'stcra.json',list(after.values()))
                print('STCRA',variant,seconds,schedule,rec,flush=True)
            save(dest/'istcra.json',list(flat(state).values()));save(dest/'events.json',all_events);save(dest/'rounds.json',rounds)
            # Native interpolation; its global args lookup is explicitly configured.
            interp=mod(repo/'BoT-SORT/tools/aic_interpolation.py','interp')
            interp.args=argparse.Namespace(distance_thres=200)
            interpolated=[];interpolation_changes=[]
            native_dir=dest/'native';native_dir.mkdir(exist_ok=True)
            for cam,tr in state.items():
                inp=native_dir/f'S001_{cam}.txt';op=native_dir/f'S001_{cam}_interpolated.txt'
                # Native writer requires one row per ID/frame; duplicates are resolved by native dti instead.
                data=[[fid,tid,*loc[:4],loc[6],-1,-1,-1] for tid,fr in tr.items() for fid,ls in fr.items() for loc in ls]
                np.savetxt(inp,np.array(data).reshape(-1,10),delimiter=',')
                if len(data)<2:
                    arr=np.array(data).reshape(-1,10);np.savetxt(op,arr,delimiter=',')
                else:
                    interp.dti(str(inp),str(op),n_min=conf['interpolation_n_min'],n_dti=conf['interpolation_n_dti'],distance_thres=200)
                    arr=np.loadtxt(op,delimiter=',',ndmin=2) if op.stat().st_size else np.empty((0,10))
                existing={(int(r[0]),int(r[1])) for r in data};added=sum((int(r[0]),int(r[1])) not in existing for r in arr)
                interpolation_changes.append({'camera':cam,'input':len(data),'output':len(arr),'added':added,
                    'removed_duplicate_observations':len(data)+added-len(arr)})
                for r in arr:interpolated.append({'camera':int(cam[1:]),'frame':int(r[0]),'source_frame':277+int(r[0]),'id':int(r[1]),'bbox':r[2:6].tolist(),'score':float(r[6])})
            save(dest/'interpolation.json',interpolated);save(dest/'interpolation_changes.json',interpolation_changes)
            summary.append({'variant':variant,'seconds':seconds,'schedule':schedule,'rounds':rounds,
                'foot_rows':len(foot_records),'feet_pass':sum(r['ankle_confidence_pass'] for r in foot_records),
                'bbox_fallback':sum(not r['ankle_confidence_pass'] for r in foot_records),
                'distance_unit':'SCOUT dataset coordinate unit; meter_candidate assumes 1 unit = 1 m, not an established AIC map conversion'})
    save(out/'stcra_summary.json',summary)

if __name__=='__main__':
    import argparse
    main()
