#!/usr/bin/env python3
"""Independently replay native tracker without tracing and check geometry/logs."""
import csv,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,'/opt/posetrack/track');sys.path.insert(0,'/project/adapters')
from Tracker.PoseTracker import PoseTracker,Detection_Sample
from util.camera import Camera
from scout_adapter.geometry import CameraGeometry
O=Path('/out');D=Path('/data')
def main():
    cams=[Camera(str(O/f'calibration/camera_{i+1:04d}/calibration.json')) for i in range(8)]
    tracker=PoseTracker(cams);data=[]
    for c in range(8):
        p=O/f'input/camera_{c+1:04d}';d=np.load(p/'detection.npy');po=np.load(p/'pose.npy');f=np.load(p/'reid.npy')
        if len(f):f=f/np.linalg.norm(f,axis=1,keepdims=True)
        data.append((d,po,f))
    predictions=[];snap=json.loads((O/'track_snapshots.json').read_text())
    for frame in range(50):
        ds=[]
        for c,(d,p,f) in enumerate(data):
            ds.append([Detection_Sample(d[i,2:].copy(),p[i,6:].reshape(17,3).copy(),f[i].copy(),c,frame) for i in np.flatnonzero(d[:,0]==frame)])
        tracker.mv_update_wo_pred(ds,frame)
        predictions.extend(a[0] for a in tracker.output(frame))
        expected=snap[frame]['after'];assert len(tracker.tracks)==len(expected)
        for t,ex in zip(tracker.tracks,expected):
            assert t.id==ex['id'] and t.state==ex['state'] and t.feat_count==ex['feat_count']
            assert [x.state for x in t.track2ds]==ex['view_states']
    expected=[]
    for r in csv.DictReader((O/'global_predictions.csv').open()):
        expected.append([int(r['camera'])+1,int(r['global_id']),int(r['frame']),*[float(r[k]) for k in ['x','y','w','h','plane_x','plane_y','output_method_code']]])
    np.testing.assert_array_equal(np.array(predictions),np.array(expected))
    allerr=[];normerr=[];W=np.array(json.loads((O/'plane.json').read_text())['W_plane_to_scout'])
    for r in json.loads((O/'observations.json').read_text()):
        c=r['camera'];g=CameraGeometry(json.loads((D/f'calibrations/cam_{c}.json').read_text()),1920,1080)
        raw=np.array(r['keypoints_raw'])[:,:2];kp=data[c][1][r['input_row'],6:].reshape(17,3)[:,:2]
        ideal=(np.linalg.inv(g.k)@np.c_[kp,np.ones(17)].T).T
        redist=cv2.projectPoints(ideal.reshape(-1,1,3),np.zeros(3),np.zeros(3),g.k,g.dist)[0].reshape(-1,2)
        allerr.extend(np.linalg.norm(redist-raw,axis=1))
        point=np.linalg.solve(W,np.r_[r['plane_world'],1]);p=cams[c].project_mat@point;p=p[:2]/p[2]
        h=cams[c].homo_mat@np.r_[point[:2],1];h=h[:2]/h[2];normerr.append(np.linalg.norm(h-p))
    assert max(allerr)<1e-3 and max(normerr)<1e-6
    report={'status':'PASS','native_uninstrumented_output':'bitwise equal, all 416 rows / 10 fields',
      'native_state_and_bank_counts':'equal at all 50 frame boundaries',
      'keypoint_redistortion_max_px':float(max(allerr)),'P_H_plane_consistency_max_px':float(max(normerr)),
      'geometry_cpp_equivalence':'2255 actual pairs checked by trace runner',
      'interpretation':'numerical/input/trace consistency only; not independent world-location or ID accuracy'}
    (O/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
