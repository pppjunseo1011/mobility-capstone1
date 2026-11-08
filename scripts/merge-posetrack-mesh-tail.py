#!/usr/bin/env python3
"""Join original 5s observations and new 25s pose/ReID, without rerunning 5s."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/project');O=Path('/out');tail=O/'inference_s30_tail/posetrack';old=R/'outputs/posetrack-transfer-20261101/handoff/03_intermediate/result'
def main():
    assert json.loads((tail/'pose/report.json').read_text())['status']=='PASS'
    assert json.loads((tail/'reid/report.json').read_text())['status']=='PASS'
    features=np.load(tail/'reid/aic24.npy');mapping=json.loads((tail/'reid/aic24-rows.json').read_text());assert len(features)==len(mapping)
    assert np.isfinite(features).all() and np.all(np.linalg.norm(features,axis=1)>0)
    hashes={}
    for cam in range(8):
        name=f'camera_{cam+1:04d}';ds=np.loadtxt(old/f'detection/scene_001/{name}.txt',delimiter=',',ndmin=2)
        ps=np.loadtxt(old/f'pose/scene_001/{name}.txt',ndmin=2);fe=np.load(old/f'reid/scene_001/{name}.npy')
        dt=np.loadtxt(tail/f'pose/cam_{cam}-det7.txt',delimiter=',',ndmin=2);pt=np.loadtxt(tail/f'pose/cam_{cam}-pose57.txt',ndmin=2)
        idx=[i for i,r in enumerate(mapping) if r['camera']==f'cam_{cam}'];ft=features[idx]
        assert len(dt)==len(pt)==len(ft) and dt[:,0].min()==50 and dt[:,0].max()==299
        assert [mapping[i]['source_frame']-277 for i in idx]==dt[:,0].tolist()
        for stage,a,ext in [('detection',np.concatenate([ds,dt]),'.txt'),('pose',np.concatenate([ps,pt]),'.txt'),('reid',np.concatenate([fe,ft]),'.npy')]:
            p=O/f'native30/{stage}/scene_001/{name}{ext}';p.parent.mkdir(parents=True,exist_ok=True)
            if ext=='.npy':np.save(p,a)
            else:np.savetxt(p,a,delimiter=',' if stage=='detection' else ' ')
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        print('MERGED',cam,len(ds),len(dt),flush=True)
    (O/'reproducibility/native30_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
if __name__=='__main__':main()
