#!/usr/bin/env python3
"""Explicit GT-box oracle: replacement detections with real-image ReID inference."""
import collections,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v3-20261107/oracles/GT_boxes';P=R/'outputs/uwipl-study-20261101/inference30'
gt=json.loads((R/'outputs/osc-analysis-20261101/s30_th40/gt_selected.json').read_text());groups=collections.defaultdict(list)
for g in gt:groups[g['camera_id'],g['source_frame']].append(g)
manifest=json.loads((P/'input/manifest.json').read_text());manifest.update(gt_used=True,evaluation_only=True,labels_used_as_detections=True)
ds=[];invalid=0
for f in manifest['frames']:
 boxes=[]
 for g in groups[f['camera'],f['source_frame']]:
  x,y,w,h=g['bbox_xywh'];a=max(0,x);b=max(0,y);c=min(1920,x+w);d=min(1080,y+h)
  if c<=a or d<=b:invalid+=1;continue
  boxes.append([a,b,c,d,1.,1.,0.])
 ds.append(dict(f,detections=boxes,selected=boxes,evaluation_only=True,gt_used=True))
(O/'inference/input').mkdir(parents=True,exist_ok=True);(O/'inference/uwipl/detection').mkdir(parents=True,exist_ok=True);(O/'inference/uwipl/reid').mkdir(parents=True,exist_ok=True)
(O/'inference/input/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(O/'inference/uwipl/detection/detections.json').write_text(json.dumps(ds,indent=2)+'\n')
# Same image files are mounted from the old inference input, with this manifest overriding it.
source=(R/'scripts/scout-sample-stage.py').read_text().replace("'gt_used': False","'gt_used': True, 'evaluation_only': True")
(O/'oracle-reid-runner.py').write_text(source)
(O/'protocol.json').write_text(json.dumps(dict(evaluation_only=True,GT_used_in_inference=True,normal_results_comparable=False,definition='GT boxes clipped to image with confidence1; real OSNet image-crop embeddings; unchanged selected BoTSORT/anchor/vote settings; no GT identities passed to tracker',gt_rows=len(gt),input_rows=sum(len(d['selected']) for d in ds),invalid_outside_image_skipped=invalid,limitation='GT confidence=1 removes score filtering losses; not a mathematically guaranteed upper bound'),indent=2)+'\n')
print('Prepared GT-box oracle',len(gt),'GT rows; skipped',invalid,flush=True)
