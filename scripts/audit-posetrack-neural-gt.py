#!/usr/bin/env python3
"""Posthoc GT association for neural observations, including unconfirmed recoveries."""
import collections,importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('eval','/project/scripts/evaluate-uwipl-study.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
p=Path('/project/outputs/posetrack-mesh-20261104');obs=json.loads((p/'full_input30/observations.json').read_text());truth=collections.defaultdict(list)
for r in json.loads(Path('/project/outputs/osc-analysis-20261101/s30_th40/gt_selected.json').read_text()):truth[(int(r['camera_id'].split('_')[1])+1,r['source_frame']-277)].append({'id':r['track_id'],'bbox':r['bbox_xywh']})
rows=[]
for r in obs:
 b=r['raw_bbox'];rows.append(dict(camera=r['camera']+1,frame=r['frame'],bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]]))
m=E.gt_matches(rows,truth);out=[]
for i,r in enumerate(obs):
 a=m.get(i);out.append(dict(observation_id=f"cam{r['camera']}_row{r['camera_row']}",camera=r['camera'],source_frame=r['source_frame'],
  matched_gt_id='' if a is None else a['gt']['id'],gt_iou='' if a is None else a['iou']))
E.csvout(Path('/out/association/neural_observation_gt.csv'),out);print('Posthoc neural GT matches',len(m),'/',len(rows))
