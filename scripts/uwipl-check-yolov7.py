#!/usr/bin/env python3
"""Load each provided YOLOv7 checkpoint and infer one SCOUT frame; no training."""
import json,time
from pathlib import Path
import torch,numpy as np
from mmdet.apis import init_detector,inference_detector
from mmyolo.utils import register_all_modules

register_all_modules();torch.set_num_threads(2)
out=Path('/out');cfg='/opt/uwipl/mmyolo/configs/custom_dataset/yolov7_base.py';reports=[]
for p in sorted(Path('/weights/synthetic').glob('*/*.pth')):
    start=time.monotonic()
    try:
        net=init_detector(cfg,checkpoint=None,device='cpu')
        state=torch.load(p,map_location='cpu');state=state.get('state_dict',state.get('model',state))
        state={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        result=net.load_state_dict(state,strict=True);net.cuda().eval()
        pred=inference_detector(net,'/inference/input/cam_0/frame_000000.jpg').pred_instances
        assert torch.isfinite(pred.bboxes).all() and torch.isfinite(pred.scores).all()
        reports.append({'file':str(p),'status':'PASS','matched_keys':len(state),'missing_keys':list(result.missing_keys),
            'unexpected_keys':list(result.unexpected_keys),'predictions_score_gt_01':int((pred.scores>.1).sum()),
            'elapsed_seconds':time.monotonic()-start,'used_for_tracking':False})
        del net,state;torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        reports.append({'file':str(p),'status':'FAIL','error':traceback.format_exc(),'used_for_tracking':False})
    print(reports[-1],flush=True)
(out/'yolov7_load_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
