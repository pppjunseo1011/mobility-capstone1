#!/usr/bin/env python3
"""Inspect official training initializer without training or replacing aic24.pkl."""
import json
from pathlib import Path
import torch
from fastreid.config import get_cfg
from fastreid.modeling.backbones.resnet import build_resnet_backbone

torch.set_num_threads(2)
path = Path('/weights/lup_moco_r101.pth')
state = torch.load(path, map_location='cpu')
assert isinstance(state, dict)
cfg = get_cfg()
cfg.merge_from_file('/opt/posetrack/fast-reid/configs/AIC24/mgn_R101_reprod.yml')
cfg.MODEL.BACKBONE.PRETRAIN = False
backbone = build_resnet_backbone(cfg)
own = backbone.state_dict()
shape_bad = [k for k in state if k in own and own[k].shape != state[k].shape]
assert not shape_bad, shape_bad
incompat = backbone.load_state_dict(state, strict=False)
matched = [k for k in state if k in own]
assert matched
report = {'file':path.name, 'status':'PASS' if not incompat.missing_keys and not incompat.unexpected_keys else 'FAIL',
    'config':'configs/AIC24/mgn_R101_reprod.yml (script/reid_train.sh)',
    'purpose':'LUP training initialization; NOT used in aic24.pkl inference',
    'checkpoint_keys':len(state), 'matched_keys':len(matched), 'shape_mismatches':shape_bad,
    'missing_keys':incompat.missing_keys, 'unexpected_keys':incompat.unexpected_keys,
    'training_started':False, 'inference_checkpoint_replaced':False}
Path('/out/lup_load.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
