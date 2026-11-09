#!/usr/bin/env python3
"""CPU-only native YOLOX input validation; no optimizer/model training."""
import json
from pathlib import Path
import numpy as np
from yolox.data import MOTDataset,TrainTransform
out=Path('/out');records=[]
for name in ['train','internal_val']:
 ds=MOTDataset(data_dir='/train',json_file=name+'.json',name='',img_size=(800,1440),preproc=None)
 seen=set()
 for i,(_,info,_) in enumerate(ds.annotations):
  camera=info[3]
  if camera in seen:continue
  seen.add(camera);image,labels,meta,ids=ds.pull_item(i)
  assert image.shape==(1080,1920,3) and np.isfinite(labels).all() and (labels[:,4]==0).all()
  transformed,target=TrainTransform(rgb_means=(.485,.456,.406),std=(.229,.224,.225),max_labels=500)(image,labels,(800,1440))
  assert transformed.shape==(3,800,1440) and target.shape==(500,6) and np.isfinite(transformed).all() and np.isfinite(target).all()
  records.append(dict(split=name,camera=camera-1,source_file=meta[-1],boxes=len(labels),transformed_image_shape=list(transformed.shape),target_shape=list(target.shape)))
 assert len(seen)==22
(out/'validation/training_loader.json').write_text(json.dumps(dict(status='PASS',model_training_run=False,samples=len(records),records=records),indent=2)+'\n')
print('PASS native training loader',len(records),'samples; no training')
