#!/usr/bin/env python3
"""Prepare sequence-disjoint detector training annotations; never launch training."""
import argparse,collections,csv,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,default=R/'datasets/scout/extracted/scout');p.add_argument('--output',type=Path,default=R/'outputs/uwipl-v4-20261108/D4_prepared');a=p.parse_args();D=a.dataset_root;O=a.output
assert not (O/'preparation.json').exists(),'Preserve an existing prepared split; choose a new --output'
cams=[0,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
ts=json.loads((D/'timestamps/timestamps_sequence_2.json').read_text());obs=[]
for cam in cams:
 # sequence_2 timestamp keys are camN; do not reuse sequence_1 device-name keys.
 key=f'cam{cam}';seen=set()
 for fi,v in sorted(ts.items(),key=lambda x:int(x[0])):
  if key not in v:continue
  t=float(v[key]);f=int(fi)
  if not math.isfinite(t) or t in seen:continue
  seen.add(t)
  if (D/f'images/sequence_2/cam_{cam}/image_{f}.jpg').is_file():obs.append(dict(camera=cam,frame=f,timestamp=t))
assert obs,'No sequence_2 timestamps/images found'
t0=min(x['timestamp'] for x in obs);t1=max(x['timestamp'] for x in obs);train_end=t0+.75*(t1-t0);val_start=t0+.85*(t1-t0)
assert val_start-train_end>=30
# One available capture nearest to each integer-second target per camera.
buckets={}
for x in obs:
 slot=round(x['timestamp']-t0);key=(x['camera'],slot);error=abs(x['timestamp']-(t0+slot))
 if error>.1:continue
 if key not in buckets or error<buckets[key][0]:buckets[key]=(error,x)
selected=sorted((x for e,x in buckets.values()),key=lambda x:(x['camera'],x['frame']))
splits={name:dict(images=[],annotations=[],categories=[dict(id=1,name='person')],videos=[dict(id=c+1,file_name=f'sequence_2/cam_{c}') for c in cams]) for name in ['train','internal_val']}
lookup={};mapping=[]
for x in selected:
 name='train' if x['timestamp']<=train_end else 'internal_val' if x['timestamp']>=val_start else None
 if name is None:continue
 iid=len(mapping)+1;cam=x['camera'];f=x['frame'];lookup[cam,f]=(name,iid)
 splits[name]['images'].append(dict(id=iid,file_name=f'images/sequence_2/cam_{cam}/image_{f}.jpg',height=1080,width=1920,frame_id=f+1,video_id=cam+1))
 mapping.append(dict(split=name,sequence='sequence_2',camera=f'cam_{cam}',source_frame=f,timestamp_seconds=x['timestamp'],image_id=iid))
skipped=collections.Counter();gt_frames=set();annid=0
for cam in cams:
 with (D/f'annotations/mot_format/sequence_2/cam_{cam}.txt').open() as stream:
  for row in csv.reader(stream):
   key=(cam,int(row[0]))
   if key not in lookup:continue
   gt_frames.add(key);name,iid=lookup[key];x,y,w,h=map(float,row[2:6]);xx=min(1920,x+w);yy=min(1080,y+h);x=max(0,x);y=max(0,y)
   if xx<=x or yy<=y:skipped['fully_outside_or_invalid']+=1;continue
   annid+=1;splits[name]['annotations'].append(dict(id=annid,image_id=iid,category_id=1,bbox=[x,y,xx-x,yy-y],area=(xx-x)*(yy-y),iscrowd=0,track_id=int(row[1])))
# Absence of any MOT record cannot prove a fully annotated empty image: exclude it.
unknown={iid for key,(name,iid) in lookup.items() if key not in gt_frames}
for split in splits.values():split['images']=[x for x in split['images'] if x['id'] not in unknown]
mapping=[x for x in mapping if x['image_id'] not in unknown]
O.mkdir(parents=True,exist_ok=True);(O/'annotations').mkdir(exist_ok=True);(O/'images').mkdir(exist_ok=True)
for name,split in splits.items():(O/'annotations'/f'{name}.json').write_text(json.dumps(split)+'\n')
with (O/'frame_mapping.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(mapping[0]));w.writeheader();w.writerows(mapping)
record=dict(status='PREPARED_NOT_TRAINED',official_source='https://github.com/cvlab-epfl/UMPN/blob/main/datasets/factory.py',sequence='sequence_2',cameras=cams,start_timestamp=t0,end_timestamp=t1,train_end=train_end,internal_val_start=val_start,temporal_guard_seconds=val_start-train_end,sampling='nearest available unique timestamp within 100ms of each integer second, per camera',counts={name:dict(images=len(x['images']),boxes=len(x['annotations'])) for name,x in splits.items()},unknown_annotation_images_excluded=len(unknown),skipped_boxes=dict(skipped),evaluation_sequence='sequence_1',evaluation_sequence_overlap=False,GT_for_training=True,GT_quality='semi-automatic/projection labels, pixel alignment and visibility need review before optimization',training_launched=False,mounts='Prepared folder -> /train; original dataset images -> /train/images read-only. MOTDataset(data_dir=/train,name="",json_file=train.json or internal_val.json).',notes=['This is a custom subset of official scouttrain, not the full official training protocol.','Internal validation is a disjoint time block; no sequence_1 frame used for checkpoint selection.','Scene and identity independence across sequences is not asserted.','Annotation confidence -1 is not used as detector confidence; all objects are person class 1.'])
(O/'preparation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
