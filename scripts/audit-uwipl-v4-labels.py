#!/usr/bin/env python3
"""Read-only checks of published annotations and a sequence-disjoint D4 proposal."""
import csv,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'datasets/scout/extracted/scout';O=R/'outputs/uwipl-v4-20261108';out=O/'label_audit';out.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'adapters'))
from scout_adapter.annotations import _array_items
# Compare representative image_277 labels directly against the distributed raw projections.
frame=next(f for f in _array_items(D/'annotations/raw/annotations_sequence_1_raw.json','frames',True) if f['frame_id']==277)
checks=[]
timestamps=json.loads((D/'timestamps/timestamps_sequence_1.json').read_text())
for cam in [0,1,3,4,5]:
 rows=[]
 with (D/f'annotations/mot_format/sequence_1/cam_{cam}.txt').open() as stream:
  for r in csv.reader(stream):
   f=int(r[0])
   if f>277:break
   if f==277:rows.append(r)
 mot={int(r[1]):list(map(float,r[2:6])) for r in rows};raw={}
 for a in frame['annotations']:
  p=a['projections_2d'].get(f'cam_{cam}')
  if p:
   (x,y),(xx,yy)=p;raw[a['track_id']]=[x,y,xx-x,yy-y]
 cal=json.loads((D/f'calibrations/cam_{cam}.json').read_text());timestamp_key=cal['view_id'].split('_')[0]
 image_sha=hashlib.sha256((D/f'images/sequence_1/cam_{cam}/image_277.jpg').read_bytes()).hexdigest();cache_sha=hashlib.sha256((R/f'outputs/uwipl-study-20261101/inference30/input/cam_{cam}/frame_000000.jpg').read_bytes()).hexdigest()
 assert image_sha==cache_sha
 checks.append(dict(camera=f'cam_{cam}',source_frame=277,image_sha256=image_sha,cache_image_equals_source=True,annotation_timestamp_ms=frame['timestamp_ms'],image_timestamp_seconds=timestamps['277'][timestamp_key],image_minus_annotation_ms=1000*timestamps['277'][timestamp_key]-frame['timestamp_ms'],mot_rows=len(mot),raw_rows=len(raw),same_ids=set(mot)==set(raw),max_box_difference=max((abs(a-b) for k in set(mot)&set(raw) for a,b in zip(mot[k],raw[k])),default=None)))
assert all(c['same_ids'] and c['max_box_difference']<=.02 for c in checks)
(out/'annotation_format_consistency.json').write_text(json.dumps(dict(status='PASS',checks=checks,interpretation='MOT BBoxes match distributed raw projections_2d; this does NOT certify pixel alignment, visibility, or timing accuracy. Visual examples include displaced boxes; no GT shifting/filtering applied.'),indent=2)+'\n')
# Official UMPN scouttrain subset restricted to sequence_2 avoids any sequence_1 timestamp overlap.
cams=[0,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24];inventory=[];samples=[]
for c in cams:
 p=D/f'images/sequence_2/cam_{c}';frames=sorted(int(f.stem[6:]) for f in p.glob('image_*.jpg'))
 available=set(frames);gtframes=set()
 with (D/f'annotations/mot_format/sequence_2/cam_{c}.txt').open() as stream:
  for row in csv.reader(stream):gtframes.add(int(row[0]))
 chosen=[f for f in frames if f%10==0 and f in gtframes]
 inventory.append(dict(sequence='sequence_2',camera=f'cam_{c}',images=len(frames),gt_frames=len(gtframes),sampled_1fps_frames=len(chosen),missing_image_gt_frames=len(gtframes-available)))
 samples.extend(dict(sequence='sequence_2',camera=f'cam_{c}',source_frame=f,image=f'images/sequence_2/cam_{c}/image_{f}.jpg') for f in chosen)
proposal=dict(status='CONDITIONAL_NOT_TRAINED',official_source='https://github.com/cvlab-epfl/UMPN/blob/main/datasets/factory.py',official_scouttrain={'sequence_2':cams,'sequence_1':list(range(8,25))},proposed_subset={'sequence_2':cams},reason='Use only sequence_2 from official training pool to avoid sequence_1 cross-camera/time leakage; sequence_1 development and holdout remain untouched.',sample_rate='Every tenth source frame, candidate inventory only; timestamps/duplicates require verification before training.',candidate_images=len(samples),inventory=inventory,development_or_holdout_sequence1_images_in_train=sum(x['sequence']=='sequence_1' for x in samples),weights_modified=False,training_launched=False,remaining=['Review D1-D3 detector and full metrics first','Check training annotation pixel alignment and visibility with representative overlays','Freeze independent sequence_2 train/internal-validation time blocks and timestamp guard gap','Export COCO/MOTDataset annotations with one person class and image-size checks','Freeze epoch budget, seed, learning rate and checkpoint selection on internal validation only','Run D4 with chosen detector resolution; do not select checkpoints using V4 holdout'])
(O/'D4_feasibility.json').write_text(json.dumps(proposal,indent=2)+'\n')
with (O/'D4_candidate_images.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(samples[0]));w.writeheader();w.writerows(samples)
print(json.dumps(dict(label_format_checks=checks,training_candidates=len(samples),training_launched=False),indent=2))
