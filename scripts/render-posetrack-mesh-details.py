#!/usr/bin/env python3
"""Zoom actual predicted boxes for readable report examples, posthoc only."""
import csv,json,collections
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-mesh-20261104';V=O/'visualizations'
rows=list(csv.DictReader((O/'C_final/posthoc_gt_matches.csv').open()));examples=json.loads((O/'visual_examples.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
def panel(r):
 cam=r['camera'];frame=r['source_frame'];gid=r['global_id'];im=Image.open(R/f'datasets/scout/extracted/scout/images/sequence_1/{cam}/image_{frame}.jpg').convert('RGB')
 x,y,w,h=[float(r[k]) for k in ['x','y','width','height']];d=ImageDraw.Draw(im);d.rectangle([x,y,x+w,y+h],outline='#ff4040',width=3)
 size=max(150,h*2.2,w*3);cx=x+w/2;cy=y+h/2
 left=max(0,min(im.width-size*1.6,cx-size*.8));top=max(0,min(im.height-size,cy-size*.5))
 crop=im.crop((int(left),int(top),int(left+size*1.6),int(top+size))).resize((900,510))
 out=Image.new('RGB',(900,570),'white');out.paste(crop,(0,60));d=ImageDraw.Draw(out)
 d.text((12,3),f"{cam} frame {frame} | PRED ID {gid}",font=font,fill='black')
 d.text((12,32),f"Posthoc IoU match: GT {r['matched_gt_id']} (not tracking input)",font=font,fill='#4b5563')
 return out
for example in examples:
 gid=str(example['pred_id']);rr=[r for r in rows if r['global_id']==gid]
 if example['type'].startswith('same'):
  rr=[r for r in rr if int(r['source_frame'])==example['frame']][:4];name='cross_camera_detail.jpg'
 else:
  chosen={}
  for r in rr:
   if r['matched_gt_id']:chosen.setdefault(r['matched_gt_id'],r)
  rr=list(chosen.values())[:4];name='suspected_id_merge_detail.jpg'
 canvas=Image.new('RGB',(1800,1140),'#f2f4f6')
 for i,r in enumerate(rr):canvas.paste(panel(r),((i%2)*900,(i//2)*570))
 canvas.save(V/name,quality=92)
print('Detailed actual-box visuals saved')
