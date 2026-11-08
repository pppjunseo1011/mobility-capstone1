#!/usr/bin/env python3
"""Actual input/GT/prediction evidence panels, and timestamp diagnostic plots."""
import collections,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/posetrack-time-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/posetrack-mesh-20261104';O=R/'outputs/posetrack-time-study-20261106';V=O/'visualizations';D=R/'datasets/scout/extracted/scout'
def read(p):return list(csv.DictReader(p.open()))
def main():
 V.mkdir(exist_ok=True);pred=read(OLD/'C_30s/posthoc_gt_matches.csv');obs=json.loads((OLD/'full_input30/observations.json').read_text());ob={f"cam{r['camera']}_row{r['camera_row']}":r for r in obs};ng={r['observation_id']:r for r in read(O/'association/neural_observation_gt.csv')};gt=json.loads((R/'outputs/osc-analysis-20261101/s30_th40/gt_selected.json').read_text());gtmap={(r['camera_id'],r['source_frame'],str(r['track_id'])):r['bbox_xywh'] for r in gt}
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25);manifest=[]
 def panel(oid,gid,caption):
  r=ob[oid];c=r['camera'];sf=r['source_frame'];im=Image.open(D/f'images/sequence_1/cam_{c}/image_{sf}.jpg').convert('RGB');draw=ImageDraw.Draw(im);b=r['raw_bbox'];draw.rectangle(b[:4],outline='#ff2020',width=3)
  match=ng[oid]['matched_gt_id'];gb=gtmap.get((f'cam_{c}',sf,match))
  if gb:draw.rectangle([gb[0],gb[1],gb[0]+gb[2],gb[1]+gb[3]],outline='#00b040',width=2)
  size=max(160,(b[3]-b[1])*2.8);left=max(0,min(im.width-size*1.6,(b[0]+b[2])/2-size*.8));top=max(0,min(im.height-size,(b[1]+b[3])/2-size*.5));crop=im.crop((int(left),int(top),int(left+size*1.6),int(top+size))).resize((900,500))
  out=Image.new('RGB',(900,590),'white');out.paste(crop,(0,90));dr=ImageDraw.Draw(out)
  for i,line in enumerate([f'cam_{c} / source frame {sf} / PRED ID {gid}',f'GT ID {match or "UNMATCHED"} (green); predicted/detected box (red)',caption]):dr.text((10,2+28*i),line,font=font,fill='black')
  return out
 def board(name,items):
  canvas=Image.new('RGB',(1800,590*((len(items)+1)//2)),'#f4f5f6')
  for i,(oid,gid,caption) in enumerate(items):canvas.paste(panel(oid,gid,caption),((i%2)*900,(i//2)*590))
  canvas.save(V/name,quality=92);manifest.append({'file':name,'items':items,'GT_usage':'posthoc visualization only; prediction IDs are model outputs'})
 ids=json.loads((O/'association/global_id_analysis.json').read_text())
 normal=[r for r in pred if r['global_id']=='28' and r['matched_gt_id']=='500' and int(r['model_frame'])<17];groups=collections.defaultdict(list)
 for r in normal:groups[r['model_frame']].append(r)
 rr=max(groups.values(),key=len)[:4];board('id28_cross_camera.jpg',[(r['observation_id'],28,'Consistent cross-camera GT association') for r in rr])
 for tid in [28,38]:
  bad=ids[str(tid)]['first_inconsistent_observation'];good=[r for r in pred if int(r['global_id'])==tid and r['matched_gt_id']==ids[str(tid)]['initial_GT_consensus'] and int(r['model_frame'])<int(bad['model_frame'])]
  samecam=[r for r in good if r['camera']==bad['camera']];prior=(samecam or good)[-1]
  board(f'id{tid}_first_conflict.jpg',[(prior['observation_id'],tid,'Earlier GT-consistent output'),(bad['observation_id'],tid,'First posthoc GT conflict (not necessarily first real error)')])
 recovery=json.loads((O/'association/missing_reactivation_detail.json').read_text());wrong=[r for r in recovery if r['status']=='incorrect']
 for i,e in enumerate(wrong):
  before=[r for r in pred if int(r['global_id'])==e['track_id'] and r['matched_gt_id']==e['previous_GT'] and int(r['model_frame'])<=e['last_GT_supported_frame']][-1]
  board(f'missing_reactivation_conflict_{i}.jpg',[(before['observation_id'],e['track_id'],'Last GT-supported prior identity'),(e['new_observation_ids'][0],e['track_id'],f"Recovered by cosine {e['similarity']:.3f} > 0.5; initially unconfirmed")])
 unresolved=next(r for r in recovery if r['status'].startswith('unresolved'));board('missing_reactivation_unresolved.jpg',[(unresolved['previous_observations'][0],unresolved['track_id'],'Before Missing; no reliable GT overlap'),(unresolved['new_observation_ids'][0],unresolved['track_id'],'After recovery; cannot score identity from available GT')])
 geo=read(O/'geometry/mesh_intersections_all.csv');byoid={r['observation_id']:r for r in pred}
 failures=[r for r in geo if r['bbox_status']!='hit'];good=[r for r in geo if r['bbox_status']=='hit']
 key=lambda r:(ob[r['observation_id']]['raw_bbox'][2]-ob[r['observation_id']]['raw_bbox'][0])*(ob[r['observation_id']]['raw_bbox'][3]-ob[r['observation_id']]['raw_bbox'][1])
 fail=max(failures,key=key);hit=max(good,key=key)
 board('mesh_hit_vs_miss.jpg',[(r['observation_id'],byoid.get(r['observation_id'],{}).get('global_id','NO OUTPUT'),label) for r,label in [(hit,'BBox ray intersects mesh'),(fail,'BBox ray enters bounds but hits no mesh triangle')]])
 timeline=read(O/'timestamp_before_after.csv');x=[int(r['nominal_source_frame']) for r in timeline];a=[float(r['before_skew_seconds']) for r in timeline];b=[float(r['after_skew_seconds']) for r in timeline]
 fig,ax=plt.subplots(figsize=(10,3.5));ax.plot(x,np.array(a)*1000,label='Before: same frame index');ax.plot(x,np.array(b)*1000,label='After: unique timestamp observations');ax.axhline(50,color='red',ls='--',label='50 ms');ax.set(xlabel='Nominal source frame',ylabel='Cross-camera skew (ms)');ax.legend();fig.tight_layout();fig.savefig(V/'timestamp_before_after.png',dpi=160);plt.close(fig)
 # Explicit availability map: gaps must not disappear from the review.
 mapping=read(O/'frame_mapping.csv');availability=np.zeros((8,300))
 for r in mapping:availability[int(r['camera']),int(r['common_time_index'])]=r['status']=='selected'
 fig,ax=plt.subplots(figsize=(10,3));ax.imshow(availability,aspect='auto',cmap='Greys',vmin=0,vmax=1);ax.set(xlabel='Common time index',ylabel='Camera',title='Black: observed; white: missing (no synthetic frame)');fig.tight_layout();fig.savefig(V/'camera_availability.png',dpi=150);plt.close(fig)
 (V/'visualization_manifest.json').write_text(json.dumps({'examples':manifest,'correct_reactivation_example':'not generated: none of the 14 original events could be verified as correct with available GT'},indent=2)+'\n');print('VISUALS',len(manifest))
if __name__=='__main__':main()
