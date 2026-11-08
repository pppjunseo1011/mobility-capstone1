#!/usr/bin/env python3
"""Convert existing scout_adapter canonical prediction CSV to aligned evaluation CSV."""
import argparse,csv,json,math
from pathlib import Path
FIELDS=['sequence_id','camera_id','source_frame_id','timestamp_seconds','common_time_index','common_timestamp_seconds','x','y','width','height','global_id','local_id','detection_confidence','model_name','experiment_configuration_id']
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--frame-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--config',required=True);a=p.parse_args()
 mapping={(f"cam_{r['camera']}",int(r['source_frame'])):r for r in csv.DictReader(a.frame_map.open()) if r['status']=='selected'}
 rows=[];excluded=0;seen=set()
 for r in csv.DictReader(a.input.open()):
  if r['kind']!='prediction' or r['image_space']!='raw' or r['id_scope']!='sequence':raise ValueError('Requires raw-image prediction with sequence-global IDs; local IDs cannot be treated as global')
  if r['sequence']!='sequence_1':raise ValueError('Wrong sequence')
  cam=r['camera'];src=int(r['frame']);m=mapping.get((cam,src))
  if m is None:excluded+=1;continue
  bbox=[float(r[k]) for k in ['x','y','width','height']]
  if not all(math.isfinite(x) for x in bbox) or min(bbox[2:])<=0:raise ValueError('Invalid bbox')
  if r['timestamp_seconds'] and abs(float(r['timestamp_seconds'])-float(m['timestamp_seconds']))>1e-8:raise ValueError('Timestamp does not match original capture')
  ident=int(r['native_track_id']);key=(cam,src,ident)
  if key in seen:raise ValueError('Duplicate camera/source-frame/global-ID row')
  seen.add(key);rows.append(dict(zip(FIELDS,['sequence_1',cam,src,m['timestamp_seconds'],m['common_time_index'],m['common_timestamp_seconds'],*bbox,ident,'',r['score'],a.model,a.config])))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',newline='') as f:w=csv.DictWriter(f,FIELDS);w.writeheader();w.writerows(rows)
 a.output.with_suffix('.conversion.json').write_text(json.dumps({'input':str(a.input),'output_rows':len(rows),'excluded_outside_shared_map':excluded,'bbox_space':'raw','source_frame_and_timestamp_from_explicit_shared_map':True,'GT_used':False},indent=2)+'\n')
 print('Normalized',len(rows),'rows; excluded',excluded)
if __name__=='__main__':main()
