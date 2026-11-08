#!/usr/bin/env python3
"""Visualize recorded native pose rejection examples without new inference."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

R=Path(__file__).resolve().parents[1];B=R/'outputs/osc-pilot-uncapped-20261101'
A=R/'outputs/osc-analysis-20261101';I=R/'outputs/osc-inference-uncapped-20261101/input'
def data(camera,serial):
    cid=int(camera.split('_')[1])+1
    tracks=json.loads((B/('tracking/repeat_01/whole_tracking_results.json')).read_text())[str(cid)]
    row=tracks[serial];bbox=[row['Coordinate'][k] for k in ('x1','y1','x2','y2')]
    poses=json.loads((B/('inputs/Pose/scene_001/camera_%04d/camera_%04d_out_keypoint.json'%(cid,cid))).read_text())
    kp=next(p['keypoints'] for p in poses[str(row['Frame'])] if [int(v) for v in p['bbox'][:4]]==bbox)
    return row,bbox,kp
def draw(ax,camera,serial,threshold,title):
    row,b,kp=data(camera,serial)
    img=Image.open(I/camera/('frame_%06d.jpg'%(row['Frame']-1)))
    left=max(0,b[0]-30);top=max(0,b[1]-30);right=min(img.width,b[2]+30);bottom=min(img.height,b[3]+30)
    ax.imshow(img.crop((left,top,right,bottom)))
    for j,(x,y,s) in enumerate(kp):
        ax.scatter(x-left,y-top,c='#00ff00' if s>threshold else '#ff0000',s=18)
        ax.text(x-left+2,y-top,str(j),fontsize=7,color='white')
    ax.set_xlim(0,right-left);ax.set_ylim(bottom-top,0);ax.axis('off')
    ax.set_title(title+f'\nframe {row["Frame"]+276}, confident joints: {sum(p[2]>threshold for p in kp)}/17',fontsize=10)
    return kp

removed=[r for r in csv.DictReader((A/'forensics5/global_row_exclusions.csv').open()) if r['had_global_id']=='True']
fig,axs=plt.subplots(1,4,figsize=(12,4));scores=[]
for ax,r in zip(axs,removed):
    kp=draw(ax,r['camera'],r['serial'],.75,r['camera']+' / '+r['reason'])
    scores.append(dict(camera=r['camera'],serial=r['serial'],source_frame=r['source_frame'],reason=r['reason'],
                       scores=[p[2] for p in kp],count_strictly_above_075=sum(p[2]>.75 for p in kp)))
fig.suptitle('Assigned Global rows removed by native pose-noise filtering (green > 0.75)')
fig.tight_layout(rect=(0,0,1,.90));fig.savefig(A/'figures/postfilter_pose.png',dpi=150);plt.close(fig)
(A/'forensics5/postfilter_pose_scores.json').write_text(json.dumps(scores,indent=2)+'\n')

reps=json.loads((B/'tracking/repeat_01/representative_nodes_scene1.json').read_text())['6']
longest=max(reps.items(),key=lambda x:len(x[1]['all_serials']))
good=next((k,v) for k,v in reps.items() if v['representative_node']['score']==1)
fig,axs=plt.subplots(1,2,figsize=(8,5))
for ax,(lid,v) in zip(axs,[longest,good]):
    draw(ax,'cam_5',v['representative_node']['serial'],.7,
         f'Local {lid}: length {len(v["all_serials"])}, condition {v["representative_node"]["score"]}')
fig.suptitle('cam_5: long but poor pose vs good pose but only 8 observations')
fig.tight_layout();fig.savefig(A/'figures/cam5_pose_length.png',dpi=150);plt.close(fig)
print('Saved postfilter and cam_5 pose diagnostics')
