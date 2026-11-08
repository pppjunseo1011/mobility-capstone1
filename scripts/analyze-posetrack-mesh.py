#!/usr/bin/env python3
"""Make posthoc comparison, lineage/formula checks and review images."""
import collections,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/posetrack-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-mesh-20261104';V=O/'visualizations'
def load(p):return json.loads(p.read_text())
def read(p):return list(csv.DictReader(p.open()))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    V.mkdir(exist_ok=True);comparison=[];cameras=[];formulas={}
    for run in ['B_final','C_final','C_30s']:
        p=O/run
        if not (p/'metrics.json').exists():continue
        metrics=load(p/'metrics.json')['summary'];comparison.append(dict(run=run,**{k:v for k,v in metrics.items() if k!='protocol'}))
        cameras.extend(dict(run=run,**r) for r in read(p/'per_camera.csv'))
        maxerr=0.;n=0
        for row in csv.DictReader((p/'affinity_pairs.csv').open()):
            a={k:float(row[k]) for k in ['Ab','Aepi','Ah','As','norm','reid_weight','a_box','a_epi','a_homo','a_reid','final_score']}
            score=(a['a_epi']*a['Aepi']+a['a_box']*a['Ab']+a['a_homo']*a['Ah']+a['As']*a['a_reid']*a['reid_weight'])/(1+a['reid_weight'])
            if a['norm']>0:score-=(a['a_box']-a['norm'])*.1
            maxerr=max(maxerr,abs(max(0,score)-a['final_score']));n+=1
        assert maxerr<1e-10;formulas[run]={'pairs':n,'formula_max_abs_error':maxerr}
    csvout(O/'experiment_comparison.csv',comparison);csvout(O/'per_camera_comparison.csv',cameras)
    save(O/'affinity_formula_verification.json',formulas)
    # A solver change and the common view-swap fix must not alter C's result here.
    a=(O/'C_mesh/global_predictions.csv').read_bytes();b=(O/'C_final/global_predictions.csv').read_bytes()
    save(O/'C_solver_equivalence.json',{'GLPK_vs_HiGHS_and_common_swap_fix_output_bitwise_equal':a==b})
    obs=load(O/'full_input/observations.json');lookup={f"cam{x['camera']}_row{x['camera_row']}":x for x in obs}
    rows=read(O/'C_final/posthoc_gt_matches.csv');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    def draw(cam,frame,selected=None,title='PREDICTED Global IDs; original boxes'):
        image=Image.open(R/f'datasets/scout/extracted/scout/images/sequence_1/cam_{cam}/image_{frame}.jpg').convert('RGB');d=ImageDraw.Draw(image)
        for r in rows:
            if r['camera']!=f'cam_{cam}' or int(r['source_frame'])!=frame:continue
            gid=int(r['global_id'])
            if selected is not None and gid!=selected:continue
            x,y,w,h=[float(r[k]) for k in ['x','y','width','height']];color=tuple(int(v) for v in np.random.default_rng(gid).integers(70,255,3))
            d.rectangle([x,y,x+w,y+h],outline=color,width=4);d.text((x,max(30,y-24)),f'PRED ID {gid}',fill=color,font=font,stroke_width=1,stroke_fill='black')
        d.rectangle([0,0,1920,34],fill='white');d.text((8,3),f'cam_{cam} frame {frame} | {title}',font=font,fill='black')
        return image
    for cam in range(8):draw(cam,302).save(V/f'cam_{cam}_frame302.jpg',quality=88)
    groups=collections.defaultdict(list)
    for r in rows:groups[(int(r['source_frame']),int(r['global_id']))].append(r)
    groups=sorted(groups.items(),key=lambda kv:-len({r['camera'] for r in kv[1]}))
    examples=[]
    for (frame,gid),rr in groups:
        if len(rr)<3:continue
        gt=[r['matched_gt_id'] for r in rr if r['matched_gt_id']]
        if len(gt)>=3 and len(set(gt))==1:
            im=Image.new('RGB',(1920,1080),'white')
            for i,r in enumerate(rr[:4]):
                c=int(r['camera'].split('_')[1]);im.paste(draw(c,frame,gid).resize((960,540)),((i%2)*960,(i//2)*540))
            im.save(V/'cross_camera_example.jpg',quality=90);examples.append({'type':'same predicted ID across cameras','frame':frame,'pred_id':gid,'posthoc_gt_ids':gt});break
    # An ID linked to different GT people over time is a suspected merge example.
    ids=collections.defaultdict(list)
    for r in rows:
        if r['matched_gt_id']:ids[int(r['global_id'])].append(r)
    candidate=sorted(ids.items(),key=lambda kv:-len({r['matched_gt_id'] for r in kv[1]}))
    if candidate:
        gid,rr=candidate[0];selected={}
        for r in rr:selected.setdefault(r['matched_gt_id'],r)
        im=Image.new('RGB',(1920,1080),'white')
        for i,r in enumerate(list(selected.values())[:4]):
            c=int(r['camera'].split('_')[1]);f=int(r['source_frame']);im.paste(draw(c,f,gid,title=f"PRED ID {gid}; posthoc matched GT {r['matched_gt_id']}").resize((960,540)),((i%2)*960,(i//2)*540))
        im.save(V/'suspected_id_merge.jpg',quality=90);examples.append({'type':'posthoc suspected ID merge; GT matching IoU>=.5','pred_id':gid,'distinct_matched_gt':list(selected)})
    pts=[r for r in obs if r['mesh_world'] is not None];xyz=np.array([r['mesh_world'] for r in pts]);p=np.array([r['z0_plane_xy'] for r in pts]);err=np.linalg.norm(p-xyz[:,:2],axis=1)
    fig,axs=plt.subplots(1,2,figsize=(10,4));z=axs[0].scatter(xyz[:,0],xyz[:,1],c=xyz[:,2],s=2);fig.colorbar(z,ax=axs[0],label='Mesh ground Z (dataset units)')
    ix=np.argsort(err)[-30:];axs[0].quiver(xyz[ix,0],xyz[ix,1],p[ix,0]-xyz[ix,0],p[ix,1]-xyz[ix,1],angles='xy',scale_units='xy',scale=1,color='red')
    axs[0].set_title('Mesh contacts -> Z=0 plane estimates');axs[0].set_xlabel('World X');axs[0].set_ylabel('World Y')
    axs[1].hist(err,bins=40);axs[1].set_xlabel('Z=0 vs mesh XY distance (dataset units)');axs[1].set_ylabel('Observations');axs[1].set_title('Geometry discrepancy, NOT GT error')
    fig.tight_layout();fig.savefig(V/'geometry_discrepancy.png',dpi=160);plt.close(fig)
    csvout(O/'largest_geometry_discrepancies.csv',[{'camera':pts[i]['camera'],'source_frame':pts[i]['source_frame'],
      'mesh_x':xyz[i,0],'mesh_y':xyz[i,1],'mesh_z':xyz[i,2],'z0_x':p[i,0],'z0_y':p[i,1],'distance':err[i]} for i in ix[::-1]])
    save(O/'visual_examples.json',examples)
    save(O/'findings.json',{'experiment_rows':comparison,'geometry_discrepancy_xy':{'median':float(np.median(err)),'p95':float(np.percentile(err,95)),'max':float(err.max())},
      'mesh_hit_rows':len(pts),'mesh_miss_rows':len(obs)-len(pts),'C_all_cameras_have_output':all(int(r['output_rows'])>0 for r in cameras if r['run']=='C_final'),
      'flat_replay':load(O/'A_replay/verification.json'),'geometry_verification':load(O/'geometry_verification.json')})
    print('ANALYZED',len(comparison),'runs',formulas)
if __name__=='__main__':main()
