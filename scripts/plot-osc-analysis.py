#!/usr/bin/env python3
"""Plot stored experiment tables; no model inference or host dependency installation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];A=R/'outputs/osc-analysis-20261101'
P=A/'figures';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
summary=json.loads((A/'forensics5/summary.json').read_text())
bins=['1','2','3-9','10-19','20-29','30-39','40+']
cams=list(summary['camera_summary']);fig,ax=plt.subplots(figsize=(10,3.4))
bottom=np.zeros(8)
for b,color in zip(bins,['#183d55','#24748e','#55a0ab','#90c4bc','#cbd8ae','#f0be77','#cf733d']):
    values=np.array([summary['camera_summary'][c]['length_bins'].get(b,0) for c in cams])
    ax.bar(cams,values,bottom=bottom,label=b,color=color);bottom+=values
ax.set_ylabel('Local tracklets');ax.set_title('5-second baseline: observation-count distribution')
ax.legend(title='Length (frames)',ncol=7,loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
fig.tight_layout();fig.savefig(P/'tracklet_lengths.png');plt.close(fig)

rows=list(csv.DictReader((A/'experiment_comparison.csv').open()))
small=sorted([r for r in rows if r['duration_nominal_seconds']=='5'],key=lambda r:int(r['short_track_th']))
fig,axs=plt.subplots(1,2,figsize=(10,3.4))
x=[int(r['short_track_th']) for r in small]
axs[0].plot(x,[float(r['global_idf1_multiview'])*100 for r in small],'-o',color='#24748e')
axs[0].set_ylabel('Multi-view Global IDF1 (%)');axs[0].set_ylim(0,20)
axs[1].bar(x,[int(r['final_global_rows']) for r in small],width=6,color='#24748e')
axs[1].set_ylabel('Final Global observations');axs[1].set_ylim(0,850)
for ax in axs:ax.set_xlabel('short_track_th (frames)');ax.set_xticks(x);ax.grid(axis='y',alpha=.2)
fig.suptitle('5-second controlled sweep: all other settings fixed');fig.tight_layout()
fig.savefig(P/'threshold_sweep.png');plt.close(fig)

skew=list(csv.DictReader((A/'timestamp_skew.csv').open()))
fig,ax=plt.subplots(figsize=(10,3.0))
ax.plot([int(r['source_frame']) for r in skew],[float(r['skew_seconds']) for r in skew],color='#cf733d')
ax.axhline(.05,linestyle='--',color='#24748e',label='0.05 s export tolerance')
ax.set_xlabel('SCOUT source frame');ax.set_ylabel('Max camera skew (s)');ax.legend()
ax.set_title('30-second nominal clip: same-frame timestamp disagreement')
fig.tight_layout();fig.savefig(P/'timestamp_skew.png');plt.close(fig)

pairs=list(csv.DictReader((A/'forensics5/adjacent_gt_appearance.csv').open()))
fig,ax=plt.subplots(figsize=(10,3.0))
for changed,col in [('False','#24748e'),('True','#cf733d')]:
    values=[float(r['cosine']) for r in pairs if r['raw_id_changed']==changed]
    ax.hist(values,bins=np.linspace(0,1,41),alpha=.65,color=col,label='Raw ID changed' if changed=='True' else 'Raw ID maintained')
ax.axvline(.9,color='black',linestyle='--',label='Nominal SCPT gate 0.9')
ax.set_xlabel('OSNet cosine similarity');ax.set_ylabel('Adjacent-frame matched pairs');ax.legend()
ax.set_title('IoU-associated same-GT pairs (diagnostic, not causal proof)')
fig.tight_layout();fig.savefig(P/'appearance_gate.png');plt.close(fig)
print(P)
