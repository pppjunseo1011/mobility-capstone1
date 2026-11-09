#!/usr/bin/env python3
"""Standalone comparison figure; percentages use the exact exported CSV values."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=Path('/out');metrics=list(csv.DictReader((O/'experiment_comparison.csv').open(encoding='utf-8-sig')));det=list(csv.DictReader((O/'detection_comparison.csv').open(encoding='utf-8-sig')))
assert len(metrics)==36
variants=['baseline','D1_1088x1920','D1_1280x2304','D2_tiles2x2','D3_nms0.8'];labels=['800 (V3)','1088','1280','2x2 tiles','NMS 0.8']
fig,axes=plt.subplots(2,2,figsize=(11,7.3));x=np.arange(5)
for col,scope in enumerate(['development','holdout']):
 d={r['variant']:r for r in det if r['scope']==scope and r['seconds']=='30' and r['stage']=='selected' and r['camera']=='0' and r['gt_height']=='all'}
 m={r['variant']:r for r in metrics if r['scope']==scope and r['seconds']=='30'}
 for row,source,fields in [(0,d,[('recall','Recall'),('precision','Precision')]),(1,m,[('HOTA_2D','HOTA 2D'),('IDF1_view_aware','IDF1 view')])]:
  ax=axes[row,col]
  for j,(field,label) in enumerate(fields):
   ys=[100*float(source[v][field]) for v in variants];bars=ax.bar(x+(j-.5)*.35,ys,.35,label=label,color=['#3176ad','#dc8132'][j])
   for b,value in zip(bars,ys):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,f'{value:.1f}',ha='center',va='bottom',fontsize=7)
  ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,100 if row==0 else 65);ax.set_ylabel('Percent');ax.set_title(scope.capitalize()+' 30s: '+('detection' if row==0 else 'tracking'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=8)
fig.suptitle('UWIPL V4: one-factor detector changes, STCRA off / vote 15',fontsize=13)
fig.tight_layout(rect=(0,0,1,.96));dest=O/'report';dest.mkdir(exist_ok=True);fig.savefig(dest/'detector_tracking_comparison.png',dpi=180);fig.savefig(dest/'detector_tracking_comparison.svg');plt.close(fig)
print('Standalone PNG/SVG comparison saved')
