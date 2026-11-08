#!/usr/bin/env python3
"""Timestamp-only holdout selector; reproduces the recorded 2026-11-07 search."""
import argparse,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=R/'datasets/scout/extracted/scout');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 assert not args.out.exists(),'Preserve existing frozen selection'
 D=args.data;T=json.loads((D/'timestamps/timestamps_sequence_1.json').read_text());keys=[json.loads((D/f'calibrations/cam_{c}.json').read_text())['view_id'].rsplit('_',1)[0] for c in range(8)]
 fs=sorted(map(int,T));a=np.array([[T[str(f)][k] for k in keys] for f in fs]);oldmax=max(T[str(f)][k] for f in range(277,577) for k in keys);best=None
 for idx in range(607,len(fs)-350,50):
  origin=float(np.median(a[idx]))
  if origin<=oldmax+.1:continue
  counts=[];mapping=[]
  for c in range(8):
   rows={};seen=set()
   for pos in range(max(577,idx-100),min(len(fs),idx+400)):
    ts=a[pos,c];slot=int(np.floor((ts-origin)/.1+.5));err=abs(ts-origin-slot*.1)
    if ts in seen:continue
    seen.add(ts)
    if 0<=slot<300 and err<=.025000001:
     r=dict(camera=c,source_frame=fs[pos],index=slot,timestamp_seconds=float(ts),offset_seconds=float(ts-origin-slot*.1))
     if slot not in rows or abs(r['offset_seconds'])<abs(rows[slot]['offset_seconds']):rows[slot]=r
   counts.append(len(rows));mapping.extend(rows.values())
  if best is None or len(mapping)>len(best['mapping']):best=dict(origin=origin,counts=counts,mapping=mapping,nominal=idx)
  if len(mapping)>=2380 and min(counts)>=290:break
 assert best
 result=dict(best,policy='first candidate at source index607 then +50 with >=2380/2400 slots and each camera>=290; otherwise greatest coverage; unique timestamp, nearest 100ms grid within25ms; timestamp-only selection, origins after max development timestamp',GT_used=False)
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print('selected',best['nominal'],best['counts'])
if __name__=='__main__':main()
