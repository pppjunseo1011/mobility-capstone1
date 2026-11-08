#!/usr/bin/env python3
"""Package verified review artifacts, with complete per-file SHA validation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/posetrack-transfer-20261101/handoff'
NAME='PoseTrack_SCOUT_handoff_20261101'
STEM='PoseTrack_SCOUT_실행검증_인계서_20261101'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((OUT/'02_adapter/audit_summary.json').read_text())['module_alignment']=='PASS'
    video=json.loads((OUT/'03_intermediate/visualizations/pose_videos/verification.json').read_text())
    assert len(video)==8 and all(v['status']=='PASS' and v['decoded_frames']==50 for v in video)
    for ext in ['pdf','html']:
        src=ROOT/'docs/handoff'/f'{STEM}.{ext}';assert src.stat().st_size>10000
        shutil.copy2(src,OUT/src.name)
    reference=ROOT.parent/'PoseTrack_SCOUT_Transfer_v3_fixed.pdf'
    (OUT/'00_reference').mkdir(exist_ok=True)
    shutil.copy2(reference,OUT/'00_reference'/reference.name)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt'])
    records=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    with (OUT/'FILES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['path','bytes','sha256']);w.writeheader();w.writerows(records)
    (OUT/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in records)+sha(OUT/'FILES.csv')+'  FILES.csv\n')
    files=sorted(p for p in OUT.rglob('*') if p.is_file())
    dest=ROOT/'deliverables'/f'{NAME}.zip';dest.parent.mkdir(exist_ok=True)
    partial=dest.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,NAME+'/'+str(p.relative_to(OUT)))
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        for r in records:
            raw=z.read(NAME+'/'+r['path'])
            assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],r['path']
        for p in [OUT/'FILES.csv',OUT/'SHA256SUMS.txt']:
            assert z.read(NAME+'/'+p.name)==p.read_bytes()
    partial.replace(dest)
    checksum=sha(dest)
    dest.with_suffix('.zip.sha256').write_text(checksum+'  '+dest.name+'\n')
    report={'status':'PASS','zip':str(dest),'bytes':dest.stat().st_size,'files':len(files),
        'sha256':checksum,'crc_and_all_member_sha256_verified':True}
    (ROOT/'outputs/posetrack-transfer-20261101/package_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
