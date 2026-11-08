#!/usr/bin/env python3
"""Package local-plane review artifacts and verify CRC and every payload hash."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-flat-20261101'
STEM='PoseTrack_SCOUT_평면ROI_내부계산검증';NAME='PoseTrack_SCOUT_flat_ROI_trace'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert json.loads((O/'verification.json').read_text())['status']=='PASS'
    for ext in ['pdf','html']:
        p=R/'docs/handoff'/f'{STEM}.{ext}';assert p.stat().st_size>10000;shutil.copy2(p,O/p.name)
    shutil.copy2(Path(__file__),O/'reproducibility/scripts'/Path(__file__).name)
    files=sorted(p for p in O.rglob('*') if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt','package_verification.json'])
    records=[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    with (O/'FILES.csv').open('w',newline='') as f:w=csv.DictWriter(f,records[0]);w.writeheader();w.writerows(records)
    (O/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in records)+sha(O/'FILES.csv')+'  FILES.csv\n')
    path=R/'deliverables'/f'{NAME}.zip';part=path.with_suffix('.zip.partial')
    with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[O/'FILES.csv',O/'SHA256SUMS.txt']:z.write(p,NAME+'/'+str(p.relative_to(O)))
    with zipfile.ZipFile(part) as z:
        assert z.testzip() is None
        for r in records:assert hashlib.sha256(z.read(NAME+'/'+r['path'])).hexdigest()==r['sha256']
        for name in ['FILES.csv','SHA256SUMS.txt']:assert z.read(NAME+'/'+name)==(O/name).read_bytes()
        count=len(z.namelist())
    part.replace(path);checksum=sha(path);path.with_suffix('.zip.sha256').write_text(checksum+'  '+path.name+'\n')
    report={'status':'PASS','files':count,'bytes':path.stat().st_size,'sha256':checksum,'every_payload_hash_verified':True,'path':str(path)}
    (O/'package_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
