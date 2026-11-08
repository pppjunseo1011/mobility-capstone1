#!/usr/bin/env python3
"""Package completed UWIPL review artifacts and verify every archived file."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'outputs/uwipl-study-20261101';O=A/'handoff'
NAME='UWIPL_SCOUT_handoff_20261101';STEM='UWIPL_SCOUT_알고리즘검증_20261101'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    with (A/'experiment_comparison.csv').open() as f:assert len(list(csv.DictReader(f)))==32
    assert json.loads((A/'evaluation/self_test.json').read_text())['status']=='PASS'
    for ext in ['pdf','html']:
        p=R/'docs/handoff'/f'{STEM}.{ext}';assert p.stat().st_size>10000;shutil.copy2(p,O/p.name)
    files=sorted(p for p in O.rglob('*') if p.is_file() and p.name not in ['FILES.csv','SHA256SUMS.txt'])
    records=[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    with (O/'FILES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,records[0]);w.writeheader();w.writerows(records)
    (O/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in records)+sha(O/'FILES.csv')+'  FILES.csv\n')
    target=R/'deliverables'/f'{NAME}.zip';part=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(O.rglob('*')):
            if p.is_file():z.write(p,NAME+'/'+str(p.relative_to(O)))
    with zipfile.ZipFile(part) as z:
        assert z.testzip() is None
        for x in records:
            raw=z.read(NAME+'/'+x['path']);assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256']
        for name in ['FILES.csv','SHA256SUMS.txt']:assert z.read(NAME+'/'+name)==(O/name).read_bytes()
        count=len(z.namelist())
    part.replace(target);digest=sha(target)
    target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n')
    result={'status':'PASS','files':count,'bytes':target.stat().st_size,'sha256':digest,'all_member_sha256_verified':True,'zip':str(target)}
    (A/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
