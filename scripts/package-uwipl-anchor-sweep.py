#!/usr/bin/env python3
"""Package evaluated anchor experiments, excluding the large feature cache."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-anchor-sweep-20261102'
STEM='UWIPL_SCOUT_앵커_단일변수실험_20261102';NAME='UWIPL_SCOUT_anchor_sweep_20261102'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    with (O/'experiment_comparison.csv').open() as f:assert len(list(csv.DictReader(f)))==16
    assert json.loads((O/'cache_verification.json').read_text())['status']=='PASS'
    for ext in ['pdf','html']:
        src=R/'docs/handoff'/f'{STEM}.{ext}';assert src.stat().st_size>10000;shutil.copy2(src,O/src.name)
    shutil.copy2(Path(__file__),O/'reproducibility/scripts'/Path(__file__).name)
    exclude={'sct_current_features.npy','FILES.csv','SHA256SUMS.txt','package_verification.json'}
    files=sorted(p for p in O.rglob('*') if p.is_file() and p.name not in exclude)
    records=[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    with (O/'FILES.csv').open('w',newline='') as f:w=csv.DictWriter(f,records[0]);w.writeheader();w.writerows(records)
    (O/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in records)+sha(O/'FILES.csv')+'  FILES.csv\n')
    dest=R/'deliverables'/f'{NAME}.zip';part=dest.with_suffix('.zip.partial')
    with zipfile.ZipFile(part,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[O/'FILES.csv',O/'SHA256SUMS.txt']:z.write(p,NAME+'/'+str(p.relative_to(O)))
    with zipfile.ZipFile(part) as z:
        assert z.testzip() is None
        for r in records:assert hashlib.sha256(z.read(NAME+'/'+r['path'])).hexdigest()==r['sha256']
        for name in ['FILES.csv','SHA256SUMS.txt']:assert z.read(NAME+'/'+name)==(O/name).read_bytes()
        count=len(z.namelist())
    part.replace(dest);checksum=sha(dest);dest.with_suffix('.zip.sha256').write_text(checksum+'  '+dest.name+'\n')
    result={'status':'PASS','files':count,'bytes':dest.stat().st_size,'sha256':checksum,'all_payload_hashes_verified':True,
      'excluded_large_cache':'sct_current_features.npy retained on server','path':str(dest)}
    (O/'package_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
