#!/usr/bin/env python3
"""Package existing OSC review artifacts; never run inference or change results."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'deliverables'
NAME = 'OSC_SCOUT_handoff_uncapped_20261101'
files = set()

def add(path):
    path = ROOT / path
    if path.is_symlink():
        raise ValueError('Unexpected symlink: ' + str(path))
    if path.is_dir():
        for item in path.iterdir():
            if item.name not in ('__pycache__', '.git'):
                add(item.relative_to(ROOT))
    elif path.is_file():
        files.add(path)
    else:
        raise FileNotFoundError(path)

add('outputs/osc-pilot-uncapped-20261101')
for path in ('docs/handoff/OSC_SCOUT_인계서_제한해제_20261101.pdf',
             'docs/handoff/OSC_SCOUT_인계서_제한해제_20261101.html',
             'docs/osc-checklist.md',
             'sources.lock.json'):
    add(path)
for run in ('osc-inference-uncapped-20261101',):
    base = Path('outputs') / run / 'yachiyo'
    add(base / 'detection/detections.json')
    for stage in ('detection', 'pose', 'reid'):
        for name in ('report.json', 'run.log'):
            add(base / stage / name)

DEST.mkdir(exist_ok=True)
archive = DEST / (NAME + '.zip')
partial = archive.with_suffix('.zip.partial')
rows = []
checks = []
with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(files):
        rel = 'scout-mtmc/' + path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        z.writestr(NAME + '/' + rel, data)
        rows.append((rel, len(data), sha))
        checks.append((sha, rel))
    listing = io.StringIO(newline='')
    writer = csv.writer(listing)
    writer.writerow(('path', 'bytes', 'sha256'))
    writer.writerows(rows)
    for rel, data in (
        ('START_HERE.txt', (ROOT/'docs/handoff/START_HERE.txt').read_bytes()),
        ('FILES.csv', listing.getvalue().encode('utf-8')),
    ):
        z.writestr(NAME + '/' + rel, data)
        checks.append((hashlib.sha256(data).hexdigest(), rel))
    z.writestr(NAME + '/SHA256SUMS', ''.join(f'{sha}  {rel}\n' for sha, rel in checks))

# Check every archived payload against its source hash before publishing the ZIP.
with zipfile.ZipFile(partial) as z:
    assert len(z.namelist()) == len(set(z.namelist())) == len(checks) + 1
    for sha, rel in checks:
        assert hashlib.sha256(z.read(NAME + '/' + rel)).hexdigest() == sha, rel
partial.replace(archive)
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(f'{sha}  {archive.name}\n')
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'files': len(checks) + 1, 'payload_bytes': sum(r[1] for r in rows),
                  'sha256': sha, 'all_archived_hashes_verified': True}, indent=2))
