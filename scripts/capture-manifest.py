#!/usr/bin/env python3
"""Record exact local image IDs and environment file hashes without running models."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
sources = json.loads((root / "sources.lock.json").read_text())
environments = json.loads((root / "environments.json").read_text())
report = {
    "recorded_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
    "sources": sources,
    "images": [],
    "files_sha256": {},
}
missing = []
for model, stages in environments.items():
    for stage in stages:
        tag = "scout-mtmc/{}-{}:{}".format(model, stage, sources["repositories"][model]["commit"][:7])
        result = subprocess.run(["docker", "image", "inspect", tag], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            report["images"].append({"tag": tag, "available": False, "error": result.stderr.strip()})
            missing.append(tag)
            continue
        item = json.loads(result.stdout)[0]
        report["images"].append({"tag": tag, "available": True, "id": item["Id"],
                                 "created": item["Created"], "size_bytes": item["Size"],
                                 "architecture": item["Architecture"],
                                 "workdir": item["Config"].get("WorkingDir"),
                                 "labels": item["Config"].get("Labels")})
for base in (root / "scripts", root / "docker"):
    for path in sorted(base.rglob("*")):
        if path.is_file() and "vendor" not in path.parts and "__pycache__" not in path.parts:
            report["files_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
dest = root / "logs" / "environment-manifest.json"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(str(dest))
if missing:
    raise SystemExit("Missing images: " + ", ".join(missing))
print("All {} expected images are present.".format(len(report["images"])))
