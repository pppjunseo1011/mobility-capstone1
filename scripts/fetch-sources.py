#!/usr/bin/env python3
"""Fetch exactly the recorded upstream commits; never overwrite an existing checkout."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
sources = json.loads((root / "sources.lock.json").read_text())["repositories"]
(root / "repos").mkdir(exist_ok=True)
for name, spec in sources.items():
    path = root / "repos" / name
    if path.exists():
        current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if current != spec["commit"]:
            raise SystemExit("Refusing to change existing checkout: " + str(path))
        print(name + ": verified " + current)
        continue
    subprocess.run(["git", "init", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", spec["url"]], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["commit"]], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True)
