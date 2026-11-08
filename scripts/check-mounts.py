#!/usr/bin/env python3
"""Exercise the mount contract with tiny temporary fixtures, not SCOUT data."""
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
code = """
import errno, os
from pathlib import Path
assert Path('/data/scout/probe.txt').read_text() == 'fixture'
for directory in ('/data/scout', '/weights'):
    try:
        Path(directory, 'must-not-write.txt').write_text('invalid')
    except OSError as error:
        assert error.errno == errno.EROFS, repr(error)
    else:
        raise AssertionError('Input mount is writable: ' + directory)
Path('/outputs/persisted.txt').write_text('output-ok')
print('PASS: data and weights EROFS; output write; uid=' + str(os.getuid()))
"""
with tempfile.TemporaryDirectory(prefix="scout-mount-check-") as temp:
    base = Path(temp)
    for name in ("data", "weights"):
        (base / name).mkdir()
    (base / "data/probe.txt").write_text("fixture")
    command = ["bash", str(root / "scripts/run.sh"), "yachiyo", "tracking", "--cpu",
               "--data", str(base / "data"), "--weights", str(base / "weights"),
               "--output", str(base / "results"), "--", "python", "-c", code]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    (root / "logs/mount-validation.txt").write_text(result.stdout + "\nexit_code=" + str(result.returncode) + "\n")
    if result.returncode:
        sys.exit(result.returncode)
    assert (base / "results/persisted.txt").read_text() == "output-ok"
    assert (base / "results/persisted.txt").stat().st_uid == __import__("os").getuid()
    print("PASS: result persisted on host with invoking user's UID")
