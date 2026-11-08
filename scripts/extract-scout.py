#!/usr/bin/env python3
"""Extract the verified SCOUT archive into a new directory; no model execution."""
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets/scout"
ARCHIVE = DATA / "archives/scout.tar.gz"
DOWNLOAD_STATUS = DATA / "archives/scout.tar.gz.status.json"
DESTINATION = DATA / "extracted"
STAGING = DATA / ".extracting-scout"
STATUS = DATA / "scout-extract.status.json"
LOCK = DATA / "scout-extract.lock"
LOG = ROOT / "logs/scout-extract.log"
RESERVE = 50 * 1024**3
STOP = False


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def process_start(pid):
    return Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()[19]


def check_stop():
    if STOP:
        raise InterruptedError("Stopped; partial extraction retained in the staging directory.")


def on_stop(signum, frame):
    global STOP
    STOP = True


def member_path(member):
    """Only regular files/directories under scout/, with no traversal or links."""
    name = member.name
    parts = PurePosixPath(name).parts
    if (not parts or parts[0] != "scout" or name.startswith("/") or
            ".." in parts or "\\" in name or "\x00" in name):
        raise ValueError("Unsafe or unexpected archive path: " + repr(name))
    if not (member.isfile() or member.isdir()) or member.issparse():
        raise ValueError("Archive links, special files and sparse files are not accepted: " + repr(name))
    if member.size < 0:
        raise ValueError("Negative archive member size")
    return Path(*parts)


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


def extract_archive(source, staging, expected_sha256, reserve_bytes=RESERVE,
                    progress=None, stop_check=check_stop):
    """One streaming pass, exclusive writes, bounded memory, no extractall().

    Failed/interrupted staging is kept for diagnosis; never silently reused.
    Completion requires the same compressed SHA-256 as the verified download.
    """
    source, staging = Path(source), Path(staging)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular, non-symlink archive")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Extraction staging already exists; refusing overwrite: " + str(staging))
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Invalid expected SHA-256")
    if reserve_bytes < 0:
        raise ValueError("Disk reserve cannot be negative")
    before = source.stat()
    staging.mkdir(mode=0o700)
    files, directories, written, declared = 0, 0, 0, 0
    groups = Counter()
    current = None
    started = time.monotonic()
    last_disk_check = 0.0
    last_progress = 0.0
    free = shutil.disk_usage(staging).free

    def emit(reader, force=False):
        nonlocal last_disk_check, last_progress, free
        stop_check()
        elapsed = time.monotonic()
        if force or elapsed-last_disk_check >= 5:
            free = shutil.disk_usage(staging).free
            last_disk_check = elapsed
            if free < reserve_bytes:
                raise RuntimeError("Free disk space is below the extraction reserve")
        if progress and (force or elapsed-last_progress >= 5):
            seconds = elapsed-started
            speed = reader.bytes_read / seconds if seconds > 0 else 0
            progress({"files_written": files, "directory_entries": directories,
                      "written_bytes": written, "declared_file_bytes_seen": declared,
                      "current_member": current, "compressed_bytes_read": reader.bytes_read,
                      "compressed_total_bytes": before.st_size,
                      "compressed_progress_percent": round(100*reader.bytes_read/before.st_size, 3),
                      "elapsed_seconds": round(seconds), "average_compressed_bytes_per_second": round(speed),
                      "eta_seconds": round((before.st_size-reader.bytes_read)/speed) if speed > 0 else None,
                      "free_bytes": free, "groups": dict(groups)})
            last_progress = elapsed

    with source.open("rb") as stream:
        reader = HashReader(stream)
        emit(reader, True)
        with tarfile.open(fileobj=reader, mode="r|gz", bufsize=1024**2) as archive:
            while True:
                stop_check()
                member = archive.next()
                if member is None:
                    break
                relative = member_path(member)
                current = str(relative)
                target = staging / relative
                # The staging tree is created by this run; never follow a link
                # if another process changes it while extraction is running.
                for parent in [target] + list(target.parents):
                    if parent == staging.parent:
                        break
                    if parent.is_symlink():
                        raise ValueError("Symlink appeared in extraction tree: " + str(parent))
                if member.isdir():
                    target.mkdir(parents=True, mode=0o700, exist_ok=True)
                    directories += 1
                else:
                    if shutil.disk_usage(staging).free < member.size + reserve_bytes:
                        raise RuntimeError("Insufficient free disk space for the next file plus reserve: " + current)
                    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                    declared += member.size
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("Missing regular file content: " + current)
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                    with handle, os.fdopen(os.open(target, flags, 0o600), "wb") as output:
                        remaining = member.size
                        while remaining:
                            stop_check()
                            chunk = handle.read(min(4*1024**2, remaining))
                            if not chunk:
                                raise ValueError("Truncated archive member: " + current)
                            output.write(chunk)
                            remaining -= len(chunk)
                            written += len(chunk)
                            emit(reader)
                    files += 1
                    groups["/".join(relative.parts[1:3])] += 1
                emit(reader)
                # Streaming archives need no retained per-file TarInfo objects.
                archive.members.clear()
        # Include trailing compressed bytes in the fingerprint as well.
        while reader.read(4*1024**2):
            emit(reader)
        actual_sha256 = reader.digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Compressed archive SHA-256 differs from the verified download")
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Archive changed during extraction")
        if not files or written != declared:
            raise ValueError("Empty or incomplete archive extraction")
        emit(reader, True)
    return {"files_written": files, "directory_entries": directories,
            "written_bytes": written, "groups": dict(groups),
            "archive_sha256": actual_sha256, "archive_sha256_matches_verified_download": True,
            "compressed_bytes_read": before.st_size, "compressed_progress_percent": 100.0,
            "elapsed_seconds": round(time.monotonic()-started)}


def run():
    os.umask(0o077)
    for path in (DATA, ARCHIVE, STAGING, DESTINATION, STATUS, LOCK, LOG):
        if path.is_symlink():
            raise ValueError("Refusing symlink: " + str(path))
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A SCOUT extraction is already running")
        if DESTINATION.exists() or STAGING.exists():
            raise ValueError("Extraction destination/staging exists; refusing overwrite")
        state = {"state": "preflight", "started_at": now(), "pid": os.getpid(),
                 "process_start_ticks": process_start(os.getpid()), "archive": str(ARCHIVE),
                 "destination": str(DESTINATION), "staging": str(STAGING),
                 "reserve_bytes": RESERVE, "model_inference_run": False,
                 "dataset_image_validation": "not_run", "automatic_model_execution": False}
        last_log = 0.0

        def report(phase, extra=None):
            nonlocal last_log
            state.update(extra or {})
            state.update(state=phase, updated_at=now())
            write_json(STATUS, state)
            if phase != "extracting" or time.monotonic()-last_log >= 60:
                keys = ("state", "compressed_progress_percent", "files_written", "written_bytes", "current_member", "eta_seconds", "free_bytes", "error")
                print(now(), json.dumps({k: state[k] for k in keys if k in state}, ensure_ascii=False), flush=True)
                last_log = time.monotonic()

        signal.signal(signal.SIGTERM, on_stop)
        signal.signal(signal.SIGINT, on_stop)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        os.nice(10)
        report("preflight")
        try:
            download = read_json(DOWNLOAD_STATUS)
            if download.get("state") != "complete" or download.get("gzip_integrity") != "passed":
                raise ValueError("Download/gzip integrity validation is not complete")
            expected = download.get("sha256", "")
            checksum = ARCHIVE.with_name("scout.tar.gz.sha256").read_text().split()
            if not checksum or checksum[0] != expected:
                raise ValueError("Download status and checksum sidecar disagree")
            if ARCHIVE.stat().st_size != download["total_bytes"]:
                raise ValueError("Archive size differs from completed download")
            state.update(archive_bytes=ARCHIVE.stat().st_size, expected_sha256=expected,
                         initial_free_bytes=shutil.disk_usage(DATA).free)
            result = extract_archive(ARCHIVE, STAGING, expected,
                                     progress=lambda data: report("extracting", data))
            check_stop()
            for name in ("calibrations", "timestamps", "images", "annotations"):
                if not (STAGING / "scout" / name).is_dir():
                    raise ValueError("Missing expected SCOUT directory: " + name)
            if DESTINATION.exists() or DESTINATION.is_symlink():
                raise ValueError("Destination appeared during extraction; refusing overwrite")
            STAGING.rename(DESTINATION)
            report("complete", dict(result, completed_at=now(), dataset_root=str(DESTINATION / "scout"),
                                    current_member=None, eta_seconds=0,
                                    free_bytes=shutil.disk_usage(DATA).free))
            return 0
        except BaseException as error:
            report("interrupted" if STOP or isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                   {"error": str(error), "partial_files_preserved": STAGING.exists()})
            return 130 if STOP else 1


def start():
    os.umask(0o077)
    if not DATA.is_dir() or DESTINATION.exists() or STAGING.exists():
        raise ValueError("Data directory missing or extraction destination/staging already exists")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "run"]
    if shutil.which("ionice"):
        command = ["ionice", "-c", "2", "-n", "7"] + command
    with LOG.open("a") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(ROOT), start_new_session=True)
    for unused in range(30):
        state = read_json(STATUS)
        if state.get("pid") == process.pid and state.get("state") in ("extracting", "complete"):
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        if process.poll() is not None:
            raise RuntimeError("Extraction exited; inspect " + str(LOG))
        time.sleep(1)
    raise RuntimeError("Extraction startup confirmation timed out; inspect status/log before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status"))
    action = parser.parse_args().action
    if action == "status":
        print(json.dumps(read_json(STATUS) or {"state": "not_started"}, ensure_ascii=False, indent=2))
        return 0
    return start() if action == "start" else run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("SCOUT extraction error: " + str(error), file=sys.stderr)
        sys.exit(1)
