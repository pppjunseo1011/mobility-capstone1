#!/usr/bin/env python3
"""Download the official SCOUT archive; standard library + system curl only."""
import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = ROOT / "datasets/scout/archives"
ARCHIVE = ARCHIVES / "scout.tar.gz"
PART = ARCHIVES / "scout.tar.gz.part"
META = ARCHIVES / "scout.tar.gz.meta.json"
STATUS = ARCHIVES / "scout.tar.gz.status.json"
LOCK = ARCHIVES / "scout.tar.gz.lock"
LOG = ROOT / "logs/scout-download.log"
URL = "https://scout.epfl.ch/datas/datasets/scout/1.0/scout.tar.gz"
RESERVE = 50 * 1024**3
STOP = False
CHILD = None
DOWNLOAD_DIR = None


def configure_download_dir(value):
    """Keep archive, resume metadata, lock, status and log together when overridden."""
    global DOWNLOAD_DIR, ARCHIVES, ARCHIVE, PART, META, STATUS, LOCK, LOG
    if value is None:
        return
    DOWNLOAD_DIR = Path(os.path.abspath(os.path.expanduser(value)))
    ARCHIVES = DOWNLOAD_DIR
    ARCHIVE = ARCHIVES / "scout.tar.gz"
    PART = ARCHIVES / "scout.tar.gz.part"
    META = ARCHIVES / "scout.tar.gz.meta.json"
    STATUS = ARCHIVES / "scout.tar.gz.status.json"
    LOCK = ARCHIVES / "scout.tar.gz.lock"
    LOG = ARCHIVES / "scout-download.log"


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def file_size(path):
    return path.stat().st_size if path.exists() else 0


def private_directories():
    os.umask(0o077)
    directories = (ARCHIVES,) if DOWNLOAD_DIR is not None else (ROOT / "datasets", ROOT / "datasets/scout", ARCHIVES)
    for directory in directories:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink():
            raise RuntimeError("Refusing a symlink dataset directory: " + str(directory))
    LOG.parent.mkdir(exist_ok=True)
    for path in (ARCHIVE, PART, META, STATUS, LOCK, LOG):
        if path.is_symlink():
            raise RuntimeError("Refusing a symlink: " + str(path))


def process_start(pid):
    return Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()[19]


def interrupted(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()


def run(limit_rate):
    global CHILD
    private_directories()
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SCOUT downloader holds the lock.")
        state = {
            "state": "preflight", "url": URL, "archive": str(ARCHIVE),
            "pid": os.getpid(), "process_start_ticks": process_start(os.getpid()),
            "started_at": now(), "limit_rate": limit_rate,
            "bandwidth_limit_enabled": limit_rate != "0",
            "registration": "User states registration and terms acceptance are complete; not independently verified.",
            "automatic_extraction": False,
        }

        def report(phase, **extra):
            state.update(extra)
            state.update(state=phase, updated_at=now())
            size = file_size(ARCHIVE if ARCHIVE.exists() else PART)
            state["downloaded_bytes"] = size
            if state.get("total_bytes"):
                state["download_percent"] = round(100 * size / state["total_bytes"], 3)
            write_json(STATUS, state)

        def check_stop():
            if STOP:
                raise InterruptedError("Stopped by request; partial file preserved.")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        os.nice(10)
        report("preflight")
        try:
            request = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "SCOUT-academic-download/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200 or not response.geturl().startswith("https://scout.epfl.ch/"):
                    raise RuntimeError("Unexpected official download response or redirect.")
                remote = {
                    "url": URL, "total_bytes": int(response.headers["Content-Length"]),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "accept_ranges": response.headers.get("Accept-Ranges"),
                }
            if remote["total_bytes"] <= 0 or not remote["etag"]:
                raise RuntimeError("Missing valid remote size or ETag.")
            previous = read_json(META)
            if PART.exists() or ARCHIVE.exists():
                if any(previous.get(key) != remote.get(key) for key in ("url", "total_bytes", "etag")):
                    raise RuntimeError("Existing archive metadata differs or is absent; refusing overwrite/resume.")
            if PART.exists() and ARCHIVE.exists():
                raise RuntimeError("Both final and partial archive exist; manual inspection required.")
            if file_size(PART) > remote["total_bytes"]:
                raise RuntimeError("Partial archive exceeds expected length.")
            available = shutil.disk_usage(ARCHIVES).free
            needed = 0 if ARCHIVE.exists() else remote["total_bytes"] - file_size(PART)
            if available < needed + RESERVE:
                raise RuntimeError("Insufficient free space for archive plus 50 GiB reserve.")
            write_json(META, dict(remote, checked_at=now(), free_bytes_at_preflight=available))
            state.update(remote)
            print(now(), "Preflight passed:", remote, "free_bytes:", available, flush=True)
            check_stop()

            if not ARCHIVE.exists() and file_size(PART) < remote["total_bytes"]:
                for attempt in range(1, 7):
                    check_stop()
                    command = [
                        "curl", "--fail", "--silent", "--show-error", "--location",
                        "--proto", "=https", "--proto-redir", "=https",
                        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                        "--continue-at", "-",
                        "--header", "If-Match: " + remote["etag"], "--output", str(PART), URL,
                    ]
                    if limit_rate != "0":
                        command[1:1] = ["--limit-rate", limit_rate]
                    start_size, start_time = file_size(PART), time.monotonic()
                    # Retain the lock in curl as well if this supervisor is unexpectedly killed.
                    CHILD = subprocess.Popen(command, stdin=subprocess.DEVNULL, pass_fds=(lock.fileno(),))
                    report("downloading", attempt=attempt, curl_pid=CHILD.pid,
                           curl_process_start_ticks=process_start(CHILD.pid))
                    last_report = 0
                    last_log = 0
                    while CHILD.poll() is None:
                        check_stop()
                        if shutil.disk_usage(ARCHIVES).free < RESERVE:
                            raise RuntimeError("Free disk space fell below 50 GiB reserve; stopping.")
                        if time.monotonic() - last_report >= 5:
                            size = file_size(PART)
                            speed = (size - start_size) / max(1, time.monotonic() - start_time)
                            report("downloading", average_bytes_per_second=round(speed),
                                   eta_seconds=round((remote["total_bytes"] - size) / speed) if speed > 0 else None)
                            last_report = time.monotonic()
                            if time.monotonic() - last_log >= 60:
                                print(now(), "PROGRESS", json.dumps({key: state.get(key) for key in (
                                    "download_percent", "downloaded_bytes", "total_bytes",
                                    "average_bytes_per_second", "eta_seconds")}), flush=True)
                                last_log = time.monotonic()
                        time.sleep(1)
                    returncode = CHILD.wait()
                    CHILD = None
                    state.pop("curl_pid", None)
                    state.pop("curl_process_start_ticks", None)
                    check_stop()
                    size = file_size(PART)
                    print(now(), "Download attempt", attempt, "exit", returncode, "bytes", size, flush=True)
                    if returncode == 0 and size == remote["total_bytes"]:
                        break
                    if returncode in (22, 23, 33, 60) or size >= remote["total_bytes"] or attempt == 6:
                        raise RuntimeError("Download failed: curl exit {}, bytes {}.".format(returncode, size))
                    report("retry_wait", curl_exit_code=returncode)
                    for unused in range(min(60, 10 * attempt)):
                        check_stop()
                        time.sleep(1)

            target = ARCHIVE if ARCHIVE.exists() else PART
            if file_size(target) != remote["total_bytes"]:
                raise RuntimeError("Archive size does not match official Content-Length.")
            report("verifying", verified_compressed_bytes=0)
            digest = hashlib.sha256()

            class HashReader:
                def __init__(self, stream):
                    self.stream = stream

                def read(self, count=-1):
                    block = self.stream.read(count)
                    digest.update(block)
                    return block

            last_report = time.monotonic()
            with target.open("rb") as raw:
                with gzip.GzipFile(fileobj=HashReader(raw), mode="rb") as compressed:
                    while compressed.read(8 * 1024**2):
                        check_stop()
                        if time.monotonic() - last_report >= 5:
                            report("verifying", verified_compressed_bytes=raw.tell())
                            last_report = time.monotonic()
            check_stop()
            if target == PART:
                if ARCHIVE.exists():
                    raise RuntimeError("Final archive appeared unexpectedly; refusing overwrite.")
                PART.rename(ARCHIVE)
            checksum = digest.hexdigest()
            ARCHIVE.with_name("scout.tar.gz.sha256").write_text(checksum + "  scout.tar.gz\n")
            report("complete", completed_at=now(), gzip_integrity="passed", sha256=checksum,
                   verified_compressed_bytes=remote["total_bytes"],
                   checksum_note="Locally computed fingerprint; not compared with a publisher checksum.")
            print(now(), "COMPLETE: size and gzip CRC verified; SHA256", checksum, flush=True)
            return 0
        except BaseException as error:
            if CHILD is not None and CHILD.poll() is None:
                CHILD.terminate()
                try:
                    CHILD.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    CHILD.kill()
                    CHILD.wait()
            report("interrupted" if STOP or isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                   error=str(error))
            print(now(), state["state"].upper(), str(error), flush=True)
            return 130 if STOP else 1


def start(limit_rate):
    private_directories()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--limit-rate", limit_rate]
    if DOWNLOAD_DIR is not None:
        command += ["--download-dir", str(DOWNLOAD_DIR)]
    with LOG.open("a") as output:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for unused in range(30):
        if process.poll() is not None:
            raise RuntimeError("Downloader exited; inspect " + str(LOG))
        state = read_json(STATUS)
        if state.get("pid") == process.pid:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            print("Background log:", LOG)
            return 0
        time.sleep(0.1)
    raise RuntimeError("Downloader did not publish startup status; inspect " + str(LOG))


def stop():
    state = read_json(STATUS)
    pid = state.get("pid")
    if not pid or state.get("state") in ("complete", "interrupted", "failed"):
        print("No active SCOUT download reported.")
        return 0
    try:
        command = Path("/proc/{}/cmdline".format(pid)).read_bytes().split(b"\0")
        working_directory = Path("/proc/{}/cwd".format(pid)).resolve()
        script_matches = any(
            argument and (working_directory / os.fsdecode(argument)).resolve() == Path(__file__).resolve()
            for argument in command[1:]
        )
        if (process_start(pid) != state.get("process_start_ticks") or
                not script_matches or b"run" not in command):
            raise RuntimeError("PID identity does not match this downloader; refusing signal.")
        os.kill(pid, signal.SIGTERM)
        print("Stop requested for SCOUT downloader PID", pid, "; partial file will be preserved.")
    except FileNotFoundError:
        child_pid = state.get("curl_pid")
        if child_pid:
            try:
                command = Path("/proc/{}/cmdline".format(child_pid)).read_bytes().split(b"\0")
                if (process_start(child_pid) != state.get("curl_process_start_ticks") or
                        Path(os.fsdecode(command[0])).name != "curl" or
                        str(PART).encode() not in command or URL.encode() not in command):
                    raise RuntimeError("Child PID identity mismatch; refusing signal.")
                os.kill(child_pid, signal.SIGTERM)
                print("Stop requested for orphaned SCOUT curl PID", child_pid)
                for unused in range(50):
                    with LOCK.open("a+") as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            time.sleep(0.1)
                            continue
                        latest = read_json(STATUS)
                        if latest.get("pid") == pid and latest.get("curl_pid") == child_pid:
                            latest.update(state="interrupted", updated_at=now(), downloaded_bytes=file_size(PART),
                                          error="Orphaned transfer stopped by request; partial file preserved.")
                            if latest.get("total_bytes"):
                                latest["download_percent"] = round(100 * file_size(PART) / latest["total_bytes"], 3)
                            write_json(STATUS, latest)
                        print("Orphaned transfer lock released; stop confirmed.")
                        return 0
                print("Stop signal sent; lock release not yet confirmed. Check status and file growth.")
                return 0
            except FileNotFoundError:
                pass
        print("Recorded downloader and child are no longer running; partial file preserved.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "status", "stop"))
    parser.add_argument("--limit-rate", default="0", help="curl bandwidth cap; 0 means unlimited (default)")
    parser.add_argument("--download-dir", help="Directory for archive, resume metadata, status, lock and log; use the same path for start/run/status/stop")
    args = parser.parse_args()
    configure_download_dir(args.download_dir)
    if args.action == "status":
        print(json.dumps(read_json(STATUS) or {"state": "not_started"}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "start":
        return start(args.limit_rate)
    if args.action == "stop":
        return stop()
    return run(args.limit_rate)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
