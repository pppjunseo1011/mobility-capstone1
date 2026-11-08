#!/usr/bin/env python3
"""Download the three audited PoseTrack inference checkpoints (stdlib only)."""
import fcntl
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPCookieProcessor

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'weights' / 'posetrack'
AUDIT = ROOT / 'logs' / 'weight-link-audit-20261031.json'


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action')
            self.active = True
        if self.active and tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def download(item, destination=None):
    name = item['filename']
    assert Path(name).name == name
    directory = Path(destination) if destination is not None else DEST
    target = directory / name
    partial = directory / (name + '.part')
    expected = item['size_bytes']
    if target.exists():
        if target.stat().st_size != expected:
            raise RuntimeError('Existing file size mismatch; refusing overwrite: ' + name)
        print(now(), 'EXISTS', name, flush=True)
    else:
        opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        is_drive = urlparse(item['url']).hostname == 'drive.google.com'
        file_id = item['url'].split('/d/')[1].split('/')[0] if is_drive else None
        initial = ('https://drive.google.com/uc?' + urlencode({'export': 'download', 'id': file_id})) if is_drive else item['url']
        for attempt in range(1, 4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > expected:
                    raise RuntimeError('Partial file larger than expected')
                if offset == expected:
                    break
                url = initial
                for _ in range(4):
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    if offset:
                        headers['Range'] = 'bytes=%d-' % offset
                    response = opener.open(Request(url, headers=headers), timeout=60)
                    if 'text/html' not in response.headers.get('Content-Type', '').lower():
                        break
                    with response:
                        page = response.read(1024 * 1024).decode('utf-8', errors='replace')
                    form = DownloadForm()
                    form.feed(page)
                    if not form.action or urlparse(form.action).scheme != 'https' or urlparse(form.action).hostname not in ('drive.google.com', 'drive.usercontent.google.com'):
                        raise RuntimeError('Drive did not provide a public download confirmation form (quota/access may apply)')
                    if form.fields.get('id') != file_id:
                        raise RuntimeError('Unexpected file ID in confirmation form')
                    url = form.action + '?' + urlencode(form.fields)
                else:
                    raise RuntimeError('Too many HTML confirmation responses')
                with response:
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if not match or int(match[1]) != offset or int(match[3]) != expected:
                            raise RuntimeError('Invalid range response')
                    elif response.status == 200:
                        offset = 0  # Server ignored Range: restart partial, never append.
                    else:
                        raise RuntimeError('Unexpected HTTP status %s' % response.status)
                    length = response.headers.get('Content-Length')
                    if length and int(length) != expected - offset:
                        raise RuntimeError('Unexpected response size: ' + length)
                    if 'text/' in response.headers.get('Content-Type', '').lower():
                        raise RuntimeError('Refusing text response as checkpoint')
                    print(now(), 'DOWNLOAD', name, 'start_bytes', offset, 'expected_bytes', expected, flush=True)
                    start, last, received = time.monotonic(), time.monotonic(), offset
                    with partial.open('ab' if offset else 'wb') as dest:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if received + len(block) > expected:
                                raise RuntimeError('Response exceeds expected size')
                            dest.write(block)
                            received += len(block)
                            if time.monotonic() - last >= 10:
                                elapsed = time.monotonic() - start
                                rate = (received - offset) / elapsed
                                print(now(), 'PROGRESS', name, '%d/%d %.1f%% %.2f MB/s ETA %.0fs' % (received, expected, received * 100 / expected, rate / 1e6, (expected - received) / rate), flush=True)
                                last = time.monotonic()
                        dest.flush()
                        os.fsync(dest.fileno())
                if partial.stat().st_size != expected:
                    raise RuntimeError('Incomplete download; partial preserved')
                break
            except Exception as error:
                print(now(), 'ATTEMPT_FAILED', name, attempt, str(error), flush=True)
                if attempt == 3:
                    raise
                time.sleep(2)
        partial.rename(target)
    checksum = digest(target)
    print(now(), 'VERIFIED_SIZE_SHA256', name, expected, checksum, flush=True)
    return {'filename': name, 'size_bytes': expected, 'sha256': checksum, 'source_url': item['url'], 'verified_at': now(), 'publisher_checksum_verified': False, 'checkpoint_load_tested': False}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = json.loads(AUDIT.read_text())
        items = [item for item in audit['checks'] if item['name'] in ('posetrack_detector', 'posetrack_pose', 'posetrack_reid')]
        assert len(items) == 3
        records = []
        for item in items:
            records.append(download(item))
            manifest = {'updated_at': now(), 'model': 'posetrack', 'source_commit': audit['sources']['posetrack']['commit'], 'complete': len(records) == 3, 'files': records}
            temp = DEST / 'manifest.json.tmp'
            temp.write_text(json.dumps(manifest, indent=2) + '\n')
            temp.replace(DEST / 'manifest.json')
        (DEST / 'SHA256SUMS').write_text(''.join(row['sha256'] + '  ' + row['filename'] + '\n' for row in records))
        print(now(), 'COMPLETE', sum(row['size_bytes'] for row in records), 'bytes', flush=True)


if __name__ == '__main__':
    main()
