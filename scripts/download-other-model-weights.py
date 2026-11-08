#!/usr/bin/env python3
"""Download official YACHIYO/UWIPL checkpoints; preserve independent failures."""
import fcntl
from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('weight_download', ROOT / 'scripts/download-posetrack-weights.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
AUDIT = json.loads((ROOT / 'logs/weight-link-audit-20261031.json').read_text())
SHARED = ROOT / 'weights/shared'
REPORT = ROOT / 'logs/other-model-weights-download-validation-20261031.json'


def html(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=45) as response:
        if 'text/html' not in response.headers.get('Content-Type', ''):
            raise RuntimeError('Expected metadata HTML')
        data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise RuntimeError('Metadata exceeds bound')
        return data.decode('utf-8')


class Entries(HTMLParser):
    def __init__(self):
        super().__init__()
        self.entries = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get('data-id') and a.get('data-tooltip'):
            self.entries[a['data-id']] = a['data-tooltip']


def folder(file_id):
    parser = Entries()
    parser.feed(html('https://drive.google.com/drive/folders/' + file_id))
    return parser.entries


def metadata(file_id):
    url = 'https://drive.google.com/file/d/' + file_id + '/view'
    page = html(url)
    data = json.JSONDecoder().raw_decode(page.split('itemJson: ', 1)[1])[0]
    return dict(url=url, filename=data[1], size_bytes=int(data[25][2]))


def save(report):
    report['updated_at'] = base.now()
    tmp = REPORT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(REPORT)


def fetch(item, consumers, report, relative=None):
    try:
        record = base.download(item, SHARED)
        path = SHARED / item['filename']
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError('Archive CRC failed: ' + bad)
            record['archive_crc_check'] = 'passed'
        else:
            record['archive_crc_check'] = 'not_zip_not_applicable'
        record['models'] = consumers
        record['model_relative_path'] = relative or item['filename']
        for model in consumers:
            dest = ROOT / 'weights' / model / record['model_relative_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not os.path.samefile(path, dest) and base.digest(dest) != record['sha256']:
                    raise RuntimeError('Existing model checkpoint differs: ' + str(dest))
            else:
                os.link(path, dest)
        report['files'].append(record)
    except Exception as error:
        report['failures'].append(dict(url=item['url'], filename=item.get('filename'), error=str(error)))
        print(base.now(), 'FAILED', item.get('filename'), str(error), flush=True)
    save(report)


def main():
    SHARED.mkdir(parents=True, exist_ok=True)
    with (SHARED / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {'started_at': base.now(), 'files': [], 'failures': [], 'complete': False, 'checkpoint_load_tested': False,
                  'notes': ['Official scripts/4_inference_det_syn.sh expects epoch_100_S018_storage.pth, but the official storage folder distributes epoch_10_S018_storage.pth. Preserve the distributed filename; native path and checkpoint compatibility require validation.']}
        for item in AUDIT['checks']:
            name = item['name']
            if name in ('yachiyo_uwipl_detector', 'yachiyo_uwipl_hrnet_w48', 'osnet_ms_m_c'):
                consumers = ['yachiyo', 'uwipl']
            elif name.startswith('osnet_'):
                consumers = ['uwipl']
            else:
                continue
            item = dict(item)
            if 'filename' not in item:
                item['filename'] = item['url'].rsplit('/', 1)[1]
            fetch(item, consumers, report)

        # Recheck the exact official synthetic ReID link; do not substitute another model.
        synthetic_id = '1cP-3esZSnktw64SXMHn5cDk6BX15e2-q'
        try:
            item = metadata(synthetic_id)
            fetch(item, ['uwipl'], report, 'synthetic/synthetic_reid_model_60_epoch.pth')
        except Exception as error:
            report['failures'].append({'component': 'uwipl_synthetic_reid', 'url': 'https://drive.google.com/file/d/' + synthetic_id + '/view', 'error': str(error)})
            print(base.now(), 'UNAVAILABLE synthetic ReID', str(error), flush=True)
            save(report)

        expected = {
            'yolov7_base': 'epoch_100_base.pth',
            'yolov7_hospital': 'epoch_10_S014_hospital.pth',
            'yolov7_market': 'epoch_10_S003_market.pth',
            'yolov7_office': 'epoch_10_S009_office.pth',
            'yolov7_storage': 'epoch_10_S018_storage.pth',
        }
        try:
            subfolders = folder('10LT1BlBAfYnr-fJjka_Lnzf4nH0N723-')
            for category, filename in expected.items():
                try:
                    ids = [fid for fid, label in subfolders.items() if label == category + ' Shared folder' or label == category]
                    if len(ids) != 1:
                        raise RuntimeError('Expected subfolder missing or ambiguous: ' + category)
                    entries = folder(ids[0])
                    ids = [fid for fid, label in entries.items() if label == filename or label.startswith(filename + ' ')]
                    if len(ids) != 1:
                        raise RuntimeError('Expected checkpoint missing or ambiguous: ' + filename)
                    item = metadata(ids[0])
                    if item['filename'] != filename:
                        raise RuntimeError('Metadata filename mismatch')
                    fetch(item, ['uwipl'], report, 'synthetic/' + category + '/' + filename)
                except Exception as error:
                    report['failures'].append({'component': category, 'filename': filename, 'error': str(error)})
                    print(base.now(), 'FAILED', category, str(error), flush=True)
                    save(report)
        except Exception as error:
            report['failures'].append({'component': 'uwipl_yolov7_folder', 'error': str(error)})
        report['complete'] = not report['failures']
        report['finished_at'] = base.now()
        save(report)
        for model in ('yachiyo', 'uwipl'):
            dest = ROOT / 'weights' / model
            dest.mkdir(parents=True, exist_ok=True)
            files = [r for r in report['files'] if model in r['models']]
            manifest = {'updated_at': base.now(), 'model': model, 'source_commit': AUDIT['sources'][model]['commit'], 'files': files, 'failures': report['failures'] if model == 'uwipl' else [r for r in report['failures'] if r.get('filename') in ('bytetrack_x_mot17.pth.tar', 'osnet_ms_m_c.pth.tar', 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')]}
            manifest['complete'] = not manifest['failures']
            manifest['notes'] = report['notes'] if model == 'uwipl' else []
            manifest['real_video_weights_complete'] = len([r for r in files if not r['model_relative_path'].startswith('synthetic/')]) == (3 if model == 'yachiyo' else 7)
            (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            (dest / 'SHA256SUMS').write_text(''.join(r['sha256'] + '  ' + r['model_relative_path'] + '\n' for r in files))
        print(base.now(), 'FINISHED', len(report['files']), 'files;', len(report['failures']), 'unresolved', flush=True)


if __name__ == '__main__':
    main()
