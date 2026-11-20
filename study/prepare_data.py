"""Download the publisher-linked GTSRB archives and create a grouped split.

No test predictions or model-selection metrics are computed by this script.
"""
import csv
import hashlib
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
URL = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
ARCHIVES = {
    'GTSRB_Final_Training_Images.zip': None,
    'GTSRB_Final_Test_Images.zip': 'c7e4e6327067d32654124b0fe9e82185',
    'GTSRB_Final_Test_GT.zip': 'fe31e9c9270bbcd7b84b7f21a9d9d9e5',
}
EXPECTED_SHA256 = {
    'GTSRB_Final_Training_Images.zip': 'd32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c',
    'GTSRB_Final_Test_Images.zip': '48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa',
    'GTSRB_Final_Test_GT.zip': 'f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d',
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download():
    DATA.mkdir(exist_ok=True)
    metadata = []
    for name, expected in ARCHIVES.items():
        path = DATA / name
        if not path.exists() or (expected is not None and digest(path, 'md5') != expected):
            partial = path.with_suffix('.partial')
            print('Downloading', name, flush=True)
            request = urllib.request.Request(URL + name, headers={'User-Agent': 'Research reconstruction'})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as f:
                total = int(response.headers.get('Content-Length') or 0)
                done = next_report = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        print(name, done // (1024 * 1024), 'MiB of', total // (1024 * 1024), flush=True)
                        next_report = done + 64 * 1024 * 1024
            if expected is not None and digest(partial, 'md5') != expected:
                raise ValueError(f'Archive checksum mismatch: {name}')
            with zipfile.ZipFile(partial) as candidate:
                if candidate.testzip() is not None:
                    raise ValueError(f'Archive CRC check failed: {name}')
            partial.replace(path)
        measured_sha256=digest(path)
        if measured_sha256 != EXPECTED_SHA256[name]:
            raise ValueError('Archive differs from the recorded study version: '+name)
        metadata.append(dict(name=name, source=URL + name, md5=digest(path, 'md5'),
                             externally_published_md5=expected,
                             sha256=measured_sha256, bytes=path.stat().st_size))
        marker = DATA / (name + '.extracted')
        if not marker.exists():
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    target = (DATA / info.filename).resolve()
                    if not target.is_relative_to(DATA.resolve()) or '\\' in info.filename:
                        raise ValueError('Unsafe archive member path')
                    if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                        raise ValueError('Archive symbolic links not permitted')
                archive.extractall(DATA)
            marker.write_text('Checksum-verified extraction\n')
    (DATA / 'archive_checksums.json').write_text(json.dumps(metadata, indent=2))


def build_manifest():
    rows = []
    training = DATA / 'GTSRB' / 'Final_Training' / 'Images'
    for table in sorted(training.glob('*/GT-*.csv')):
        with table.open(newline='') as f:
            for row in csv.DictReader(f, delimiter=';'):
                path = table.parent / row['Filename']
                label = int(row['ClassId'])
                group = f"{label}:{row['Filename'].split('_')[0]}"
                rows.append(dict(sample_id='train/' + path.relative_to(training).as_posix(),
                    path=path.relative_to(ROOT).as_posix(), label=label, group_id=group,
                    roi=[int(row[k]) for k in ('Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')],
                    sha256='', split='train'))
    if len(rows) != 39209:
        raise ValueError(f'Unexpected training population: {len(rows)}')
    table = DATA / 'GT-final_test.csv'
    with table.open(newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            path = DATA / 'GTSRB' / 'Final_Test' / 'Images' / row['Filename']
            rows.append(dict(sample_id='test/' + row['Filename'],
                path=path.relative_to(ROOT).as_posix(), label=int(row['ClassId']),
                group_id='', roi=[int(row[k]) for k in ('Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')],
                sha256='', split='test'))
    if sum(row['split'] == 'test' for row in rows) != 12630:
        raise ValueError('Unexpected official test population')
    print('Computing image checksums', len(rows), flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = pool.map(lambda row: digest(ROOT / row['path']), rows)
        for i, (row, value) in enumerate(zip(rows, hashes), 1):
            row['sha256'] = value
            if i % 5000 == 0:
                print('Image checksums', i, '/', len(rows), flush=True)
    test_hashes={r['sha256'] for r in rows if r['split']=='test'}
    contaminated_groups={r['group_id'] for r in rows
                         if r['split']!='test' and r['sha256'] in test_hashes}
    excluded=[r for r in rows if r['split']!='test' and r['group_id'] in contaminated_groups]
    rows=[r for r in rows if r not in excluded]
    (DATA/'excluded_training_tracks.json').write_text(json.dumps(dict(
        reason='Exact file overlap with official test archive; exclude whole source track',
        groups=sorted(contaminated_groups),samples=excluded),indent=2))
    # Merge any training groups sharing byte-identical images before validation sampling.
    parents={r['group_id']:r['group_id'] for r in rows if r['split']!='test'}
    def root(group):
        while parents[group]!=group:
            parents[group]=parents[parents[group]]
            group=parents[group]
        return group
    first_hash={}
    for r in rows:
        if r['split']=='test':
            continue
        if r['sha256'] in first_hash:
            other=first_hash[r['sha256']]
            if r['label']!=other['label']:
                raise ValueError('Conflicting training labels on byte-identical images')
            a,b=root(r['group_id']),root(other['group_id'])
            parents[max(a,b)]=min(a,b)
        else:
            first_hash[r['sha256']]=r
    by_class=defaultdict(set)
    for r in rows:
        if r['split']!='test':
            by_class[r['label']].add(root(r['group_id']))
    rng=random.Random(20261012)
    validation_groups=set()
    for label in sorted(by_class):
        groups=sorted(by_class[label])
        rng.shuffle(groups)
        validation_groups.update(groups[:max(1,round(0.2*len(groups)))])
    for r in rows:
        if r['split']!='test':
            r['split']='validation' if root(r['group_id']) in validation_groups else 'train'
    hash_splits = defaultdict(set)
    for row in rows:
        hash_splits[row['sha256']].add(row['split'])
    overlap = {h: sorted(v) for h, v in hash_splits.items() if len(v) > 1}
    if overlap:
        review = {h: [r for r in rows if r['sha256']==h] for h in overlap}
        (DATA/'duplicate_review.json').write_text(json.dumps(review,indent=2))
        raise ValueError(f'Cross-split identical files: {len(overlap)}; inspect before training')
    counts = {split: dict(sorted(Counter(r['label'] for r in rows if r['split'] == split).items()))
              for split in ('train', 'validation', 'test')}
    (DATA / 'manifest.json').write_text(json.dumps(rows, indent=2))
    with (DATA / 'split_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'split', 'group_id', 'sha256'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    summary = dict(counts=counts, population={k: sum(v.values()) for k, v in counts.items()},
                   cross_split_exact_file_duplicates=len(overlap), split_seed=20261012,
                   excluded_training_groups=sorted(contaminated_groups),excluded_training_images=len(excluded),
                   duplicate_hashes_within_splits=sum(n>1 for n in Counter(r['sha256'] for r in rows).values()),
                   test_track_identifiers='Unavailable in official archive')
    (DATA / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    print('Manifest complete', summary['population'], flush=True)


if __name__ == '__main__':
    download()
    build_manifest()
