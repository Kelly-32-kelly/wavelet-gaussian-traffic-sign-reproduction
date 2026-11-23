"""Audit Oxford-IIIT Pet official archives and make the locked development split."""
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import struct
import tarfile

import numpy as np
from PIL import Image, UnidentifiedImageError


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.tmp')
    pending.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def official_annotations(archive, split):
    name = f'annotations/{split}.txt'
    with archive.extractfile(name) as stream:
        raw = stream.read()
    records = []
    for line in raw.decode('ascii').splitlines():
        identifier, label, species, _ = line.split()
        if not re.fullmatch(r'[A-Za-z0-9_-]+_\d+', identifier):
            raise ValueError(f'Invalid sample ID {identifier}')
        if not 1 <= int(label) <= 37 or species not in {'1', '2'}:
            raise ValueError(f'Invalid class annotation for {identifier}')
        records.append(dict(sample_id=identifier, label=int(label) - 1, species=int(species), official_split=split))
    return raw, records


def extract_selected(source, directory, identifiers):
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {f'images/{identifier}.jpg': f'{identifier}.jpg' for identifier in identifiers}
    pixel_records, unreadable = {}, {}
    with tarfile.open(source, 'r|gz') as archive:
        for member in archive:
            name = member.name.lstrip('./')
            filename = wanted.pop(name, None)
            if filename is None:
                continue
            if not member.isfile():
                raise ValueError(f'Non-file image archive entry {member.name}')
            identifier = Path(filename).stem
            with archive.extractfile(member) as stream:
                image_bytes = stream.read()
            try:
                with Image.open(BytesIO(image_bytes)) as source_image:
                    decoded = source_image.convert('RGB')
                    width, height = decoded.size
                    fingerprint = hashlib.sha256(struct.pack('>II', width, height))
                    fingerprint.update(decoded.tobytes())
            except (UnidentifiedImageError, OSError, ValueError) as error:
                unreadable[identifier] = f'unreadable:{type(error).__name__}'
                continue
            destination = directory / filename
            destination.write_bytes(image_bytes)
            pixel_records[identifier] = dict(path=destination, roi=[0, 0, width - 1, height - 1],
                                             file_sha256=hashlib.sha256(image_bytes).hexdigest(),
                                             rgb_sha256=fingerprint.hexdigest())
    unreadable.update({Path(filename).stem: 'missing_file' for filename in wanted.values()})
    return pixel_records, unreadable


def prepare(root, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    folder = root / 'data' / 'oxford_pet'
    inputs = root / 'inputs'
    archives = {}
    for name in ('annotations', 'images'):
        path = inputs / f'{name}.tar.gz'
        actual = digest(path, 'md5')
        if actual != protocol['data'][f'{name}_md5']:
            raise ValueError(f'Archive checksum mismatch: {path.name}, observed {actual}')
        archives[name] = {'path': path.name, 'size': path.stat().st_size, 'md5': actual, 'sha256': digest(path)}
    folder.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    with tarfile.open(inputs / 'annotations.tar.gz', 'r:gz') as archive:
        for split in ('trainval', 'test'):
            raw, rows_by_split[split] = official_annotations(archive, split)
            (folder / f'official_{split}.txt').write_bytes(raw)

    original_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    mappings = defaultdict(set)
    known_ids = {}
    exclusions = []
    for split, rows in rows_by_split.items():
        unique = {}
        for row in rows:
            sample = row['sample_id']
            folded = sample.casefold()
            if folded in known_ids and sample != known_ids[folded]:
                raise ValueError(f'Case-insensitive file collision for {sample}')
            known_ids[folded] = sample
            mappings[row['label']].add((sample.rsplit('_', 1)[0].casefold(), row['species']))
            if sample in unique:
                if row != unique[sample]:
                    raise ValueError(f'Conflicting annotations for {sample} in {split}')
                exclusions.append({**row, 'reason': 'repeated_annotation_row'})
            unique[sample] = row
        rows_by_split[split] = list(unique.values())
    if set(mappings) != set(range(37)) or any(len(names) != 1 for names in mappings.values()):
        raise ValueError('Inconsistent category and species mapping')

    image_metadata, unreadable = extract_selected(inputs / 'images.tar.gz', folder / 'images', known_ids.values())
    groups_by_split = {}
    for split, rows in rows_by_split.items():
        by_pixels = defaultdict(list)
        for row in rows:
            sample = row['sample_id']
            if sample in unreadable:
                exclusions.append({**row, 'reason': unreadable[sample]})
                continue
            info = image_metadata[sample]
            enriched = {**row, **info, 'path': info['path'].relative_to(root).as_posix()}
            by_pixels[enriched['rgb_sha256']].append(enriched)
        groups_by_split[split] = by_pixels

    test_groups = groups_by_split['test']
    test_ids = {row['sample_id'] for row in rows_by_split['test']}
    development = defaultdict(list)
    for pixels, rows in groups_by_split['trainval'].items():
        eligible = []
        for row in sorted(rows, key=lambda entry: entry['sample_id']):
            if row['sample_id'] in test_ids or pixels in test_groups:
                exclusions.append({**row, 'reason': 'identity_shared_with_test'})
            else:
                eligible.append(row)
        if len({row['label'] for row in eligible}) > 1:
            exclusions.extend({**row, 'reason': 'conflicting_development_duplicate'} for row in eligible)
        elif eligible:
            development[eligible[0]['label']].append(eligible[0])
            exclusions.extend({**row, 'reason': 'same_label_development_duplicate'} for row in eligible[1:])

    rng = np.random.default_rng(20261123)
    output = []
    for label in range(37):
        rows = sorted(development[label], key=lambda row: row['sample_id'])
        if len(rows) < 2:
            raise ValueError(f'Insufficient audited examples for class {label}')
        order = rng.permutation(len(rows))
        validation_indices = set(order[:max(1, min(round(.2 * len(rows)), len(rows) - 1))])
        output.extend({**row, 'split': 'validation' if index in validation_indices else 'train'}
                      for index, row in enumerate(rows))
    output.extend({**row, 'split': 'test'} for group in test_groups.values() for row in group)
    output.sort(key=lambda row: (('train', 'validation', 'test').index(row['split']), row['sample_id']))
    assert len({row['sample_id'] for row in output}) == len(output)
    fingerprints = {split: {row['rgb_sha256'] for row in output if row['split'] == split}
                    for split in ('train', 'validation', 'test')}
    assert sum(map(len, fingerprints.values())) == len(set.union(*fingerprints.values()))
    per_class = {split: Counter(row['label'] for row in output if row['split'] == split) for split in fingerprints}
    assert all(set(counts) == set(range(37)) for counts in per_class.values())
    manifest = folder / 'manifest.json'
    write_json(manifest, output)
    audit = {
        'archives': archives, 'official_counts': original_counts, 'manifest_sha256': digest(manifest),
        'protocol_sha256': digest(protocol_path), 'retained_counts': {split: sum(counts.values()) for split, counts in per_class.items()},
        'per_class': per_class, 'exclusions': exclusions,
        'test_exact_pixel_duplicate_groups': [[row['sample_id'] for row in group]
                                             for group in test_groups.values() if len(group) > 1],
        'category_map': {label: {'breed': next(iter(value))[0], 'species': next(iter(value))[1]}
                         for label, value in mappings.items()},
    }
    write_json(folder / 'audit.json', audit)
    print('Oxford-IIIT Pet retained', audit['retained_counts'], 'exclusions', len(exclusions),
          'test pixel duplicate groups', len(audit['test_exact_pixel_duplicate_groups']), flush=True)
