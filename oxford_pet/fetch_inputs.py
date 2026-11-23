"""Fetch and verify the public Oxford-IIIT Pet archives and pretrained weight file."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


OXFORD = ('https://thor.robots.ox.ac.uk/~vgg/data/pets/',
          'https://www.robots.ox.ac.uk/~vgg/data/pets/data/')
RESOURCES = {
    'images': ('inputs/images.tar.gz', tuple(url + 'images.tar.gz' for url in OXFORD),
               'md5', '5c4f3ee8e5d25df40f4fd59a7f44e54c'),
    'annotations': ('inputs/annotations.tar.gz', tuple(url + 'annotations.tar.gz' for url in OXFORD),
                    'md5', '95a8c909bbe2e81eed6a22bccdf3f68f'),
    'weights': ('torch_cache/checkpoints/convnext_tiny-983f1562.pth',
                ('https://download.pytorch.org/models/convnext_tiny-983f1562.pth',),
                'sha256', '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'),
}


def valid(path, algorithm, expected):
    if not path.exists():
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest() == expected


def fetch(root, name, entry):
    relative, urls, algorithm, expected = entry
    target = root / relative
    if valid(target, algorithm, expected):
        print(f'Verified existing {name} file', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.download')
    if valid(partial, algorithm, expected):
        partial.replace(target)
        return
    last_error = None
    for _ in range(3):
        for url in urls:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'Oxford-Pet-reproducibility/1.0', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            try:
                print(f'{name}: opening source at {offset:,} bytes', flush=True)
                with urlopen(Request(url, headers=headers), timeout=35) as response:
                    size = response.getheader('Content-Length')
                    range_header = response.getheader('Content-Range', '')
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', range_header)
                        if match is None or int(match[1]) != offset:
                            raise ValueError(f'Unexpected byte range {range_header}')
                        total = int(match[3])
                    else:
                        offset = 0
                        total = int(size) if size else None
                    last_print = time.monotonic()
                    with partial.open('ab' if offset else 'wb') as stream:
                        while block := response.read(256 * 1024):
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic() - last_print > 20:
                                print(f'{name}: {offset:,} / {total} bytes', flush=True)
                                last_print = time.monotonic()
                if (total is None or offset == total) and valid(partial, algorithm, expected):
                    partial.replace(target)
                    print(f'Verified {name}: {offset:,} bytes', flush=True)
                    return
                if total is not None and offset == total:
                    rejected = partial.with_name(partial.name + '.checksum-mismatch')
                    partial.replace(rejected)
                raise ValueError('Incomplete transfer or mismatched digest')
            except (OSError, ValueError) as error:
                last_error = error
                print(f'{name}: {str(error)}', flush=True)
    raise RuntimeError(f'{name} download did not verify: {last_error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('resources', nargs='+', choices=list(RESOURCES))
    options = parser.parse_args()
    with ThreadPoolExecutor(len(set(options.resources))) as pool:
        futures = [pool.submit(fetch, options.root, name, RESOURCES[name]) for name in dict.fromkeys(options.resources)]
    for future in futures:
        future.result()


if __name__ == '__main__':
    main()
