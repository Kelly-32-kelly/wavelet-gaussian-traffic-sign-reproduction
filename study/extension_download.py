"""Fetch the official BelgiumTSC archives and TorchVision ConvNeXt weights."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    tasks = [
        (ROOT/'data'/'belgium'/'BelgiumTSC_Training.zip',
         'https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip'),
        (ROOT/'data'/'belgium'/'BelgiumTSC_Testing.zip',
         'https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip'),
        (ROOT/'torch_cache'/'checkpoints'/'convnext_tiny-983f1562.pth',
         'https://download.pytorch.org/models/convnext_tiny-983f1562.pth')]
    records=[]
    for path, url in tasks:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            print('Downloading',path.name,flush=True)
            with urllib.request.urlopen(url,timeout=90) as response, partial.open('wb') as stream:
                done=0
                for block in iter(lambda: response.read(1024*1024), b''):
                    stream.write(block); done+=len(block)
                    if done%(32*1024*1024)<1024*1024:
                        print(path.name,done//(1024*1024),'MiB',flush=True)
            partial.replace(path)
        value=sha256(path)
        if path.suffix=='.pth' and not value.startswith('983f1562'):
            raise ValueError('Official weight filename hash does not match')
        record=dict(path=path.relative_to(ROOT).as_posix(),url=url,
                    sha256=value,bytes=path.stat().st_size)
        records.append(record)
        print('Verified',path.name,path.stat().st_size,value,flush=True)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt archive')
                for info in archive.infolist():
                    target=(path.parent/info.filename).resolve()
                    if not target.is_relative_to(path.parent.resolve()):
                        raise ValueError('Unsafe archive path')
                    if ((info.external_attr>>16)&0o170000)==0o120000:
                        raise ValueError('Archive symlink')
                archive.extractall(path.parent)
                print('Archive members',len(archive.namelist()),archive.namelist()[:15],flush=True)
    (ROOT/'data'/'belgium'/'archive_checksums.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':
    main()
