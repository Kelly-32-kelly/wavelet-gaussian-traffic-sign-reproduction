"""Fetch and pin the official DINOv2 inference source and ViT-S/14 weights."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    directory=ROOT/'references';record=directory/'dinov2_source.json'
    if record.exists():
        prior=json.loads(record.read_text());commit=prior['commit'];expected={r['path']:r['sha256'] for r in prior['files']}
    else:
        expected={}
        request=urllib.request.Request('https://api.github.com/repos/facebookresearch/dinov2/commits/main',headers={'User-Agent':'Reproducible research'})
        with urllib.request.urlopen(request,timeout=60) as response:commit=json.load(response)['sha']
    urls=[(directory/'dinov2_source.zip',f'https://codeload.github.com/facebookresearch/dinov2/zip/{commit}'),
          (ROOT/'torch_cache'/'checkpoints'/'dinov2_vits14_pretrain.pth','https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth')]
    entries=[]
    for path,url in urls:
        if not path.exists():
            print('Downloading',path.name,flush=True)
            with urllib.request.urlopen(url,timeout=90) as response,path.with_suffix('.partial').open('wb') as stream:
                for block in iter(lambda:response.read(1024*1024),b''):stream.write(block)
            path.with_suffix('.partial').replace(path)
        entries.append(dict(path=path.relative_to(ROOT).as_posix(),url=url,bytes=path.stat().st_size,sha256=digest(path)))
        if entries[-1]['path'] in expected:assert entries[-1]['sha256']==expected[entries[-1]['path']]
        print('Recorded',path.name,entries[-1]['sha256'],flush=True)
    vendor=ROOT/'extension'/'vendor';vendor.mkdir(exist_ok=True)
    with zipfile.ZipFile(directory/'dinov2_source.zip') as z:
        assert z.testzip() is None
        for entry in z.infolist():
            assert (vendor/entry.filename).resolve().is_relative_to(vendor.resolve())
            assert ((entry.external_attr>>16)&0o170000)!=0o120000
        z.extractall(vendor)
    record.write_text(json.dumps(dict(commit=commit,source_directory=f'extension/vendor/dinov2-{commit}',files=entries),indent=2))

if __name__=='__main__':main()
