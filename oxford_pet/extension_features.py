"""Extract frozen backbone features for the registered extension."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.models import vgg16,VGG16_Weights,convnext_tiny,ConvNeXt_Tiny_Weights
from run_features import TRANSFORM

ROOT=Path(__file__).resolve().parent

def manifest_path(dataset, root=ROOT):
    dataset_folder = {'gtsrb': '', 'belgium': 'belgium', 'oxford_pet': 'oxford_pet'}[dataset]
    return root/'data'/dataset_folder/'manifest.json'

class Images(Dataset):
    def __init__(self,rows,root=ROOT):self.rows=rows;self.root=root
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['path']) as im:
            im=im.convert('RGB');x1,y1,x2,y2=r.get('model_roi',r['roi'])
            if not (0<=x1<=x2<im.width and 0<=y1<=y2<im.height):
                raise ValueError('Invalid ROI '+r['sample_id'])
            return TRANSFORM(im.crop((x1,y1,x2+1,y2+1)))

def main(dataset,backbone,root=ROOT):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.hub.set_dir(str(root/'torch_cache'))
    source=manifest_path(dataset,root);rows=json.loads(source.read_text());fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
    folder=root/'extension'/'cache'/f'{dataset}_{backbone}';folder.mkdir(parents=True,exist_ok=True)
    out=folder/'features.npy';progress=folder/'progress.json';dim=25088 if backbone=='vgg16' else 768
    offset=0
    if progress.exists():
        record=json.loads(progress.read_text());assert record['manifest_sha256']==fingerprint
        if record['completed']:print('Complete cache',dataset,backbone,flush=True);return
        offset=record['processed']
    feature=np.load(out,mmap_mode='r+') if out.exists() else np.lib.format.open_memmap(out,mode='w+',dtype='float16',shape=(len(rows),dim))
    if backbone=='vgg16':
        model=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().cuda()
        extractor=lambda batch:model(batch).flatten(1)
    else:
        model=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval().cuda()
        extractor=lambda batch:model.classifier[1](model.classifier[0](model.avgpool(model.features(batch))))
        torch.save(model.state_dict(),folder/'frozen_backbone.pt')
    for p in model.parameters():p.requires_grad_(False)
    loader=DataLoader(Images(rows[offset:],root),batch_size=24,num_workers=2,pin_memory=True,shuffle=False)
    start=time.perf_counter();last=start;done=offset;forward_seconds=0
    with torch.inference_mode():
        for batch in loader:
            batch=batch.cuda(non_blocking=True);torch.cuda.synchronize();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):values=extractor(batch)
            torch.cuda.synchronize();forward_seconds+=time.perf_counter()-t
            assert torch.isfinite(values).all()
            feature[done:done+len(batch)]=values.cpu().numpy();done+=len(batch)
            now=time.perf_counter()
            if now-last>25 or done==len(rows):
                feature.flush()
                progress.write_text(json.dumps(dict(dataset=dataset,backbone=backbone,processed=done,
                    completed=done==len(rows),manifest_sha256=fingerprint,dimension=dim,
                    segment_wall_seconds=now-start,segment_forward_seconds=forward_seconds,
                    segment_start_index=offset,forward_ms_per_image=1000*forward_seconds/(done-offset),
                    backbone_parameters=sum(p.numel() for p in model.parameters()),
                    preprocessing='shared ROI and direct 224x224 resize; not the default center-crop weight transform'),indent=2))
                print('Features',dataset,backbone,done,'/',len(rows),'seconds',round(now-start,1),flush=True);last=now

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['gtsrb','belgium']);p.add_argument('backbone',choices=['vgg16','convnext_tiny'])
    a=p.parse_args();main(a.dataset,a.backbone)
