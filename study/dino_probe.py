"""Execute the fixed frozen DINOv2 all-label linear probe on each benchmark."""
import json
import os
import time
from pathlib import Path
os.environ['XFORMERS_DISABLED']='1'
import numpy as np
import torch
from torch.utils.data import DataLoader
from extension_features import Images,manifest_path
from extension_heads import save_evaluation
from kernel_heads import fit_softmax,scores

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    source=json.loads((ROOT/'references'/'dinov2_source.json').read_text())
    model=torch.hub.load(str(ROOT/source['source_directory']),'dinov2_vits14',source='local',pretrained=False)
    state=torch.load(ROOT/'torch_cache'/'checkpoints'/'dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model=model.eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    for dataset in ['gtsrb','belgium']:
        rows=json.loads(manifest_path(dataset).read_text());folder=ROOT/'extension'/'dino'/dataset;folder.mkdir(parents=True,exist_ok=True)
        cache=folder/'features.npy';progress=folder/'progress.json'
        if not progress.exists() or not json.loads(progress.read_text()).get('completed'):
            x=np.lib.format.open_memmap(cache,mode='w+',dtype='float16',shape=(len(rows),384))
            loader=DataLoader(Images(rows),batch_size=32,num_workers=2,pin_memory=True,shuffle=False)
            done=0;start=time.perf_counter();last=start
            with torch.inference_mode():
                for batch in loader:
                    with torch.autocast('cuda',dtype=torch.float16):features=model(batch.cuda(non_blocking=True))
                    assert features.shape[1]==384 and torch.isfinite(features).all()
                    x[done:done+len(batch)]=features.cpu().numpy();done+=len(batch);now=time.perf_counter()
                    if now-last>25 or done==len(rows):
                        x.flush();progress.write_text(json.dumps(dict(processed=done,completed=done==len(rows),wall_seconds=now-start,
                            dimensions=384,backbone='dinov2_vits14',frozen_parameters=sum(p.numel() for p in model.parameters()),source_commit=source['commit']),indent=2))
                        print('DINO features',dataset,done,'/',len(rows),'seconds',round(now-start,1),flush=True);last=now
        x=np.load(cache).astype(np.float32);norm=np.linalg.norm(x,axis=1,keepdims=True);assert np.all(norm>0);x/=norm
        y=np.array([r['label'] for r in rows]);ix={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
        lock=folder/'locked.json'
        if not lock.exists():
            grid=[]
            for c in [.1,1.,10.]:
                clf=fit_softmax(x[ix['train']],y[ix['train']],c);s=scores(clf,x[ix['validation']])
                grid.append(dict(c=c,validation_accuracy=float(np.mean(s.argmax(1)==y[ix['validation']]))));print('DINO selection',dataset,grid[-1],flush=True)
            lock.write_text(json.dumps(dict(grid=grid,chosen=max(grid,key=lambda r:r['validation_accuracy']),test_metrics_used=False),indent=2))
        if not (folder/'all_training_metrics.json').exists():
            config=json.loads(lock.read_text())['chosen'];clf=fit_softmax(x[ix['train']],y[ix['train']],config['c'])
            start=time.perf_counter();s=scores(clf,x[ix['test']]);elapsed=time.perf_counter()-start
            save_evaluation(rows,y,ix['test'],s,dict(model=clf,training_indices=ix['train'],dimension=384),folder,'all_training',
                dict(dataset=dataset,backbone='dinov2_vits14',seed=None,budget_per_class=None,actual_head_training_count=len(ix['train']),
                    configuration=config,fit_seconds=clf['fit_seconds'],prediction_seconds=elapsed,mode='all_training'))
    print('Both DINOv2 full-label probes completed',flush=True)

if __name__=='__main__':main()
