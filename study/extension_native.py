"""Uncompressed ConvNeXt linear probes with explicit target-label accounting."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from kernel_heads import fit_softmax,scores
from extension_heads import sample,save_evaluation
from extension_features import manifest_path

ROOT=Path(__file__).resolve().parent

def main(dataset):
    rows=json.loads(manifest_path(dataset).read_text());x=np.load(ROOT/'extension'/'cache'/f'{dataset}_convnext_tiny'/'features.npy').astype(np.float32)
    norm=np.linalg.norm(x,axis=1,keepdims=True);assert np.all(norm>0);x=x/norm
    y=np.array([r['label'] for r in rows]);ix={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    folder=ROOT/'extension'/'native'/dataset;folder.mkdir(parents=True,exist_ok=True)
    for mode in ['balanced','all_training']:
        lock=folder/(mode+'_locked.json')
        if not lock.exists():
            selected=sample(ix['train'],y,42,100 if dataset=='gtsrb' else 25) if mode=='balanced' else ix['train']
            grid=[]
            for c in [.1,1.,10.]:
                model=fit_softmax(x[selected],y[selected],c);val=scores(model,x[ix['validation']]).argmax(1)
                grid.append(dict(c=c,validation_accuracy=float(np.mean(val==y[ix['validation']])),fit_seconds=model['fit_seconds']))
                print('Native selection',dataset,mode,c,grid[-1],flush=True)
            lock.write_text(json.dumps(dict(mode=mode,grid=grid,chosen=max(grid,key=lambda r:r['validation_accuracy']),
                test_metrics_used=False,selection_seed=42,selection_training_indices=selected.tolist()),indent=2))
        config=json.loads(lock.read_text())['chosen']
        settings=[(seed,b) for seed in [42,314,2718] for b in ([100,25] if dataset=='gtsrb' else [25,10])] if mode=='balanced' else [(42,None)]
        for seed,budget in settings:
            name=f'{mode}_seed_{seed}_budget_{budget}'
            if (folder/(name+'_metrics.json')).exists():continue
            selected=sample(ix['train'],y,seed,budget) if budget else ix['train']
            model=fit_softmax(x[selected],y[selected],config['c']);start=time.perf_counter();s=scores(model,x[ix['test']]);elapsed=time.perf_counter()-start
            save_evaluation(rows,y,ix['test'],s,dict(model=model,training_indices=selected,dimension=768),folder,name,
                dict(dataset=dataset,backbone='convnext_native_768',seed=seed,budget_per_class=budget,
                     actual_head_training_count=len(selected),configuration=config,fit_seconds=model['fit_seconds'],
                     prediction_seconds=elapsed,mode=mode,config_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['gtsrb','belgium']);main(p.parse_args().dataset)
