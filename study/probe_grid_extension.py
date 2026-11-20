"""Expand boundary-selected native probes using validation only, with original records retained."""
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from extension_features import manifest_path
from extension_heads import sample,save_evaluation
from kernel_heads import fit_softmax,scores

ROOT=Path(__file__).resolve().parent

def main():
    for dataset in ['gtsrb','belgium']:
        rows=json.loads(manifest_path(dataset).read_text());y=np.array([r['label'] for r in rows])
        ix={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
        for backbone in ['convnext','dino']:
            if backbone=='convnext':
                features=ROOT/'extension'/'cache'/f'{dataset}_convnext_tiny'/'features.npy';folder=ROOT/'extension'/'native'/dataset
                modes=['balanced','all_training'];dimension=768
            else:
                folder=ROOT/'extension'/'dino'/dataset;features=folder/'features.npy';modes=['all_training'];dimension=384
            x=np.load(features).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
            for mode in modes:
                initial=folder/((mode+'_locked.json') if backbone=='convnext' else 'locked.json')
                lock=folder/(mode+'_expanded_locked.json')
                if not lock.exists():
                    record=json.loads(initial.read_text());grid=list(record['grid'])
                    selected=sample(ix['train'],y,42,100 if dataset=='gtsrb' else 25) if mode=='balanced' else ix['train']
                    pending=folder/(mode+'_expanded_validation.json')
                    if pending.exists():grid=json.loads(pending.read_text())
                    done={r['c'] for r in grid}
                    for c in [100.,1000.]:
                        if c in done:continue
                        try:
                            clf=fit_softmax(x[selected],y[selected],c)
                        except RuntimeError as error:
                            if 'did not converge' not in str(error):raise
                            grid.append(dict(c=c,validation_accuracy=None,status='ineligible_nonconvergence',reason=str(error)))
                        else:
                            validation=scores(clf,x[ix['validation']]).argmax(1)
                            grid.append(dict(c=c,validation_accuracy=float(np.mean(validation==y[ix['validation']])),fit_seconds=clf['fit_seconds']))
                        pending.write_text(json.dumps(grid,indent=2))
                        print('Expanded probe validation',dataset,backbone,mode,grid[-1],flush=True)
                    chosen=max([r for r in grid if r['validation_accuracy'] is not None],key=lambda r:r['validation_accuracy'])
                    lock.write_text(json.dumps(dict(dataset=dataset,backbone=backbone,mode=mode,grid=grid,chosen=chosen,
                        test_metrics_used=False,initial_lock_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                        protocol_sha256=hashlib.sha256((ROOT/'probe_grid_extension_protocol.json').read_bytes()).hexdigest()),indent=2))
                config=json.loads(lock.read_text())['chosen']
                settings=[(seed,b) for seed in [42,314,2718] for b in ([100,25] if dataset=='gtsrb' else [25,10])] if mode=='balanced' else [(42,None)]
                for seed,budget in settings:
                    name=f'{mode}_seed_{seed}_budget_{budget}' if backbone=='convnext' else 'all_training'
                    metric=folder/(name+'_metrics.json')
                    previous=json.loads(metric.read_text()) if metric.exists() else {}
                    if previous.get('expanded_grid_lock_sha256')==hashlib.sha256(lock.read_bytes()).hexdigest():continue
                    archive=ROOT/'extension'/'diagnostic_native_before_grid_expansion'/f'{dataset}_{backbone}'
                    archive.mkdir(parents=True,exist_ok=True)
                    for suffix in ['_metrics.json','_predictions.csv','_model.pkl']:
                        original=folder/(name+suffix);retained=archive/(name+suffix)
                        if original.exists() and not retained.exists():shutil.copy2(original,retained)
                    selected=sample(ix['train'],y,seed,budget) if budget else ix['train']
                    clf=fit_softmax(x[selected],y[selected],config['c']);start=time.perf_counter();s=scores(clf,x[ix['test']]);elapsed=time.perf_counter()-start
                    save_evaluation(rows,y,ix['test'],s,dict(model=clf,training_indices=selected,dimension=dimension),folder,name,
                        dict(dataset=dataset,backbone='convnext_native_768' if backbone=='convnext' else 'dinov2_vits14',seed=seed if backbone=='convnext' else None,
                            budget_per_class=budget,actual_head_training_count=len(selected),configuration=config,
                            fit_seconds=clf['fit_seconds'],prediction_seconds=elapsed,mode=mode,
                            expanded_grid_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest()))
    print('All validation-expanded native probes completed',flush=True)

if __name__=='__main__':main()
