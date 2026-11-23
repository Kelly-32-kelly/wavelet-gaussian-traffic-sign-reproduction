"""Validation-only configuration selection and full-test evaluation of extension heads."""
import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from kernel_heads import kernel,fit_softmax,fit_svm,fit_twsvm,scores
from audit_predictions import audit_rows
from extension_features import manifest_path

ROOT=Path(__file__).resolve().parent
NAMES=['softmax_none','svm_rbf','svm_wavelet','twsvm_rbf','twsvm_wavelet']
FITTERS={'softmax':fit_softmax,'svm':fit_svm,'twsvm':fit_twsvm}

def load(dataset,backbone,seed,root=ROOT):
    rows=json.loads(manifest_path(dataset,root).read_text())
    folder=root/'extension'/'results'/f'{dataset}_{backbone}'
    x=np.load(folder/f'seed_{seed}'/'embedding.npy');y=np.array([r['label'] for r in rows])
    ix={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    return rows,x,y,ix,folder

def sample(train,y,seed,budget):
    rng=np.random.default_rng(seed);selected=[]
    for c in sorted(set(y)):
        candidates=train[y[train]==c].copy();rng.shuffle(candidates);selected.extend(candidates[:budget])
    return np.array(selected,dtype=np.int64)

def tune(dataset,backbone,root=ROOT,budget=None,protocol=None):
    rows,x,y,ix,folder=load(dataset,backbone,42,root);directory=folder/'selection';directory.mkdir(exist_ok=True)
    protocol=protocol or root/'extension_protocol.json'
    existing=directory/'locked_configs.json'
    if existing.exists():
        record=json.loads(existing.read_text())
        assert record['validation_log_sha256']==hashlib.sha256((directory/'validation_grid.json').read_bytes()).hexdigest()
        print('Preserved existing configuration lock',dataset,backbone,flush=True)
        return
    budget=budget if budget is not None else (100 if dataset=='gtsrb' else 25);selected=sample(ix['train'],y,42,budget)
    grid_path=directory/'validation_grid.json';grid=json.loads(grid_path.read_text()) if grid_path.exists() else []
    done={(r['head'],r['kind'],r['parameter'],r['c']) for r in grid}
    inputs=[('none',0.,['softmax'])]+[(kind,p,['svm','twsvm']) for kind,parameters in [('rbf',[.09375,.375,1.5]),('wavelet',[1.,2.,4.])] for p in parameters]
    for kind,p,heads in inputs:
        if all((h,kind,p,c) in done for h in heads for c in [.1,1.,10.]):continue
        kt=x[selected] if kind=='none' else kernel(x[selected],x[selected],kind,p)
        kv=x[ix['validation']] if kind=='none' else kernel(x[ix['validation']],x[selected],kind,p)
        for head in heads:
            for c in [.1,1.,10.]:
                if (head,kind,p,c) in done:continue
                try:
                    model=FITTERS[head](kt,y[selected],c)
                except RuntimeError as error:
                    if 'Non-optimal QP' not in str(error):raise
                    grid.append(dict(head=head,kind=kind,parameter=p,c=c,status='failed_optimality',
                        val_accuracy=None,reason=str(error)))
                    grid_path.write_text(json.dumps(grid,indent=2))
                    print('Ineligible validation candidate',dataset,backbone,head,kind,p,c,str(error),flush=True)
                    continue
                prediction=scores(model,kv).argmax(1)
                record=dict(head=head,kind=kind,parameter=p,c=c,val_accuracy=float(np.mean(prediction==y[ix['validation']])),fit_seconds=model['fit_seconds'])
                grid.append(record);grid_path.write_text(json.dumps(grid,indent=2));print('Selection',dataset,backbone,head,kind,p,c,record['val_accuracy'],flush=True)
    chosen={}
    for head,kind in [('softmax','none'),('svm','rbf'),('svm','wavelet'),('twsvm','rbf'),('twsvm','wavelet')]:
        chosen[head+'_'+kind]=max([r for r in grid if r['head']==head and r['kind']==kind and r['val_accuracy'] is not None],key=lambda r:r['val_accuracy'])
    (directory/'locked_configs.json').write_text(json.dumps(dict(dataset=dataset,backbone=backbone,seed=42,budget=budget,parameters=chosen,
        selected_indices=selected.tolist(),test_metrics_used=False,
        validation_log_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()),indent=2))

def save_evaluation(rows,y,test,s,model,folder,name,metadata):
    classes=len(set(y));assert s.shape==(len(test),classes) and np.isfinite(s).all()
    predictions=[dict(sample_id=rows[i]['sample_id'],y_true=str(y[i]),y_pred=str(np.argmax(sc)),
        **{f'score_{c}':str(sc[c]) for c in range(classes)}) for i,sc in zip(test,s)]
    path=folder/(name+'_predictions.csv')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    report=audit_rows(predictions,classes,allow_missing_test_classes=True)
    supported=report['macro_class_ids']
    assert np.isclose(report['macro']['average_precision'],average_precision_score(np.eye(classes)[y[test]][:,supported],s[:,supported],average='macro'),atol=1e-12)
    with (folder/(name+'_model.pkl')).open('wb') as stream:pickle.dump(model,stream)
    report.update(metadata,predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_bytes=(folder/(name+'_model.pkl')).stat().st_size)
    (folder/(name+'_metrics.json')).write_text(json.dumps(report,indent=2))
    print('Test',metadata.get('dataset'),metadata.get('backbone'),metadata.get('seed'),metadata.get('budget_per_class'),name,report['accuracy'],flush=True)

def evaluate(dataset,backbone,root=ROOT,budgets=None):
    budgets=budgets if budgets is not None else ([100,25] if dataset=='gtsrb' else [25,10])
    for seed in [42,314,2718]:
        rows,x,y,ix,folder=load(dataset,backbone,seed,root)
        lock=json.loads((folder/'selection'/'locked_configs.json').read_text());configs=dict(lock['parameters'])
        for head in ['svm','twsvm']:
            source=configs[head+'_wavelet'];configs[head+'_rbf_matched']=dict(source,kind='rbf',parameter=3/(2*source['parameter']**2))
        for budget in budgets:
            directory=folder/f'seed_{seed}'/f'budget_{budget}';directory.mkdir(exist_ok=True)
            selected=sample(ix['train'],y,seed,budget)
            (directory/'training_sample_indices.json').write_text(json.dumps(selected.tolist()))
            for name,config in configs.items():
                if (directory/(name+'_metrics.json')).exists():continue
                start=time.perf_counter();kind,p=config['kind'],config['parameter']
                kt=x[selected] if kind=='none' else kernel(x[selected],x[selected],kind,p)
                kv=x[ix['test']] if kind=='none' else kernel(x[ix['test']],x[selected],kind,p)
                kernel_seconds=time.perf_counter()-start
                model=(fit_twsvm(kt,y[selected],config['c'],allow_fallback=True) if config['head']=='twsvm'
                       else FITTERS[config['head']](kt,y[selected],config['c']))
                start=time.perf_counter();s=scores(model,kv);prediction_seconds=time.perf_counter()-start
                metadata=dict(dataset=dataset,backbone=backbone,seed=seed,budget_per_class=budget,actual_head_training_count=len(selected),
                    per_class_training_count={str(c):int(np.sum(y[selected]==c)) for c in sorted(set(y))},configuration=config,
                    reference_rows=selected.tolist(),fit_seconds=model['fit_seconds'],kernel_seconds=kernel_seconds,prediction_seconds=prediction_seconds,
                    config_lock_sha256=hashlib.sha256((folder/'selection'/'locked_configs.json').read_bytes()).hexdigest())
                save_evaluation(rows,y,ix['test'],s,dict(model=model,training_indices=selected,reference_features=x[selected],kind=kind,parameter=p),directory,name,metadata)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['gtsrb','belgium']);p.add_argument('backbone',choices=['vgg16','convnext_tiny']);p.add_argument('phase',choices=['tune','evaluate'])
    a=p.parse_args();(tune if a.phase=='tune' else evaluate)(a.dataset,a.backbone)
