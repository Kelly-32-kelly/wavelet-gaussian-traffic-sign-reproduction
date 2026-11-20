import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from audit_predictions import audit_rows, audit_manifest
from kernel_heads import kernel, fit_twsvm, fit_svm, fit_softmax, scores

ROOT = Path(__file__).resolve().parent
CS = [0.1, 1., 10.]
PARAMETERS = {'wavelet': [1., 2., 4.], 'rbf': [0.09375, 0.375, 1.5]}


def load(seed):
    rows = json.loads((ROOT/'data'/'manifest.json').read_text())
    features = np.load(ROOT/'results'/f'seed_{seed}'/'embedding.npy', mmap_mode='r')
    labels = np.array([r['label'] for r in rows])
    indices = {s: np.array([i for i, r in enumerate(rows) if r['split']==s])
               for s in ['train', 'validation', 'test']}
    return rows, features, labels, indices


def budget_indices(train_indices, labels, seed, budget):
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(43):
        candidates = train_indices[labels[train_indices] == c].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[:budget])
    return np.array(selected, dtype=np.int64)


def tuning():
    rows, x, y, indices = load(42)
    selected = budget_indices(indices['train'], y, 42, 100)
    val = indices['validation']
    directory = ROOT/'results'/'selection'
    directory.mkdir(exist_ok=True)
    (directory/'training_sample_indices.json').write_text(json.dumps(selected.tolist()))
    results = []
    log_path = directory/'validation_grid.json'
    if log_path.exists():
        results = json.loads(log_path.read_text())
    completed = {(r['head'],r['kind'],r['parameter'],r['c']) for r in results}
    for c in CS:
        key = ('softmax', 'none', 0., c)
        if key in completed:
            continue
        model = fit_softmax(x[selected], y[selected], c)
        s = scores(model, x[val])
        results.append(dict(head='softmax', kind='none', parameter=0., c=c,
                            val_accuracy=float(accuracy_score(y[val], np.argmax(s,axis=1))),
                            fit_seconds=model['fit_seconds']))
        log_path.write_text(json.dumps(results, indent=2))
    for kind, parameters in PARAMETERS.items():
        for parameter in parameters:
            if all((head,kind,parameter,c) in completed for head in ['svm','twsvm'] for c in CS):
                continue
            print('Validation kernel', kind, parameter, flush=True)
            start = time.perf_counter()
            kt = kernel(x[selected], x[selected], kind, parameter)
            kv = kernel(x[val], x[selected], kind, parameter)
            kernel_time = time.perf_counter()-start
            for head in ['svm', 'twsvm']:
                for c in CS:
                    if (head,kind,parameter,c) in completed:
                        continue
                    print('Validation fit', head, kind, parameter, c, flush=True)
                    model = (fit_svm if head=='svm' else fit_twsvm)(kt, y[selected], c)
                    s = scores(model, kv)
                    accuracy = float(accuracy_score(y[val], np.argmax(s,axis=1)))
                    results.append(dict(head=head,kind=kind,parameter=parameter,c=c,
                        val_accuracy=accuracy,fit_seconds=model['fit_seconds'],kernel_seconds=kernel_time))
                    log_path.write_text(json.dumps(results, indent=2))
                    print('Validation accuracy', accuracy, flush=True)
    chosen = {}
    for head, kind in [('softmax','none'),('svm','rbf'),('svm','wavelet'),('twsvm','rbf'),('twsvm','wavelet')]:
        candidates = [r for r in results if r['head']==head and r['kind']==kind]
        # The loop order provides a fixed tie rule; test results play no part.
        choice = max(candidates, key=lambda r:r['val_accuracy'])
        chosen[head+'_'+kind] = choice
    payload = dict(seed=42,budget=100,selection_metric='validation accuracy',
        parameters=chosen,test_metrics_used=False,
        validation_log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest())
    (directory/'locked_configs.json').write_text(json.dumps(payload, indent=2))
    print('Configurations locked using validation only', flush=True)


def evaluate(seed, budget):
    locked = json.loads((ROOT/'results'/'selection'/'locked_configs.json').read_text())
    rows, x, y, indices = load(seed)
    selected = budget_indices(indices['train'],y,seed,budget)
    test = indices['test']
    directory=ROOT/'results'/f'seed_{seed}'/f'budget_{budget}'
    directory.mkdir(exist_ok=True)
    (directory/'training_sample_indices.json').write_text(json.dumps(selected.tolist()))
    summaries = {}
    configurations=dict(locked['parameters'])
    for head in ('svm','twsvm'):
        source=locked['parameters'][head+'_wavelet']
        configurations[head+'_rbf_matched']=dict(source,
            kind='rbf',parameter=3/(2*source['parameter']**2),
            comparison='Same C; Gaussian leading-exponent match to selected wavelet')
    for name, config in configurations.items():
        report_path=directory/(name+'_metrics.json')
        if report_path.exists():
            summaries[name]=json.loads(report_path.read_text())
            continue
        print('Locked test run',seed,budget,name,flush=True)
        start=time.perf_counter()
        kind, parameter=config['kind'],config['parameter']
        if kind=='none':
            fit_input, test_input=x[selected],x[test]
        else:
            fit_input=kernel(x[selected],x[selected],kind,parameter)
            test_input=kernel(x[test],x[selected],kind,parameter)
        kernel_seconds=time.perf_counter()-start
        fitter={'softmax':fit_softmax,'svm':fit_svm,'twsvm':fit_twsvm}[config['head']]
        model=fitter(fit_input,y[selected],config['c'])
        start=time.perf_counter()
        s=scores(model,test_input)
        prediction_seconds=time.perf_counter()-start
        if not np.all(np.isfinite(s)):
            raise ValueError('Nonfinite class scores')
        predicted=np.argmax(s,axis=1)
        prediction_path=directory/(name+'_predictions.csv')
        with prediction_path.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['sample_id','y_true','y_pred']+[f'score_{c}' for c in range(43)])
            writer.writerows([rows[ix]['sample_id'],int(y[ix]),int(p),*values]
                             for ix,p,values in zip(test,predicted,s))
        with prediction_path.open(newline='') as f:
            predictions=list(csv.DictReader(f))
        report=audit_rows(predictions,43)
        manifest=[{k:str(r.get(k,'')) for k in ['sample_id','split','group_id','sha256']} for r in rows]
        report['manifest_audit']=audit_manifest(manifest,[rows[i]['sample_id'] for i in test],
            allow_within_split_duplicates=True)
        ap=float(average_precision_score(np.eye(43)[y[test]],s,average='macro'))
        if not np.isclose(ap,report['macro']['average_precision'],atol=1e-12):
            raise AssertionError('Independent AP implementations disagree')
        report.update(seed=seed,budget_per_class=budget,actual_head_training_count=len(selected),
            configuration=config,fit_seconds=model['fit_seconds'],kernel_seconds=kernel_seconds,
            prediction_seconds=prediction_seconds,
            predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            reference_rows=selected.tolist())
        model_path=directory/(name+'_model.pkl')
        with model_path.open('wb') as f:
            pickle.dump(dict(model=model,training_indices=selected,
                             reference_features=np.asarray(x[selected]) if kind!='none' else None,
                             kind=kind,parameter=parameter),f)
        report['model_bytes']=model_path.stat().st_size
        report_path.write_text(json.dumps(report,indent=2))
        summaries[name]=report
        print('Test accuracy',report['accuracy'],'macro AP',ap,flush=True)
    (directory/'summary.json').write_text(json.dumps({k:{'accuracy':v['accuracy'],
        'macro':v['macro'],'fit_seconds':v['fit_seconds'],'kernel_seconds':v['kernel_seconds'],
        'prediction_seconds':v['prediction_seconds'],'model_bytes':v['model_bytes']}
        for k,v in summaries.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['tune','evaluate'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--budget',type=int,default=100)
    args=parser.parse_args()
    tuning() if args.phase=='tune' else evaluate(args.seed,args.budget)
