"""Audit and aggregate completed extension runs; no inferred or filled-in results."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from extension_features import manifest_path

ROOT=Path(__file__).resolve().parent
CELLS=['gtsrb_convnext_tiny','belgium_vgg16','belgium_convnext_tiny']
NAMES=['softmax_none','svm_rbf','svm_wavelet','twsvm_rbf','twsvm_wavelet','svm_rbf_matched','twsvm_rbf_matched']
SEEDS=[42,314,2718]

def read_predictions(path):
    with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
    return rows,np.array([int(r['y_true']) for r in rows]),np.array([int(r['y_pred']) for r in rows])

def aggregate(reports):
    values={key:[r['accuracy'] if key=='accuracy' else r['macro'][key] if key in ['recall','f1','average_precision'] else r[key] for r in reports]
            for key in ['accuracy','recall','f1','average_precision','fit_seconds','prediction_seconds','model_bytes']}
    return {k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),values=v) for k,v in values.items()}

def main():
    output=ROOT/'extension'/'analysis';output.mkdir(exist_ok=True)
    snapshot=json.loads((output/'delivery_input_hashes.json').read_text())
    for record in snapshot['files']:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    truth={d:{r['sample_id']:r['label'] for r in json.loads(manifest_path(d).read_text()) if r['split']=='test'} for d in ['gtsrb','belgium']}
    with (ROOT/'data'/'GT-final_test.csv').open(newline='') as stream:
        official_gtsrb={'test/'+r['Filename']:int(r['ClassId']) for r in csv.DictReader(stream,delimiter=';')}
    official_belgium={}
    for table in sorted((ROOT/'data'/'belgium'/'Testing').glob('*/GT-*.csv')):
        with table.open(newline='') as stream:
            for r in csv.DictReader(stream,delimiter=';'):
                identity='belgium/Testing/'+table.parent.name+'/'+r['Filename']
                if identity in official_belgium:assert official_belgium[identity]==int(r['ClassId'])
                official_belgium[identity]=int(r['ClassId'])
    assert truth['gtsrb']==official_gtsrb and len(official_gtsrb)==12630
    assert truth['belgium']==official_belgium and len(official_belgium)==2520
    bts=json.loads((ROOT/'data'/'belgium'/'split_summary.json').read_text())
    assert bts['population']==dict(train=3650,validation=908,test=2520)
    records=[];fallback_certificates=[]
    paths=list((ROOT/'extension'/'results').rglob('*predictions.csv'))+list((ROOT/'extension'/'native').rglob('*predictions.csv'))+list((ROOT/'extension'/'dino').rglob('*predictions.csv'))
    assert len(paths)==142,f'Expected 126 controlled heads, 14 ConvNeXt probes and 2 DINOv2 probes, got {len(paths)}'
    for path in sorted(paths):
        report=json.loads(path.with_name(path.name.replace('_predictions.csv','_metrics.json')).read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report['predictions_sha256']
        rows,y,p=read_predictions(path);dataset=report['dataset'];expected=truth[dataset]
        measured={r['sample_id']:int(r['y_true']) for r in rows}
        assert len(measured)==len(rows) and measured==expected
        assert abs(np.mean(y==p)-report['accuracy'])<1e-12
        assert report['sample_count']==len(expected)
        assert report['macro_class_count']==(43 if dataset=='gtsrb' else 53)
        score_matrix=np.array([[float(r[f'score_{c}']) for c in range(report['class_count'])] for r in rows])
        assert np.isfinite(score_matrix).all() and np.array_equal(score_matrix.argmax(1),p)
        if report['backbone'] in ['convnext_native_768','dinov2_vits14']:
            final_lock=path.parent/(report['mode']+'_expanded_locked.json')
            final_config=json.loads(final_lock.read_text())
            assert not final_config['test_metrics_used'] and report['configuration']['c']==final_config['chosen']['c']
            assert report['expanded_grid_lock_sha256']==hashlib.sha256(final_lock.read_bytes()).hexdigest()
        records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=report['predictions_sha256'],dataset=dataset,count=len(rows),correct=int(np.sum(y==p))))
        if path.name.startswith('twsvm_'):
            with path.with_name(path.name.replace('_predictions.csv','_model.pkl')).open('rb') as stream:bundle=pickle.load(stream)
            for pair in bundle['model']['pairs']:
                for diagnostic in pair['diagnostics']:
                    if diagnostic.get('solver')=='L-BFGS-B verified box QP':
                        assert diagnostic['projected_gradient_inf']<=1e-7
                        assert -1e-7<=diagnostic['gap']<=1e-7*(1+abs(diagnostic['dual_objective']))
                        fallback_certificates.append(dict(model=path.relative_to(ROOT).as_posix(),classes=[pair['p'],pair['q']],**diagnostic))
    agg={}
    for cell in CELLS:
        dataset=cell.split('_')[0];agg[cell]={}
        folder=ROOT/'extension'/'results'/cell
        lock=folder/'selection'/'locked_configs.json';locked=json.loads(lock.read_text())
        assert not locked['test_metrics_used']
        assert locked['validation_log_sha256']==hashlib.sha256((folder/'selection'/'validation_grid.json').read_bytes()).hexdigest()
        for budget in ([100,25] if dataset=='gtsrb' else [25,10]):
            agg[cell][str(budget)]={}
            for name in NAMES:
                reports=[json.loads((folder/f'seed_{seed}'/f'budget_{budget}'/(name+'_metrics.json')).read_text()) for seed in SEEDS]
                assert all(r['config_lock_sha256']==hashlib.sha256(lock.read_bytes()).hexdigest() for r in reports)
                agg[cell][str(budget)][name]=aggregate(reports)
    paired=[]
    for cell in CELLS:
        dataset=cell.split('_')[0];folder=ROOT/'extension'/'results'/cell
        for budget in ([100,25] if dataset=='gtsrb' else [25,10]):
            for seed in SEEDS:
                directory=folder/f'seed_{seed}'/f'budget_{budget}'
                for head in ['svm','twsvm']:
                    ra,ya,pa=read_predictions(directory/(head+'_wavelet_predictions.csv'))
                    rb,yb,pb=read_predictions(directory/(head+'_rbf_matched_predictions.csv'))
                    assert [r['sample_id'] for r in ra]==[r['sample_id'] for r in rb] and np.array_equal(ya,yb)
                    wins=int(np.sum((pa==ya)&(pb!=ya)));losses=int(np.sum((pa!=ya)&(pb==ya)))
                    pvalue=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
                    paired.append(dict(cell=cell,budget=budget,seed=seed,head=head,wins=wins,losses=losses,
                        disagreements=int(np.sum(pa!=pb)),wavelet_minus_matched_accuracy=float(np.mean(pa==ya)-np.mean(pb==ya)),p=pvalue))
    m=len(paired);running=0
    for rank,index in enumerate(sorted(range(m),key=lambda i:paired[i]['p'])):
        running=max(running,min(1.,(m-rank)*paired[index]['p']));paired[index]['holm_p']=running
    native={}
    for dataset in ['gtsrb','belgium']:
        folder=ROOT/'extension'/'native'/dataset
        native[dataset]={'all_training':json.loads((folder/'all_training_seed_42_budget_None_metrics.json').read_text()),'balanced':{}}
        native[dataset]['dino_all_training']=json.loads((ROOT/'extension'/'dino'/dataset/'all_training_metrics.json').read_text())
        assert not json.loads((ROOT/'extension'/'dino'/dataset/'locked.json').read_text())['test_metrics_used']
        for budget in ([100,25] if dataset=='gtsrb' else [25,10]):
            reports=[json.loads((folder/f'balanced_seed_{seed}_budget_{budget}_metrics.json').read_text()) for seed in SEEDS]
            native[dataset]['balanced'][str(budget)]=aggregate(reports)
    (output/'aggregate.json').write_text(json.dumps(agg,indent=2))
    (output/'paired.json').write_text(json.dumps(paired,indent=2))
    (output/'native.json').write_text(json.dumps(native,indent=2))
    (output/'qp_fallback_certificates.json').write_text(json.dumps(fallback_certificates,indent=2))
    (output/'integrity.json').write_text(json.dumps(dict(run_count=len(records),controlled_heads=126,convnext_probes=14,dino_probes=2,official_test_populations_verified=True,runs=records),indent=2))
    print('Verified 142 extension test populations, accuracy values, prediction hashes, and configuration locks',flush=True)
    print('Matched comparisons with Holm p<.05:',sum(r['holm_p']<.05 for r in paired),flush=True)
    for cell in CELLS:
        budget='100' if cell.startswith('gtsrb') else '25'
        print(cell,{name:round(100*agg[cell][budget][name]['accuracy']['mean'],3) for name in NAMES},flush=True)

if __name__=='__main__':main()
