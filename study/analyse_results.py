"""Aggregate saved observations; never substitute values for missing runs."""
import csv
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'cache'/'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 314, 2718]
BUDGETS = [25, 50, 100]
NAMES = ['softmax_none', 'svm_rbf', 'svm_wavelet', 'twsvm_rbf', 'twsvm_wavelet',
         'svm_rbf_matched', 'twsvm_rbf_matched']
LABELS = dict(zip(NAMES, ['Softmax', 'SVM: tuned RBF', 'SVM: wavelet', 'TWSVM: tuned RBF',
                         'TWSVM: wavelet', 'SVM: matched RBF', 'TWSVM: matched RBF']))


def observations(seed, budget, name):
    folder = ROOT/'results'/f'seed_{seed}'/f'budget_{budget}'
    report = json.loads((folder/(name+'_metrics.json')).read_text())
    prediction = folder/(name+'_predictions.csv')
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == report['predictions_sha256']
    return report


def predicted(seed, name):
    path = ROOT/'results'/f'seed_{seed}'/'budget_100'/(name+'_predictions.csv')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return ([r['sample_id'] for r in rows], np.array([int(r['y_true']) for r in rows]),
            np.array([int(r['y_pred']) for r in rows]))


def main():
    output = ROOT/'analysis'
    output.mkdir(exist_ok=True)
    figures = ROOT.parent/'111'/'study_figures'
    figures.mkdir(exist_ok=True)
    aggregates = {}
    raw = []
    for budget in BUDGETS:
        aggregates[str(budget)] = {}
        for name in NAMES:
            runs = [observations(seed, budget, name) for seed in SEEDS]
            values = dict(accuracy=[r['accuracy'] for r in runs],
                macro_precision=[r['macro']['precision'] for r in runs],
                macro_recall=[r['macro']['recall'] for r in runs],
                macro_f1=[r['macro']['f1'] for r in runs],
                macro_ap=[r['macro']['average_precision'] for r in runs],
                fit_seconds=[r['fit_seconds'] for r in runs],
                kernel_seconds=[r['kernel_seconds'] for r in runs],
                prediction_seconds=[r['prediction_seconds'] for r in runs],
                model_bytes=[r['model_bytes'] for r in runs])
            aggregates[str(budget)][name] = {key:dict(mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)), values=v) for key,v in values.items()}
            for seed, r in zip(SEEDS, runs):
                raw.append(dict(seed=seed,budget=budget,head=name,
                    **{key:values[key][SEEDS.index(seed)] for key in values}))
    gap={}
    for budget in BUDGETS:
        runs=[json.loads((ROOT/'results'/'gap_control'/f'seed_{seed}'/f'budget_{budget}_metrics.json').read_text())
              for seed in SEEDS]
        values=dict(accuracy=[r['accuracy'] for r in runs],
            macro_precision=[r['macro']['precision'] for r in runs],
            macro_recall=[r['macro']['recall'] for r in runs],
            macro_f1=[r['macro']['f1'] for r in runs],
            macro_ap=[r['macro']['average_precision'] for r in runs])
        gap[str(budget)]={key:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),values=v)
                          for key,v in values.items()}
    (output/'gap_aggregate.json').write_text(json.dumps(gap,indent=2))
    (output/'aggregate_results.json').write_text(json.dumps(aggregates,indent=2))
    with (output/'all_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)

    # Paired tests within each seed. Three runs on the same test images are
    # never pooled as three independent test populations.
    paired=[]
    comparison_pairs=[('svm_wavelet','svm_rbf_matched'),
                      ('twsvm_wavelet','twsvm_rbf_matched'),
                      ('twsvm_wavelet','svm_wavelet'),
                      ('twsvm_wavelet','softmax_none')]
    for seed in SEEDS:
        for a,b in comparison_pairs:
            ids,y,pa=predicted(seed,a)
            ids_b,yb,pb=predicted(seed,b)
            assert ids==ids_b and np.array_equal(y,yb)
            wins=int(np.sum((pa==y)&(pb!=y)))
            losses=int(np.sum((pa!=y)&(pb==y)))
            pvalue=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
            paired.append(dict(seed=seed,a=a,b=b,a_correct_b_wrong=wins,
                a_wrong_b_correct=losses,prediction_disagreement=int(np.sum(pa!=pb)),
                exact_mcnemar_p=float(pvalue),bonferroni_p=min(1.,12*pvalue)))
    (output/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))

    # Measured kernel discrepancies on a fixed validation-only random sample.
    from evaluate_heads import load
    from kernel_heads import kernel
    _,features,_,indices=load(42)
    rng=np.random.default_rng(20261012)
    sample=rng.choice(indices['validation'],size=512,replace=False)
    x=np.asarray(features[sample],dtype=np.float64)
    discrepancy=[]
    for sigma in [1.,2.,4.]:
        kw=kernel(x,x,'wavelet',sigma)
        kr=kernel(x,x,'rbf',3/(2*sigma*sigma))
        delta=np.abs(kw-kr)
        if sigma>1:
            u=(x[:,None,:]-x[None,:,:])**2/sigma**2
            lower=np.exp(-1.5*u.sum(-1)-(u*u).sum(-1)/(2*(1-1/sigma**2)))
            if np.max(lower-kw)>1e-12 or np.max(kw-kr)>1e-12:
                raise AssertionError('Gaussian comparison bound violated')
        discrepancy.append(dict(sigma=sigma,matched_gamma=3/(2*sigma*sigma),
            maximum_absolute_difference=float(delta.max()),
            mean_absolute_difference=float(delta.mean()),
            rmse=float(np.sqrt(np.mean(delta**2)))))
    (output/'kernel_discrepancy.json').write_text(json.dumps(dict(seed=42,
        split='validation',indices=sample.tolist(),pairs=512**2,results=discrepancy),indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,4.2))
    for name,marker,linestyle in zip(NAMES[:5],['o','s','^','D','v'],['-','--','-.',':','-']):
        means=[100*aggregates[str(b)][name]['accuracy']['mean'] for b in BUDGETS]
        sd=[100*aggregates[str(b)][name]['accuracy']['sd'] for b in BUDGETS]
        ax.errorbar(BUDGETS,means,yerr=sd,marker=marker,linestyle=linestyle,capsize=3,label=LABELS[name])
    ax.set(xlabel='Head training examples per class',ylabel='Official-test accuracy (%)',
           xticks=BUDGETS)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(figures/'head_budget.pdf')
    fig.savefig(figures/'head_budget.png',dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    for seed,style in zip(SEEDS,['-','--','-.']):
        log=json.loads((ROOT/'results'/f'seed_{seed}'/'training_log.json').read_text())
        epochs=[r['epoch'] for r in log]
        axes[0].plot(epochs,[r['validation_loss'] for r in log],linestyle=style,label=f'Seed {seed}')
        axes[1].plot(epochs,[100*r['validation_accuracy'] for r in log],linestyle=style,label=f'Seed {seed}')
    axes[0].set(xlabel='Epoch',ylabel='Validation cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy (%)')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures/'validation_training.pdf')
    fig.savefig(figures/'validation_training.png',dpi=220)
    plt.close(fig)

    confusions=[np.asarray(observations(s,100,'twsvm_wavelet')['confusion_matrix']) for s in SEEDS]
    matrix=np.mean(confusions,axis=0)
    recall=np.divide(matrix,matrix.sum(axis=1,keepdims=True))
    fig,ax=plt.subplots(figsize=(6.3,5.2))
    picture=ax.imshow(recall,cmap='Blues',vmin=0,vmax=1)
    ax.set(xlabel='Predicted class index',ylabel='True class index',
           xticks=range(0,43,5),yticks=range(0,43,5))
    fig.colorbar(picture,ax=ax,label='Mean row-normalised fraction')
    fig.tight_layout()
    fig.savefig(figures/'wavelet_confusion.pdf')
    fig.savefig(figures/'wavelet_confusion.png',dpi=220)
    plt.close(fig)
    print('Aggregated 63 audited runs and created empirical figures',flush=True)


if __name__=='__main__':
    main()
