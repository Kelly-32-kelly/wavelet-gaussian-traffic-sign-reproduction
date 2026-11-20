"""Regenerate the cross-representation figure from supplied audited aggregates."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    old=json.loads((ROOT/'analysis'/'aggregate_results.json').read_text())
    new=json.loads((ROOT/'extension'/'analysis'/'aggregate.json').read_text())
    heads=['softmax_none','svm_rbf','svm_wavelet','twsvm_rbf','twsvm_wavelet']
    def value(cell,name):
        return old['100'][name] if cell=='gtsrb_vgg16' else new[cell]['100' if cell.startswith('gtsrb') else '25'][name]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.7),sharey=True);positions=np.arange(5);width=.36
    for ax,dataset,label in zip(axes,['gtsrb','belgium'],['GTSRB','BelgiumTSC']):
        for i,(backbone,legend) in enumerate([('vgg16','VGG16 / 64-D'),('convnext_tiny','ConvNeXt / 64-D')]):
            means=[100*value(dataset+'_'+backbone,name)['accuracy']['mean'] for name in heads]
            errors=[100*value(dataset+'_'+backbone,name)['accuracy']['sd'] for name in heads]
            ax.bar(positions+(i-.5)*width,means,width,yerr=errors,capsize=2,label=legend,color=['#365f91','#d4863b'][i])
        ax.set(title=label,xticks=positions,xticklabels=['Softmax','RBF\nSVM','Wavelet\nSVM','RBF\nTWSVM','Wavelet\nTWSVM'],ylim=(0,100))
        ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Official-test accuracy (%)');axes[1].legend(fontsize=9,loc='lower right')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.tight_layout();fig.savefig(out/'cross_backbone.pdf');fig.savefig(out/'cross_backbone.png',dpi=220);plt.close(fig)
    print('Reproduced cross-representation figure from audited aggregate files')

if __name__=='__main__':main()
