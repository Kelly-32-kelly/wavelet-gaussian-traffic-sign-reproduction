"""Registered architecture-efficiency control using the same frozen VGG maps."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_representation import tensor_batch, validate
from evaluate_heads import budget_indices
from kernel_heads import fit_softmax, scores
from audit_predictions import audit_rows
import csv
import hashlib
import pickle
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parent


class GAPRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_layers=nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Dropout(.5))
        self.classifier=nn.Linear(64,43)
    def forward(self,x):
        return self.classifier(self.feature_layers(x))


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/'results'/'gap_control'
    folder.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'data'/'manifest.json').read_text())
    maps=np.load(ROOT/'cache'/'vgg_conv.npy',mmap_mode='r')
    gap_path=folder/'pooled_features.npy'
    if not gap_path.exists():
        pooled=np.lib.format.open_memmap(gap_path,mode='w+',dtype='float32',shape=(len(rows),512))
        for i in range(0,len(rows),256):
            pooled[i:i+256]=np.asarray(maps[i:i+256],dtype=np.float32).reshape(-1,512,49).mean(-1)
        pooled.flush()
    features=np.load(gap_path,mmap_mode='r')
    y=np.array([r['label'] for r in rows],dtype=np.int64)
    split={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    for seed in [42,314,2718]:
        location=folder/f'seed_{seed}'
        location.mkdir(exist_ok=True)
        if (location/'representation_complete.json').exists():
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng=np.random.default_rng(seed)
        model=GAPRepresentation().cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=1e-6)
        best=float('inf');stale=0;logs=[]
        started=time.perf_counter()
        for epoch in range(1,61):
            model.train()
            order=rng.permutation(split['train'])
            for offset in range(0,len(order),64):
                ix=order[offset:offset+64]
                target=torch.from_numpy(y[ix]).cuda()
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(tensor_batch(features,ix)),target)
                loss.backward();optimizer.step()
            val_loss,val_acc=validate(model,features,split['validation'],y)
            logs.append(dict(epoch=epoch,validation_loss=val_loss,validation_accuracy=val_acc,
                             elapsed_s=time.perf_counter()-started))
            (location/'training_log.json').write_text(json.dumps(logs,indent=2))
            if val_loss<best:
                best=val_loss;stale=0
                torch.save(dict(model_state=model.state_dict(),epoch=epoch),location/'representation.pt')
            else:
                stale+=1
            schedule.step()
            if epoch%10==0:
                print('GAP',seed,'epoch',epoch,'validation',val_acc,flush=True)
            if stale>=8:
                break
        saved=torch.load(location/'representation.pt',weights_only=True)
        model.load_state_dict(saved['model_state']);model.eval()
        embedding=np.empty((len(rows),64),dtype=np.float32)
        with torch.inference_mode():
            for i in range(0,len(rows),256):
                ix=np.arange(i,min(i+256,len(rows)))
                h=model.feature_layers(tensor_batch(features,ix))
                h=h/torch.linalg.vector_norm(h,dim=1,keepdim=True).clamp_min(1e-12)
                embedding[ix]=h.cpu().numpy()
        np.save(location/'embedding.npy',embedding)
        (location/'representation_complete.json').write_text(json.dumps(dict(seed=seed,
            selected_epoch=saved['epoch'],minimum_validation_loss=best,epochs_executed=len(logs),
            elapsed_s=time.perf_counter()-started,feature_parameters=32832,test_metrics_accessed=False),indent=2))
    lock=folder/'locked_config.json'
    if not lock.exists():
        x=np.load(folder/'seed_42'/'embedding.npy')
        selected=budget_indices(split['train'],y,42,100)
        grid=[]
        for c in [.1,1.,10.]:
            model=fit_softmax(x[selected],y[selected],c)
            pred=np.argmax(scores(model,x[split['validation']]),axis=1)
            grid.append(dict(c=c,validation_accuracy=float(np.mean(pred==y[split['validation']]))))
        chosen=max(grid,key=lambda r:r['validation_accuracy'])
        lock.write_text(json.dumps(dict(grid=grid,chosen=chosen,test_metrics_used=False),indent=2))
    config=json.loads(lock.read_text())['chosen']
    for seed in [42,314,2718]:
        x=np.load(folder/f'seed_{seed}'/'embedding.npy')
        for budget in [100,50,25]:
            report_path=folder/f'seed_{seed}'/f'budget_{budget}_metrics.json'
            if report_path.exists():
                continue
            selected=budget_indices(split['train'],y,seed,budget)
            model=fit_softmax(x[selected],y[selected],config['c'])
            started=time.perf_counter()
            s=scores(model,x[split['test']])
            seconds=time.perf_counter()-started
            predictions=[dict(sample_id=rows[ix]['sample_id'],y_true=str(y[ix]),
                y_pred=str(np.argmax(sc)),**{f'score_{c}':str(sc[c]) for c in range(43)})
                for ix,sc in zip(split['test'],s)]
            prediction_path=report_path.with_name(f'budget_{budget}_predictions.csv')
            with prediction_path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
            report=audit_rows(predictions,43)
            ap=float(average_precision_score(np.eye(43)[y[split['test']]],s,average='macro'))
            if not np.isclose(ap,report['macro']['average_precision'],atol=1e-12):
                raise AssertionError('Independent GAP AP implementations disagree')
            with (folder/f'seed_{seed}'/f'budget_{budget}_model.pkl').open('wb') as f:
                pickle.dump(dict(model=model,training_indices=selected,configuration=config),f)
            report.update(seed=seed,budget_per_class=budget,configuration=config,
                          fit_seconds=model['fit_seconds'],prediction_seconds=seconds,
                          predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest())
            report_path.write_text(json.dumps(report,indent=2))
            print('GAP test',seed,budget,report['accuracy'],flush=True)


if __name__=='__main__':
    main()
