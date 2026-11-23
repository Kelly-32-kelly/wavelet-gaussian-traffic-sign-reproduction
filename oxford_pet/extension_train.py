"""Train registered non-negative 64-dimensional representations on cached maps."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_representation import Representation
from extension_features import manifest_path

ROOT=Path(__file__).resolve().parent

class Compact(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.feature_layers=nn.Sequential(nn.Linear(768,64),nn.ReLU(),nn.Dropout(.5))
        self.classifier=nn.Linear(64,classes)
    def forward(self,x):return self.classifier(self.feature_layers(x))

def main(dataset,backbone,root=ROOT,protocol=None):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    protocol=protocol or root/'extension_protocol.json'
    rows=json.loads(manifest_path(dataset,root).read_text())
    cache=root/'extension'/'cache'/f'{dataset}_{backbone}'
    assert json.loads((cache/'progress.json').read_text())['completed']
    features=torch.from_numpy(np.load(cache/'features.npy').astype(np.float32)).cuda()
    labels=torch.tensor([r['label'] for r in rows],device='cuda',dtype=torch.long)
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation']}
    classes=len(set(r['label'] for r in rows))
    for seed in [42,314,2718]:
        folder=root/'extension'/'results'/f'{dataset}_{backbone}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'representation_complete.json').exists():continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        rng=np.random.default_rng(seed)
        if backbone=='vgg16':
            model=Representation();model.classifier=nn.Linear(64,classes)
        else:model=Compact(classes)
        model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=1e-6)
        best=float('inf');stale=0;logs=[];start=time.perf_counter()
        for epoch in range(1,61):
            model.train();order=rng.permutation(indices['train']);loss_sum=0;correct=0
            for o in range(0,len(order),64):
                ix=torch.as_tensor(order[o:o+64],device='cuda');optimizer.zero_grad(set_to_none=True)
                logits=model(features[ix]);loss=nn.functional.cross_entropy(logits,labels[ix]);loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(ix);correct+=(logits.argmax(1)==labels[ix]).sum().item()
            model.eval();val_loss=val_correct=0
            with torch.inference_mode():
                for o in range(0,len(indices['validation']),256):
                    ix=torch.as_tensor(indices['validation'][o:o+256],device='cuda')
                    logits=model(features[ix]);val_loss+=nn.functional.cross_entropy(logits,labels[ix],reduction='sum').item()
                    val_correct+=(logits.argmax(1)==labels[ix]).sum().item()
            val_loss/=len(indices['validation']);val_acc=val_correct/len(indices['validation'])
            logs.append(dict(epoch=epoch,validation_loss=val_loss,validation_accuracy=val_acc,
                train_loss=loss_sum/len(order),train_accuracy_dropout_enabled=correct/len(order),
                learning_rate=optimizer.param_groups[0]['lr'],elapsed_s=time.perf_counter()-start))
            (folder/'training_log.json').write_text(json.dumps(logs,indent=2))
            if val_loss<best:
                best=val_loss;stale=0
                torch.save(dict(model_state=model.state_dict(),epoch=epoch,seed=seed),folder/'representation.pt')
            else:stale+=1
            scheduler.step()
            if epoch%5==0:print('Train',dataset,backbone,seed,epoch,'val',round(val_acc,4),'seconds',round(time.perf_counter()-start,1),flush=True)
            if stale>=8:break
        saved=torch.load(folder/'representation.pt',weights_only=True);model.load_state_dict(saved['model_state']);model.eval()
        embeddings=np.empty((len(rows),64),dtype=np.float32);zeros=0
        with torch.inference_mode():
            for o in range(0,len(rows),256):
                h=model.feature_layers(features[o:o+256]);norm=h.norm(dim=1,keepdim=True);zeros+=(norm==0).sum().item()
                embeddings[o:o+len(h)]=(h/norm.clamp_min(1e-12)).cpu().numpy()
        np.save(folder/'embedding.npy',embeddings)
        (folder/'representation_complete.json').write_text(json.dumps(dict(dataset=dataset,backbone=backbone,seed=seed,
            selected_epoch=saved['epoch'],epochs_executed=len(logs),minimum_validation_loss=best,
            elapsed_s=time.perf_counter()-start,zero_embeddings=zeros,
            feature_parameters=sum(p.numel() for p in model.feature_layers.parameters()),
            classifier_parameters=sum(p.numel() for p in model.classifier.parameters()),
            protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),test_metrics_accessed=False),indent=2))
        del model,optimizer;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['gtsrb','belgium']);p.add_argument('backbone',choices=['vgg16','convnext_tiny'])
    a=p.parse_args();main(a.dataset,a.backbone)
