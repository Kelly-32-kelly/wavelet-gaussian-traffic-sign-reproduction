import argparse
import json
import os
import random
import time
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [25088, 512, 256, 128, 64]
        dropout = [0.5, 0.6, 0.55, 0.5]
        layers = []
        for i in range(4):
            layers.extend([nn.Linear(widths[i], widths[i+1]), nn.ReLU(), nn.Dropout(dropout[i])])
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, 43)

    def forward(self, x):
        return self.classifier(self.feature_layers(x))


def tensor_batch(features, indices):
    values = np.array(features[indices], dtype=np.float32)
    return torch.from_numpy(values).cuda()


def validate(model, features, indices, labels):
    model.eval()
    total_loss = correct = 0
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            ix = indices[start:start+256]
            logits = model(tensor_batch(features, ix))
            targets = torch.from_numpy(labels[ix]).cuda()
            total_loss += nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            correct += (logits.argmax(1) == targets).sum().item()
    return total_loss / len(indices), correct / len(indices)


def main(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = RESULTS / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if not json.loads((ROOT/'cache'/'features_progress.json').read_text()).get('completed'):
        raise RuntimeError('Frozen convolutional cache is not complete')
    if (folder / 'embedding.npy').exists() and (folder / 'representation_complete.json').exists():
        print('Representation already complete for seed', seed, flush=True)
        return
    rows = json.loads((ROOT / 'data' / 'manifest.json').read_text())
    features = np.load(ROOT / 'cache' / 'vgg_conv.npy', mmap_mode='r')
    labels = np.array([r['label'] for r in rows], dtype=np.int64)
    train_ix = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val_ix = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    model = Representation().cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=60, eta_min=1e-6)
    rng = np.random.default_rng(seed)
    best = float('inf')
    stale = 0
    logs = []
    start = time.perf_counter()
    for epoch in range(1, 61):
        model.train()
        order = rng.permutation(train_ix)
        loss_total = correct = 0
        for offset in range(0, len(order), 64):
            ix = order[offset:offset+64]
            x = tensor_batch(features, ix)
            y = torch.from_numpy(labels[ix]).cuda()
            optimiser.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimiser.step()
            loss_total += loss.item() * len(ix)
            correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_accuracy = validate(model, features, val_ix, labels)
        log = dict(epoch=epoch, train_loss=loss_total/len(train_ix),
                   train_accuracy_dropout_enabled=correct/len(train_ix),
                   validation_loss=val_loss, validation_accuracy=val_accuracy,
                   learning_rate=optimiser.param_groups[0]['lr'], elapsed_s=time.perf_counter()-start)
        logs.append(log)
        (folder/'training_log.json').write_text(json.dumps(logs, indent=2))
        print(f'Seed {seed} epoch {epoch}: val loss {val_loss:.4f}, val accuracy {val_accuracy:.4f}; {log["elapsed_s"]:.1f}s', flush=True)
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save(dict(model_state=model.state_dict(), epoch=epoch, seed=seed,
                            validation_loss=val_loss), folder/'representation.pt')
        else:
            stale += 1
        scheduler.step()
        if stale >= 8:
            print('Validation-loss early stopping', flush=True)
            break
    saved = torch.load(folder/'representation.pt', weights_only=True)
    model.load_state_dict(saved['model_state'])
    model.eval()
    embeddings = np.lib.format.open_memmap(folder/'embedding.npy', mode='w+',
                                           dtype='float32', shape=(len(rows), 64))
    zero_count = 0
    with torch.inference_mode():
        for offset in range(0, len(rows), 256):
            ix = np.arange(offset, min(len(rows), offset+256))
            h = model.feature_layers(tensor_batch(features, ix))
            norm = torch.linalg.vector_norm(h, dim=1, keepdim=True)
            zero_count += (norm == 0).sum().item()
            h = h / norm.clamp_min(1e-12)
            embeddings[ix] = h.cpu().numpy()
    embeddings.flush()
    (folder/'representation_complete.json').write_text(json.dumps(dict(seed=seed,
        selected_epoch=saved['epoch'], minimum_validation_loss=best,
        epochs_executed=len(logs), elapsed_s=time.perf_counter()-start,
        zero_embeddings=zero_count, feature_parameters=sum(p.numel() for p in model.feature_layers.parameters()),
        selection='validation cross entropy only', test_metrics_accessed=False), indent=2))
    print('Saved normalised embeddings for seed', seed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    main(parser.parse_args().seed)
