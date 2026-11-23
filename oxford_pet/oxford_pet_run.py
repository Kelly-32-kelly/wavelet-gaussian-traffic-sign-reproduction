"""Run the locked Oxford-IIIT Pet comparison using the original study components."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from oxford_pet_prepare import digest, prepare, write_json


DATASET, BACKBONE = 'oxford_pet', 'convnext_tiny'
SEEDS = (42, 314, 2718)
BUDGET = 25
HEADS = ('softmax_none', 'svm_rbf', 'svm_wavelet', 'twsvm_rbf', 'twsvm_wavelet',
         'svm_rbf_matched', 'twsvm_rbf_matched')


def validate_inputs(root, protocol):
    data = root / 'data' / DATASET
    manifest = data / 'manifest.json'
    audit = json.loads((data / 'audit.json').read_text(encoding='utf-8'))
    if digest(manifest) != audit['manifest_sha256'] or digest(protocol) != audit['protocol_sha256']:
        raise ValueError('The audited split or protocol has changed')
    return manifest, audit


def weight_path(root, protocol):
    weights = root / 'torch_cache' / 'checkpoints' / 'convnext_tiny-983f1562.pth'
    expected = json.loads(protocol.read_text(encoding='utf-8'))['model']['weight_sha256']
    if digest(weights) != expected:
        raise ValueError('Frozen backbone weight checksum does not match the protocol')
    return weights


def smoke(root, protocol, manifest):
    import numpy as np
    import torch
    from torch import nn
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    from extension_features import Images
    from extension_train import Compact
    from kernel_heads import fit_softmax, fit_svm, fit_twsvm, kernel, scores

    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    weight_path(root, protocol)
    torch.hub.set_dir(str(root / 'torch_cache'))
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    selected = [row for label in range(3) for row in [r for r in rows if r['split'] == 'train' and r['label'] == label][:8]]
    images = Images(selected, root)
    batch = torch.stack([images[index] for index in range(len(images))]).cuda()
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval().cuda()
    model.requires_grad_(False)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        frozen = model.classifier[1](model.classifier[0](model.avgpool(model.features(batch))))
    labels = torch.tensor([row['label'] for row in selected], device='cuda')
    adapter = Compact(3).cuda()
    loss = nn.functional.cross_entropy(adapter(frozen.float()), labels)
    loss.backward()
    torch.optim.Adam(adapter.parameters(), lr=1e-4).step()
    adapter.eval()
    with torch.no_grad():
        features = nn.functional.normalize(adapter.feature_layers(frozen.float()), dim=1).cpu().numpy()
    assert frozen.shape == (24, 768) and features.shape == (24, 64) and np.isfinite(features).all()
    y = labels.cpu().numpy()
    fitted = fit_softmax(features, y, 1.)
    assert scores(fitted, features).shape == (24, 3)
    for kind, parameter in [('wavelet', 2.), ('rbf', .375)]:
        matrix = kernel(features, features, kind, parameter)
        assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
        for fit in (fit_svm, fit_twsvm):
            predictions = scores(fit(matrix, y, 1.), matrix)
            assert predictions.shape == (24, 3) and np.isfinite(predictions).all()
    write_json(root / 'smoke_check.json', dict(official_test_rows_used=0, training_rows=len(selected),
                                              manifest_sha256=digest(manifest), frozen_features=list(frozen.shape),
                                              torch=torch.__version__, cuda_device=torch.cuda.get_device_name(0)))
    print('Passed training-only smoke check from image decode through all three classifier implementations.', flush=True)


def basis_diagnostic(root, result, manifest):
    import numpy as np
    from extension_basis import measured

    rows = json.loads(manifest.read_text(encoding='utf-8'))
    train = np.array([index for index, row in enumerate(rows) if row['split'] == 'train'])
    records = []
    for seed in SEEDS:
        features = np.load(result / f'seed_{seed}' / 'embedding.npy').astype(np.float64)
        rng = np.random.default_rng(20261014)
        first = rng.choice(train, size=4096, replace=True)
        second = rng.choice(train, size=4096, replace=True)
        a, b = features[first], features[second]
        orthogonal, triangular = np.linalg.qr(rng.standard_normal((64, 64)))
        orthogonal *= np.sign(np.diag(triangular))[None, :]
        permutation = rng.permutation(64)
        base = measured(a, b)
        rotated = measured(a @ orthogonal, b @ orthogonal)
        permuted = measured(a[:, permutation], b[:, permutation])
        distance_change = float(np.max(np.abs(base[0] - rotated[0])))
        gaussian_change = float(np.max(np.abs(base[4] - rotated[4])))
        permutation_change = float(np.max(np.abs(base[3] - permuted[3])))
        assert max(distance_change, gaussian_change, permutation_change) < 1e-12
        wavelet_change = np.abs(base[3] - rotated[3])
        records.append(dict(seed=seed, source_split='train', sigma=2, pairs=len(first),
                            mean_correction_original=float(base[1].mean()),
                            mean_correction_rotated=float(rotated[1].mean()),
                            maximum_wavelet_rotation_difference=float(wavelet_change.max()),
                            mean_wavelet_rotation_difference=float(wavelet_change.mean()),
                            maximum_gaussian_rotation_difference=gaussian_change,
                            maximum_squared_distance_term_change=distance_change,
                            maximum_wavelet_permutation_difference=permutation_change,
                            first_indices=first.tolist(), second_indices=second.tolist(),
                            rotation=orthogonal.tolist(), permutation=permutation.tolist()))
    write_json(root / 'analysis' / 'basis.json', records)
    print(f'Verified the rotation and permutation controls in {len(records)} training representations.', flush=True)


def analyse(root, result, manifest, audit, protocol):
    import csv
    from statistics import mean, stdev
    from scipy.stats import beta, binomtest
    from audit_predictions import audit_rows

    expected = {row['sample_id']: row['label'] for row in json.loads(manifest.read_text(encoding='utf-8'))
                if row['split'] == 'test'}
    n = len(expected)
    pair_results = []
    runs = []
    for seed in SEEDS:
        predictions = {}
        for head in HEADS:
            folder = result / f'seed_{seed}' / f'budget_{BUDGET}'
            source = folder / f'{head}_predictions.csv'
            recorded = json.loads((folder / f'{head}_metrics.json').read_text(encoding='utf-8'))
            assert digest(source) == recorded['predictions_sha256']
            with source.open(encoding='ascii', newline='') as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == n and {row['sample_id']: int(row['y_true']) for row in rows} == expected
            report = audit_rows(rows, 37)
            assert report['accuracy'] == recorded['accuracy'] and report['macro'] == recorded['macro']
            predictions[head] = {row['sample_id']: row['y_pred'] == row['y_true'] for row in rows}
            runs.append(dict(seed=seed, head=head, accuracy=recorded['accuracy'], macro_f1=recorded['macro']['f1'],
                             macro_ap=recorded['macro']['average_precision'],
                             predictions=source.relative_to(root).as_posix(), sha256=digest(source)))
        for head in ('svm', 'twsvm'):
            wavelet = predictions[head + '_wavelet']
            gaussian = predictions[head + '_rbf_matched']
            wins = sum(wavelet[key] and not gaussian[key] for key in expected)
            losses = sum(gaussian[key] and not wavelet[key] for key in expected)
            p_value = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
            tail = .0125
            w_low = beta.ppf(tail, wins, n - wins + 1) if wins else 0.
            w_high = beta.ppf(1 - tail, wins + 1, n - wins) if wins < n else 1.
            l_low = beta.ppf(tail, losses, n - losses + 1) if losses else 0.
            l_high = beta.ppf(1 - tail, losses + 1, n - losses) if losses < n else 1.
            pair_results.append(dict(seed=seed, head=head, n=n, wins=wins, losses=losses,
                                     difference_pp=100 * (wins - losses) / n,
                                     lower_95_pp=100 * (w_low - l_high), upper_95_pp=100 * (w_high - l_low),
                                     mcnemar=float(p_value)))
    previous = 0.
    for index, row in enumerate(sorted(pair_results, key=lambda entry: entry['mcnemar'])):
        previous = min(1., max(previous, (len(pair_results) - index) * row['mcnemar']))
        row['holm_adjusted'] = previous
    summary = {}
    for head in HEADS:
        selected = [row for row in runs if row['head'] == head]
        summary[head] = {metric: dict(mean=mean(row[metric] for row in selected),
                                      sample_sd=stdev(row[metric] for row in selected))
                         for metric in ('accuracy', 'macro_f1', 'macro_ap')}
    directory = root / 'analysis'
    directory.mkdir(exist_ok=True)
    with (directory / 'paired_accuracy.csv').open('w', encoding='ascii', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_results[0]))
        writer.writeheader()
        writer.writerows(pair_results)
    write_json(directory / 'results.json', dict(dataset=DATASET, backbone=BACKBONE, n_test=n,
                                                splits=audit['retained_counts'], protocol_sha256=digest(protocol),
                                                paired=pair_results, summary=summary, all_runs=runs))
    print(f'Verified {len(runs)} prediction files and {len(pair_results)} matched comparisons.', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('prepare', 'smoke', 'features', 'train', 'tune', 'evaluate', 'analyse'))
    cli.add_argument('--root', type=Path, required=True)
    options = cli.parse_args()
    root = options.root.resolve()
    protocol = root / 'oxford_pet_protocol.json'
    if options.phase == 'prepare':
        if (root / 'test_evaluation_lock.json').exists():
            raise ValueError('Create a fresh root to regenerate the split after test evaluation has begun')
        prepare(root, protocol)
        return
    manifest, audit = validate_inputs(root, protocol)
    if options.phase == 'smoke':
        smoke(root, protocol, manifest)
        return
    from extension_features import main as extract
    from extension_train import main as train
    from extension_heads import evaluate, tune

    result = root / 'extension' / 'results' / f'{DATASET}_{BACKBONE}'
    if options.phase == 'features':
        smoke_check = json.loads((root / 'smoke_check.json').read_text(encoding='utf-8'))
        assert smoke_check['manifest_sha256'] == digest(manifest)
        weight_path(root, protocol)
        extract(DATASET, BACKBONE, root)
    elif options.phase == 'train':
        train(DATASET, BACKBONE, root, protocol)
        basis_diagnostic(root, result, manifest)
    elif options.phase == 'tune':
        tune(DATASET, BACKBONE, root, BUDGET, protocol)
    elif options.phase == 'evaluate':
        configs = result / 'selection' / 'locked_configs.json'
        locked = json.loads(configs.read_text(encoding='utf-8'))
        assert locked['protocol_sha256'] == digest(protocol) and not locked['test_metrics_used']
        assert locked['budget'] == BUDGET
        assert digest(configs.with_name('validation_grid.json')) == locked['validation_log_sha256']
        lock = {'config_sha256': digest(configs), 'manifest_sha256': digest(manifest),
                'protocol_sha256': digest(protocol), 'embeddings_sha256': {
                    str(seed): digest(result / f'seed_{seed}' / 'embedding.npy') for seed in SEEDS}}
        check = root / 'test_evaluation_lock.json'
        if check.exists():
            previous = json.loads(check.read_text(encoding='utf-8'))
            assert all(previous[key] == value for key, value in lock.items())
        else:
            write_json(check, dict(lock, locked_at=datetime.now(timezone.utc).isoformat()))
        evaluate(DATASET, BACKBONE, root, budgets=[BUDGET])
    else:
        analyse(root, result, manifest, audit, protocol)


if __name__ == '__main__':
    main()
