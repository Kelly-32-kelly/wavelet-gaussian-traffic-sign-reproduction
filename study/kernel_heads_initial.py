"""Reference classifiers for the registered experiment, without test selection."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
from itertools import combinations

import numpy as np
import torch
from cvxopt import matrix, solvers
from scipy.linalg import cho_factor, cho_solve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

solvers.options.update(show_progress=False, abstol=1e-7, reltol=1e-7, feastol=1e-7, maxiters=100)


def kernel(x, y, kind, parameter):
    if kind not in {'wavelet', 'rbf'}:
        raise ValueError(kind)
    if parameter <= 0:
        raise ValueError('Kernel parameter must be positive')
    gpu = torch.cuda.is_available()
    device = 'cuda' if gpu else 'cpu'
    y_gpu = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=device)
    out = np.empty((len(x), len(y)), dtype=np.float64)
    with torch.inference_mode():
        for offset in range(0, len(x), 64):
            batch = torch.as_tensor(np.asarray(x[offset:offset+64]), dtype=torch.float64, device=device)
            delta2 = (batch[:, None, :] - y_gpu[None, :, :]).square()
            if kind == 'rbf':
                block = (-parameter * delta2.sum(-1)).exp()
            else:
                u = delta2 / parameter**2
                block = (1-u).prod(-1) * (-0.5*u.sum(-1)).exp()
            out[offset:offset+len(batch)] = block.cpu().numpy()
    return out


def _dual_solution(s, r, c, epsilon):
    h = s.T @ s + epsilon * np.eye(s.shape[1])
    factor = cho_factor(h, lower=True, check_finite=False)
    a = cho_solve(factor, r.T, check_finite=False)
    q = r @ a
    q = (q + q.T) * 0.5
    n = len(r)
    solution = solvers.qp(matrix(q), matrix(-np.ones(n)),
                         matrix(np.vstack([-np.eye(n), np.eye(n)])),
                         matrix(np.concatenate([np.zeros(n), np.full(n, c)])))
    if solution['status'] != 'optimal':
        raise RuntimeError('Non-optimal QP: ' + solution['status'])
    alpha = np.asarray(solution['x']).ravel()
    z = a @ alpha
    if not np.all(np.isfinite(z)) or np.linalg.norm(z[:-1]) == 0:
        raise RuntimeError('Invalid fitted kernel hyperplane')
    return z, dict(iterations=int(solution['iterations']),
                   primal_infeasibility=float(solution['primal infeasibility']),
                   dual_infeasibility=float(solution['dual infeasibility']),
                   gap=float(solution['gap']))


def fit_twsvm(train_kernel, labels, c, epsilon=1e-6):
    classes = np.unique(labels)
    pairs = []
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for count, (p, q) in enumerate(combinations(classes, 2), 1):
            ix1 = np.flatnonzero(labels == p)
            ix2 = np.flatnonzero(labels == q)
            reference = np.concatenate([ix1, ix2])
            s = np.column_stack([train_kernel[np.ix_(ix1, reference)], np.ones(len(ix1))])
            r = np.column_stack([train_kernel[np.ix_(ix2, reference)], np.ones(len(ix2))])
            z1, diagnostic1 = _dual_solution(s, r, c, epsilon)
            z2, diagnostic2 = _dual_solution(r, s, c, epsilon)
            z1 = -z1
            pairs.append(dict(p=int(p), q=int(q), reference=reference,
                              z1=z1, z2=z2, diagnostics=[diagnostic1, diagnostic2]))
            if count % 150 == 0:
                print(f'TWSVM fitted {count} pairs; {time.perf_counter()-start:.1f}s', flush=True)
    return dict(type='twsvm', classes=classes, pairs=pairs, c=c, epsilon=epsilon,
                fit_seconds=time.perf_counter()-start)


def twsvm_scores(model, evaluated_kernel, lam=0.01):
    classes = model['classes']
    votes = np.zeros((len(evaluated_kernel), len(classes)), dtype=np.float64)
    margins = np.zeros_like(votes)
    lookup = {int(c): i for i, c in enumerate(classes)}
    with threadpool_limits(limits=1):
        for pair in model['pairs']:
            k = evaluated_kernel[:, pair['reference']]
            z1, z2 = pair['z1'], pair['z2']
            dp = np.abs(k @ z1[:-1] + z1[-1]) / np.linalg.norm(z1[:-1])
            dq = np.abs(k @ z2[:-1] + z2[-1]) / np.linalg.norm(z2[:-1])
            p, q = lookup[pair['p']], lookup[pair['q']]
            choose_p = dp <= dq
            votes[:, p] += choose_p
            votes[:, q] += ~choose_p
            difference = np.divide(dq-dp, dq+dp, out=np.zeros_like(dp), where=(dq+dp)>0)
            margins[:, p] += difference
            margins[:, q] -= difference
    return votes + lam * margins / (len(classes)-1)


def fit_svm(train_kernel, labels, c):
    start = time.perf_counter()
    classifier = SVC(C=c, kernel='precomputed', decision_function_shape='ovr',
                     break_ties=True, tol=1e-6, cache_size=512)
    with threadpool_limits(limits=1):
        classifier.fit(train_kernel, labels)
    return dict(type='svm', estimator=classifier, classes=classifier.classes_,
                c=c, fit_seconds=time.perf_counter()-start)


def fit_softmax(features, labels, c):
    start = time.perf_counter()
    classifier = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, tol=1e-7)
    with threadpool_limits(limits=1):
        classifier.fit(features, labels)
    if np.any(classifier.n_iter_ >= 2000):
        raise RuntimeError('Softmax optimiser did not converge')
    return dict(type='softmax', estimator=classifier, classes=classifier.classes_,
                c=c, fit_seconds=time.perf_counter()-start)


def scores(model, inputs):
    if model['type'] == 'twsvm':
        return twsvm_scores(model, inputs)
    if model['type'] == 'softmax':
        return model['estimator'].predict_proba(inputs)
    return model['estimator'].decision_function(inputs)
