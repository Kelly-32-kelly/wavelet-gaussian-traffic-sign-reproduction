"""Training-only, fixed-scale diagnosis of basis-sensitive product corrections."""
import json
from pathlib import Path
import numpy as np
from extension_features import manifest_path

ROOT=Path(__file__).resolve().parent

def measured(a,b,sigma=2.):
    u=(a-b)**2/sigma**2
    assert np.max(u)<1
    S=u.sum(1);correction=(-np.log1p(-u)-u).sum(1);leading=.5*(u*u).sum(1)
    # Original non-negative domain has squared distances <=2; orthogonal changes preserve it.
    rho=2/sigma**2
    remainder_bound=(u**3).sum(1)/(3*(1-rho))
    assert np.max(leading-correction)<1e-12
    assert np.max(correction-leading-remainder_bound)<1e-12
    kw=np.exp(-1.5*S-correction);kr=np.exp(-1.5*S)
    return S,correction,leading,kw,kr

def main():
    records=[]
    for cell in ['gtsrb_vgg16','gtsrb_convnext_tiny','belgium_vgg16','belgium_convnext_tiny']:
        dataset=cell.split('_')[0];rows=json.loads(manifest_path(dataset).read_text());train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
        for seed in [42,314,2718]:
            path=(ROOT/'results'/f'seed_{seed}'/'embedding.npy') if cell=='gtsrb_vgg16' else (ROOT/'extension'/'results'/cell/f'seed_{seed}'/'embedding.npy')
            x=np.load(path).astype(np.float64);rng=np.random.default_rng(20261014)
            ia=rng.choice(train,size=4096,replace=True);ib=rng.choice(train,size=4096,replace=True)
            a,b=x[ia],x[ib]
            Q,R=np.linalg.qr(rng.standard_normal((64,64)));Q=Q*np.sign(np.diag(R))[None,:]
            permutation=rng.permutation(64)
            S,c,t,kw,kr=measured(a,b);Sr,cr,tr,kwr,krr=measured(a@Q,b@Q)
            Sp,cp,tp,kwp,krp=measured(a[:,permutation],b[:,permutation])
            assert np.max(np.abs(S-Sr))<1e-12 and np.max(np.abs(kr-krr))<1e-12
            assert np.max(np.abs(kw-kwp))<1e-12
            effective_original=np.divide(S*S,2*t,out=np.zeros_like(S),where=t>0)
            effective_rotated=np.divide(Sr*Sr,2*tr,out=np.zeros_like(Sr),where=tr>0)
            records.append(dict(cell=cell,seed=seed,pairs=4096,sigma=2.,source_split='train',
                mean_original_correction=float(c.mean()),mean_rotated_correction=float(cr.mean()),
                maximum_wavelet_rotation_difference=float(np.max(np.abs(kw-kwr))),
                mean_wavelet_rotation_difference=float(np.mean(np.abs(kw-kwr))),
                maximum_rbf_rotation_difference=float(np.max(np.abs(kr-krr))),
                maximum_wavelet_permutation_difference=float(np.max(np.abs(kw-kwp))),
                mean_fourth_order_approximation_error=float(np.mean(c-t)),
                mean_effective_coordinate_dimension_original=float(effective_original[t>0].mean()),
                mean_effective_coordinate_dimension_rotated=float(effective_rotated[tr>0].mean()),
                maximum_fourth_order_approximation_error=float(np.max(c-t)),
                pair_indices_a=ia.tolist(),pair_indices_b=ib.tolist(),orthogonal_matrix=Q.tolist(),coordinate_permutation=permutation.tolist()))
    folder=ROOT/'extension'/'analysis';folder.mkdir(exist_ok=True)
    (folder/'basis.json').write_text(json.dumps(records,indent=2))
    print('Verified basis diagnostics and fourth-order correction bounds on 12 training-only representation runs',flush=True)

if __name__=='__main__':main()
