# Reproduction material: wavelet and Gaussian traffic-sign heads

This package supports **Bandwidth Matching and Coordinate Sensitivity of Wavelet and Gaussian Kernels for Traffic Sign Classification**. It implements the explicitly attributed VGG16 feature architecture of Sun and Chen (2021), DOI https://doi.org/10.4236/jamp.2021.912204, under the new protocol documented in the manuscript. It does not reuse that publication's reported performance values. 

## Included observations and scope

The main study has three representation seeds (42, 314, 2718), three nested per-class head budgets (100, 50, 25), and seven principal/control heads, for 63 full-test prediction files. Nine additional global-average-pooling (GAP) softmax runs supply the architecture control. Every run evaluates all 12,630 official test images, with 43 actual output scores per image. Metrics, confusion matrices, model states, validation grids, and training logs are supplied.

The extension adds 126 controlled head runs (three additional dataset/backbone cells, three seeds, two budgets, seven heads), 14 native ConvNeXt linear probes, and two full-label DINOv2 probes. Together the study supplies **214 complete-test prediction files**. BelgiumTSC has 2,520 unique official test image files and 62 prediction alternatives, of which 53 have positive test support. Its macro metrics use those 53 supported classes; predictions of any other class remain errors in accuracy. The feature architecture is a reference configuration, not a new hybrid invention. Head budgets are conditional on representation training with the full target training partition. No end-to-end few-shot, domain-transfer, TX2, detector, or fully fine-tuned modern state-of-the-art superiority experiment is claimed.

## Software and machine

Actual runs used Python 3.12.14, NVIDIA RTX 4050 Laptop GPU (6 GB), Intel Core i5-13500H, approximately 16 GB RAM, and Windows 11. The convolutional cache uses CUDA FP16 autocast and float16 storage; FC training uses float32; kernels and TWSVM solves use float64. A CUDA-capable GPU is required by the representation scripts. At least 10 GB of free disk space is advisable for data, dependencies, caches, and observations.

Create a Python 3.12 environment and install the CUDA 12.6 builds before the other requirements:

```text
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements-research.txt
```

On Linux, substitute `.venv/bin/python` for the Windows launcher. The explicit package versions are in `requirements-research.txt`; the additional actual environment snapshot is in `requirements-installed.txt`. CUDA, system BLAS, GPU, and dependency differences may change timings and floating-point results. Seeds do not promise bitwise equality across hardware.

## Data and split

Original GTSRB archive: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/ . The three archives used are `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, and `GTSRB_Final_Test_GT.zip`. Source URLs, byte counts, measured SHA-256/MD5, and available externally published MD5 values appear in `data/archive_checksums.json`. The measured training SHA-256 is a package integrity reference, not a checksum independently endorsed by the dataset distributor. Public images and original pretrained weights are not redistributed.

The final training archive has 39,209 images. Eight files in training sequence `14:00023` exactly match official test files; the entire 30-image training sequence is excluded. The remaining train/validation split keeps sequences together using seed 20261012: 31,349 training images and 7,830 validation images. The official 12,630-image test set is unchanged. Test physical-sign group identifiers are unavailable. Byte-level overlap is audited, not semantic near-duplicate independence.

`data/manifest.json` records label, relative path, ROI, split, training track ID, and SHA-256 for every retained image. `data/excluded_training_tracks.json` and `data/duplicate_review.json` document the exclusion. All labels in each prediction file are verified against the distributor's `GT-final_test.csv`.

## Reproduce from scratch

Run from the directory containing these scripts:

```text
.venv\Scripts\python reproduce.py
```

The sequence downloads and audits data, caches frozen VGG features, trains all three representations, selects head settings using validation only, locks settings, evaluates the three budgets, runs the GAP control, aggregates observations, and checks all 72 exported prediction populations. Complete existing caches/runs are skipped. To perform a fresh independent rerun, use a new extracted package directory without supplied `results/` or `cache/`; retain the supplied observations separately for comparison. Do not overwrite the reference package while comparing results.

Individual stages are also available:

```text
python prepare_data.py
python run_features.py
python train_representation.py --seed 42
python train_representation.py --seed 314
python train_representation.py --seed 2718
python evaluate_heads.py tune
python evaluate_heads.py evaluate --seed 42 --budget 100
python gap_control.py
python analyse_results.py
python verify_study.py
```

Repeat the evaluation command for all three seeds and budgets. `analyse_results.py` requires all 63 main runs and nine GAP runs. It computes sample SD across seeds, never substitutes missing measurements, and never pools repeated test images as independent populations. `verify_study.py` verifies predictions and accuracy against original official test labels and the locked grid checksum. `test_audit.py` and `test_kernel_heads.py` provide focused metric and solver checks.

## Reproduce the cross-dataset / ConvNeXt extension

After the original study's data and results have been prepared, run:

```text
python extension_download.py
python extension_prepare.py
python extension_pipeline.py
python dino_download.py
python dino_probe.py
python probe_grid_extension.py
python extension_snapshot_inputs.py
python extension_analyse.py
python extension_basis.py
python extension_plots.py
```

The official BelgiumTSC archives are obtained from https://btsd.ethz.ch/shareddata/ . Their measured SHA-256 values are in `data/belgium/archive_checksums.json`; they are integrity references, not distributor-published endorsements. Source tables have 4,591 training and 2,534 test rows, including 16 and 14 repeated identical filename annotations. The unique image files number 4,575 and 2,520. Duplicate annotation rows are recorded and removed; every official test image file is evaluated exactly once. Six class/filename-prefix group collisions lead to conservative exclusion of 17 training images. The retained split has 3,650 training and 908 validation images, with seed 20261014. Filename-prefix separation and decoded-pixel checks do not independently verify physical-sign identity.

`extension_protocol_initial.json` preserves the initial extension protocol. `extension_protocol_audit.json` records the identifier audit, `extension_protocol_pre_dedup.json` adds the crop audit, and `extension_protocol.json` adds annotation-row deduplication. GTSRB seed 42 completed adapter training under the initial version; later GTSRB seeds used the identifier-audit version. Amendments concern Belgium source-data integrity and supported-class metric handling. The first Belgium feature batch failed ROI bounds checking before any features were exported or adapters trained. A later VGG training/validation attempt was stopped by duplicate prediction-ID auditing before any valid test report, archived locally, excluded from final results, and rerun on the unique-file manifest. There are 2,805 invalid ROIs among 7,095 unique-image annotations. All Belgium images use the entire supplied classification crop; original ROI metadata is retained. The extension was planned after the original GTSRB/VGG16 observations, not before the whole study.

The additional cells are GTSRB/ConvNeXt-Tiny, BelgiumTSC/VGG16, and BelgiumTSC/ConvNeXt-Tiny. The ImageNet-1K V1 ConvNeXt checkpoint has SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. GTSRB ROI / Belgium full supplied crop and direct-resize preprocessing are shared within each dataset, replacing ConvNeXt's default centre-crop transform. Frozen 768-channel pooled/normalised features are cached in FP16; the supervised 768-to-64 ReLU/dropout adapter is trained in FP32 with the original Adam/schedule/validation-loss rules. Belgium VGG retains the original four FC layers. The backbones are never fine-tuned.

Head budgets are 100/25 for GTSRB and upper bounds 25/10 for Belgium. Rare Belgium classes use all available training examples without replacement. Every head in a cell/seed/budget uses identical samples. Each cell independently selects the same C/kernel grids on seed-42 validation data at its largest budget, then locks them. `extension_analyse.py` requires all 142 additional predictions, verifies original label tables, complete unique-file test populations, accuracies, hashes and configuration locks, and reports means/sample SD. The 36 new matched-kernel comparisons use exact within-seed McNemar tests with Holm correction as a separate additional comparison family. They do not pool repeated images or assert independence of physical signs.

Native ConvNeXt probes use signed, unit-normalised 768-D frozen features. Balanced probes use only sampled labels (three sample seeds); the adapted 64-D representation already used all training labels. Each all-training native probe uses all target training labels, has its own validation-only C selection, and is fitted once. The configurations are not a causal experiment on dimension alone. `extension_basis.py` uses fixed training-only pairs at sigma=2 to verify distance/permutation controls and the fourth-order correction bound; it does not select a rotation through test outcomes.

`extension_protocol_pre_solver.json` preserves the protocol before the numerical solver amendment. Original fits and extension validation eligibility still require optimal CVXOPT status; failed validation candidates are recorded as ineligible, not assigned scores. Only final locked extension fits permit L-BFGS-B on the identical box-constrained dual when CVXOPT fails. The objective, C, scale, epsilon and samples are unchanged. Accept fallback only with projected-gradient infinity residual <=1e-7 and primal--dual gap between -1e-7 and 1e-7*(1+abs(dual objective)). The amendment followed a later-seed fitting failure, after some earlier test results were known; it does not trigger retuning. Certificates are retained and independently checked by `extension_analyse.py`. `test_qp_fallback.py` checks a hand-solved convex problem and rejects a wrong candidate even if the optimizer reports success.

To rerun from scratch, copy only scripts, requirements, README, protocol versions and pinned source metadata/archive into a new empty study directory. Do not copy reference `results/`, `cache/` or `extension/` observations into that fresh run. To audit saved observations instead, combine all 17 Online Resource archives under the same `study/` directory. The locally archived diagnostic attempt preceding annotation deduplication is excluded from supplementary results and predictions.

`dino_protocol.json` fixes an additional self-supervised reference, specified while the extension was running. `dino_download.py` pins the official source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and model checkpoint hash `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. The download script obtains the pinned source and checkpoint from their official URLs. `dino_probe.py` uses the frozen ViT-S/14 distilled model without registers, with 384-D signed class-token features, FP16 extraction, unit normalisation, and a full-training-label multinomial probe. Its C grid is validation-only; each dataset has one selected test evaluation. xFormers is explicitly disabled. Its larger self-supervised pretraining data and objective differ from the supervised backbones, so this reference does not isolate architecture or claim universal superiority.

All six initial native-probe grids favoured C=10, their upper boundary. `probe_grid_extension_protocol.json` records an exploratory, validation-driven expansion to C={0.1,1,10,100,1000}, made after initial test results were already known. `probe_grid_extension.py` preserves initial locks and a local diagnostic copy of initial native results; separate expanded locks and their hashes identify final reported predictions. C is selected only by validation accuracy, with fixed tie-breaking and convergence eligibility. The original/extension adapted 64-D head configurations remain unchanged. Reported native-probe fit times are descriptive; some CPU fits overlapped independent DINO feature extraction or diagnostic analysis and are not exclusive benchmark timings. `extension_snapshot_inputs.py` records final delivery hashes rather than prospective registration.

## Original-study model selection and metrics

Only seed 42 and budget 100 determine principal head settings. The C grid is 0.1, 1, 10. Wavelet sigma is 1, 2, 4; Gaussian gamma is 0.09375, 0.375, 1.5. Settings maximise validation accuracy, ties following fixed grid order. Matched Gaussian controls inherit selected wavelet C and gamma = 3/(2 sigma**2). The GAP softmax has its own validation-only C selection. Other seeds and budgets retain the selected settings.

The exported scores are native softmax probabilities, conventional SVM OvR scores, or TWSVM vote scores with bounded distance correction. AP is non-interpolated and groups equal score thresholds. It is not VOC 11-point AP. TWSVM distances use Euclidean coefficient norm in the empirical kernel-coordinate space. Epsilon is 1e-6 and lambda is 0.01. CVXOPT requires optimal status; diagnostics are retained. Each seed uses 903 class pairs. Saved Python pickle classifiers should only be loaded from a trusted copy of this package.

Kernel construction time includes both full training and full test-reference matrices. Head prediction assumes precomputed matrices and excludes image preprocessing/feature extraction. Timings are descriptive single-pass workstation measurements; disk bundle sizes are not RAM peaks or full deployment sizes. Paired tests are exact McNemar tests within each seed, with a Bonferroni factor of 12 for four planned comparisons across three seeds. Unknown test grouping limits independence assumptions.

## Supplementary archives

Online Resource 1 contains the manifests and saved experiment summaries. Online Resources 2 through 9 contain the per-image prediction and score files. Online Resources 10 through 17 contain the trained model states, embeddings and classifier bundles. Extract all 17 archives into the same directory, preserving the shared study/ structure, when auditing the saved runs.

Run the scripts from the study/ directory. The dataset download and preparation scripts obtain and verify the public data. The DINOv2 download script retrieves the pinned upstream source and checkpoint.
