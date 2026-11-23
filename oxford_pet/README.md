# Oxford-IIIT Pet follow-up

This directory reproduces the 37-class pet-breed experiment for the bandwidth-matched wavelet and Gaussian comparison. The protocol was fixed before feature extraction, training and test prediction. The traffic-sign results were already known.

## Environment

Use Python 3.12, a CUDA-capable GPU and the package versions below. Run these commands in a fresh virtual environment from this directory.

```text
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 torchvision==0.23.0+cu126
python -m pip install numpy==2.3.5 Pillow==12.3.0 scipy==1.16.2 scikit-learn==1.7.2 cvxopt==1.3.2 threadpoolctl==3.6.0
```

## Run

```text
python fetch_inputs.py --root . annotations images weights
python oxford_pet_run.py prepare --root .
python oxford_pet_run.py smoke --root .
python oxford_pet_run.py features --root .
python oxford_pet_run.py train --root .
python oxford_pet_run.py tune --root .
python oxford_pet_run.py evaluate --root .
python oxford_pet_run.py analyse --root .
```

The setup verifies the public archives and upstream checkpoint and records the split audit. Validation selects the classifier settings before the official test predictions. The analysis checks the complete predictions and writes six paired accuracy differences and intervals to `analysis/paired_accuracy.csv`.

For a fresh run in a separate directory, copy `oxford_pet_protocol.json` there and use that directory as `--root`. The original images are available from [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/). The article's Online Resource 18 contains the archived splits, predictions, and fitted models.
