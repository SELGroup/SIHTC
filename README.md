# SIHTC: Hierarchical Text Classification Optimization via Structural Entropy and Singular Smoothing
This repository implements SIHTC, an optimized model via structural entropy and singular smoothing for hierarchical text classification.

## Preprocess
For details about data acquisition, processing, and baseline parameter settings, please refer to [HPT](https://github.com/wzh9969/HPT).

## Train
Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively (`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).
The training requires the modification of parameters based on the dataset. `--seloss-wight` is for the wight of structural entropy loss, and `--label-loss-wight`, `--hie-label-loss-wight` are for the wight of singular value
smoothing regularization loss.
We take the main results as the average of six random experiments.
### Elamples
```
python train.py --name test --batch 30 --data WebOfScience --seloss-wight 0.05 --label-loss-wight 0.05 --hie-label-loss-wight 0.05
python train.py --name test --batch 30 --data rcv1 --seloss-wight 0.1 --label-loss-wight 0.005 --hie-label-loss-wight 0.005
python train.py --name test --batch 30 --data NYT --seloss-wight 0.01 --label-loss-wight 0.005 --hie-label-loss-wight 0.005
```

## Test
Use `--extra _macro` or `--extra _micro` to choose from using `checkpoint_best_macro.pt` or `checkpoint_best_micro.pt` respectively.
### Elamples
```
python test.py --name WebOfScience-test
```
