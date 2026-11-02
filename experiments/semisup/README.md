# Semi-supervised experiments (Tables 3 and 4)

LP-DeepSSL, Mean Teacher, FixMatch and SimMatch on CIFAR-10 and SVHN in the transductive
setting. The training code is adapted from the official implementations (`LP-DeepSSL/`,
`FixMatch/`, with their licenses); `--reg_weight` > 0 adds the OT regularizer R (`otbound.ssl`).
LP-DeepSSL needs `faiss-gpu`.

Commands are run from the repository root unless they `cd`; use `--dataset svhn` for SVHN.

```bash
python experiments/semisup/prepare_data.py --root data
```

## Table 3: bound vs. generalization gap

Train a model, then `correlation.py` appends its gap, PAC bound, OT-oracle and OT bound to the CSV.

```bash
# FixMatch (SimMatch: train_simmatch.py, --algo simmatch)
python experiments/semisup/FixMatch/train.py --dataset cifar10 --model-depth 28 --model-width 2
python experiments/semisup/correlation.py --algo fixmatch --dataset cifar10 --e_depth 28 --width 2 \
    --checkpoint results/semisup/fixmatch_cifar10_d28_w2_cd1/model_best.pth.tar --result_csv results/semisup/table3.csv

# LP-DeepSSL (Mean Teacher: train_stage1.py --isMT True only, --algo mt)
cd experiments/semisup/LP-DeepSSL
python train_stage1.py --dataset cifar10 --n-blocks 2 --base-channels 64 --exclude-unlabeled True
python train_stage2.py --dataset cifar10 --n-blocks 2 --base-channels 64
cd -
python experiments/semisup/correlation.py --algo lp --dataset cifar10 --e_depth 2 --width 64 \
    --checkpoint experiments/semisup/LP-DeepSSL/models/cifar10_1200_ss_split_10_isL2_1_nb2_bc64_cd1_k50_bs64/best.ckpt \
    --result_csv results/semisup/table3.csv
```

## Table 4: OT regularizer

Train with and without R (`--reg_weight 0`), then `accuracy.py` reports the mean ± std over runs
of the median accuracy over the last 5K iterations.

```bash
mkdir -p results/semisup/logs
python experiments/semisup/FixMatch/train.py --dataset cifar10 --reg_weight 1e-4 --seed 1 \
    2>&1 | tee results/semisup/logs/fixmatch_seed1.log
python experiments/semisup/accuracy.py results/semisup/logs/fixmatch_seed*.log
```

SimMatch (`train_simmatch.py`), Mean Teacher (`LP-DeepSSL/train_stage1.py --isMT True`) and
LP-DeepSSL (`train_stage1.py` then `train_stage2.py`) take the same `--reg_weight`.
