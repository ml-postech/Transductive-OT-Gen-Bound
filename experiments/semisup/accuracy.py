"""Table 4 (semi-supervised columns): accuracy over seeds, from training logs.

Per run, accuracy is the median of the evaluations (on the unlabelled samples) in the
last 5,000 training iterations, following USB; the cell reports mean +- std over runs.

Usage: python experiments/semisup/accuracy.py logs/fixmatch_seed*.log
"""

import re
import sys

import numpy as np

WINDOW = 5000


def run_accuracy(log):
    text = open(log, errors='replace').read()
    if 'top-1 acc:' in text:  # FixMatch / SimMatch: one evaluation every eval_step iterations
        accs = [float(a) for a in re.findall(r'(?<!Best )(?<!Mean )top-1 acc: ([\d.]+)', text)]
        step = int(re.search(r"'eval_step': (\d+)", text).group(1))
        total = int(re.search(r"'total_steps': (\d+)", text).group(1))
    else:  # LP-DeepSSL / Mean Teacher: EMA model evaluated once per epoch (stage 2 for LP-DeepSSL)
        text = text.split('=== Stage 2 ===')[-1]
        accs = [float(a) for a in re.findall(r'Evaluating the EMA model:\s*\n\s*\* Prec@1\s+([\d.]+)', text)]
        step = int(re.search(r'Epoch: \[0\]\[0/(\d+)\]', text).group(1))
        total = len(accs) * step
    assert accs, f'no evaluations found in {log}'
    return float(np.median([a for i, a in enumerate(accs, 1) if i * step >= total - WINDOW]))


def main():
    accs = []
    for log in sys.argv[1:]:
        accs.append(run_accuracy(log))
        print(f'{accs[-1]:.2f}  {log}')
    print(f'{np.mean(accs):.2f} +- {np.std(accs):.2f}  (n={len(accs)})')  # population std


if __name__ == '__main__':
    main()
