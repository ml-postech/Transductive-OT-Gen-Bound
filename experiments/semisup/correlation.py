"""Table 3 (semi-supervised columns): gap, PAC bound, OT-oracle and OT bound of a trained model.
Evaluates one checkpoint on the transductive pool and appends a row to --result_csv;
the rank correlations of Table 3 are taken over the rows of a sweep.  See README.md
for the training commands.
"""

import argparse
import os
import sys

import bound_utils as bu

sys.path.insert(0, os.path.join(bu.HERE, '..'))
from _common import append_csv_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', required=True, choices=['lp', 'mt', 'fixmatch', 'simmatch'])
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'svhn'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--e_depth', type=int, required=True, help='CNN blocks (lp, mt) or WRN depth')
    parser.add_argument('--width', type=int, required=True, help='CNN base channels (lp, mt) or WRN width')
    parser.add_argument('--c_depth', type=int, default=1, help='classifier depth')
    parser.add_argument('--n_permutations', type=int, default=4, help="number of pi' samples (App. C)")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--result_csv', default=None)
    args = parser.parse_args()

    # Mean Teacher, FixMatch and SimMatch are evaluated with their EMA model
    use_ema = args.algo != 'lp'
    if args.algo in ('lp', 'mt'):
        model = bu.load_cnn(args.checkpoint, args.e_depth, args.width, args.c_depth, use_ema)
        mean, std = bu.NORM['lp'][args.dataset]
    else:
        model = bu.load_wrn(args.checkpoint, args.e_depth, args.width, args.c_depth)
        mean, std = bu.NORM['fixmatch'][args.dataset]

    data_dir = os.path.join(args.data_root, args.dataset, 'by-image')
    loader, labeled_mask = bu.build_pool(data_dir, 'all', mean, std, batch_size=256, workers=4,
                                         label_file=bu.label_file(args.dataset))
    features, logits, labels = bu.extract(model, loader, 'cuda')

    result = bu.ot_bounds(features, logits, labels, labeled_mask, n_permutations=args.n_permutations)
    result['pac_bound'] = bu.pac_bound(args.checkpoint, bu.NUM_LABELED[args.dataset], use_ema)
    print(f"gap {result['gen_gap']:.4f}  PAC {result['pac_bound']:.4f}  "
          f"OT-oracle {result['ot_oracle']:.4f}  OT bound {result['ot_bound']:.4f}")

    append_csv_row(args.result_csv, {'algo': args.algo, 'dataset': args.dataset, 'e_depth': args.e_depth,
                                     'width': args.width, 'c_depth': args.c_depth,
                                     'checkpoint': args.checkpoint, **result})


if __name__ == '__main__':
    main()
