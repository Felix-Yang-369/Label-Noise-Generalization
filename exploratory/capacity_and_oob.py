"""Post-holdout exploratory analyses (not part of the frozen confirmatory matrix).

1. Random-forest out-of-bag (OOB) accuracy for the 15 main RF conditions.
   The forests are refitted with the frozen settings plus ``oob_score=True``;
   OOB scoring does not change tree construction, so each refit is checked
   against the recorded clean-test Macro-F1 before its OOB values are used.
2. The predeclared optional MLP capacity study (18 fits): a fixed 3,000-row
   stratified training subset, widths (32,), (128, 64), (512, 256), p in {0, 1},
   seeds 17/29/43, alpha=0 and a fixed 1,000-epoch budget.

Decisions use training and validation data only; no test score is reported for
the capacity study. Outputs go to results/exploratory/.

Usage (from the repository root):
    python -m exploratory.capacity_and_oob [--jobs 4]
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.core import prepare_frame, noise_labels, make_model, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'exploratory'


def load():
    protocol = json.loads((ROOT / 'config' / 'protocol.json').read_text(encoding='utf-8'))
    selected = json.loads((ROOT / 'config' / 'selected_params.json').read_text(encoding='utf-8'))
    frame = pd.read_excel(ROOT / 'data' / 'raw' / 'Dry_Bean_Dataset.xlsx')
    feats, classes = protocol['dataset']['features'], protocol['dataset']['classes']
    _, splits, _ = prepare_frame(frame, feats, classes, protocol['split']['first_random_state'],
                                 protocol['split']['second_random_state'])
    counts = {k: len(v) for k, v in splits.items()}
    if counts != protocol['split']['expected_counts']:
        raise ValueError(f'Unexpected split counts: {counts}')
    mapping = {c: i for i, c in enumerate(classes)}
    data = {k: (v[feats].to_numpy(dtype=float), v.Class.map(mapping).to_numpy(dtype=np.int64))
            for k, v in splits.items()}
    return protocol, selected, data


def rf_job(args):
    protocol, params, data, p, seed = args
    Xtr, ytr = data['train']
    Xte, yte = data['test']
    noisy = noise_labels(ytr, p, seed, 7)['noisy']
    pipe = make_model('RF', protocol['models']['RF'], params, seed)
    pipe.set_params(model__oob_score=True)
    with threadpool_limits(limits=1):
        pipe.fit(Xtr, noisy)
    rf = pipe.named_steps['model']
    oob_pred = rf.classes_[np.argmax(rf.oob_decision_function_, axis=1)]
    test_pred = pipe.predict(Xte)
    return {'model': 'RF', 'p': p, 'seed': seed,
            'train_acc_noisy': float(accuracy_score(noisy, pipe.predict(Xtr))),
            'oob_acc_noisy': float(accuracy_score(noisy, oob_pred)),
            'oob_acc_clean': float(accuracy_score(ytr, oob_pred)),
            'test_macro_f1_refit': float(f1_score(yte, test_pred, labels=list(range(7)),
                                                  average='macro', zero_division=0))}


def mlp_job(args):
    spec, Xs, ys, Xval, yval, width, p, seed = args
    noisy = noise_labels(ys, p, seed, 7)['noisy']
    fixed = dict(spec['fixed'])
    pipe = make_model('MLP', {'fixed': fixed}, {'hidden_layer_sizes': list(width)}, seed)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        pipe.fit(Xs, noisy)
    seconds = time.perf_counter() - start
    mlp = pipe.named_steps['model']
    pred_tr = pipe.predict(Xs)
    loss = list(map(float, mlp.loss_curve_))
    return {'width': 'x'.join(map(str, width)), 'p': p, 'seed': seed,
            'parameters': int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
            'train_acc_noisy': float(accuracy_score(noisy, pred_tr)),
            'train_acc_clean': float(accuracy_score(ys, pred_tr)),
            'validation_macro_f1': float(f1_score(yval, pipe.predict(Xval), labels=list(range(7)),
                                                  average='macro', zero_division=0)),
            'epochs': int(mlp.n_iter_), 'final_loss': loss[-1], 'fit_seconds': seconds,
            'warnings': len(caught), 'loss_curve': loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    protocol, selected, data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    recorded = pd.read_csv(ROOT / 'results' / 'test_metrics.csv')

    rf_params = selected['selected']['RF']['params']
    rf_args = [(protocol, rf_params, data, p, s) for s in protocol['noise']['seeds'] for p in protocol['noise']['p']]
    with ProcessPoolExecutor(args.jobs) as pool:
        rf_rows = list(pool.map(rf_job, rf_args))
    rf = pd.DataFrame(rf_rows)
    ref = recorded[recorded.model == 'RF'][['p', 'seed', 'test_macro_f1', 'train_acc_noisy']]
    rf = rf.merge(ref, on=['p', 'seed'], how='left')
    rf['refit_matches_record'] = np.isclose(rf.test_macro_f1_refit, rf.test_macro_f1, atol=1e-12, rtol=0)
    rf.to_csv(OUT / 'rf_oob.csv', index=False)
    print(rf.to_string(), flush=True)

    spec = protocol['optional_capacity']
    Xtr, ytr = data['train']
    idx = np.arange(len(ytr))
    sub, _ = train_test_split(idx, train_size=spec['subset_n'], stratify=ytr, random_state=spec['subset_seed'])
    sub = np.sort(sub)
    Xs, ys = Xtr[sub], ytr[sub]
    Xval, yval = data['validation']
    jobs = [(spec, Xs, ys, Xval, yval, tuple(w), p, s)
            for w in spec['hidden_layer_sizes'] for p in spec['p'] for s in spec['seeds']]
    jobs.sort(key=lambda j: -np.prod(j[5]))  # largest first for load balance
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(mlp_job, jobs))
    curves = {f"{r['width']}_p{int(r['p'] * 100):03d}_s{r['seed']}": r.pop('loss_curve') for r in rows}
    cap = pd.DataFrame(rows).sort_values(['width', 'p', 'seed'])
    cap.to_csv(OUT / 'mlp_capacity.csv', index=False)
    write_json(OUT / 'mlp_capacity_loss_curves.json', curves)
    summary = cap.groupby(['width', 'p']).agg(
        parameters=('parameters', 'first'),
        train_acc_noisy_mean=('train_acc_noisy', 'mean'), train_acc_noisy_std=('train_acc_noisy', 'std'),
        validation_macro_f1_mean=('validation_macro_f1', 'mean'),
        validation_macro_f1_std=('validation_macro_f1', 'std'),
        final_loss_mean=('final_loss', 'mean'), fit_seconds_mean=('fit_seconds', 'mean')).reset_index()
    summary.to_csv(OUT / 'mlp_capacity_summary.csv', index=False)
    write_json(OUT / 'subset_row_positions.json', {'subset_seed': spec['subset_seed'], 'positions': sub.tolist()})
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
