"""Presentation charts in the deck's palette, built from recorded results only.

Usage (from the repository root):
    python reports/presentation/build_slide_charts.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INK, MUTED, GRID = '#10283f', '#5b6b7a', '#e3e8ee'
COLORS = {'LR': '#245b84', 'RF': '#ae721c', 'MLP': '#805a8d'}
NAMES = {'LR': 'Logistic regression', 'RF': 'Random forest', 'MLP': 'MLP'}

plt.rcParams.update({
    'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 15, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 1, 'axes.axisbelow': True, 'savefig.dpi': 200,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0.15, facecolor='white')
    plt.close(fig)


def macro_f1_curve(summary):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axvspan(0.70, 0.80, color='#f4ecdc', zorder=0)
    ax.text(0.75, 1.02, 'p = 0.75', ha='center', va='bottom', color=MUTED, fontsize=13)
    for model in ['LR', 'RF', 'MLP']:
        d = summary[summary.model == model].sort_values('p')
        ax.errorbar(d.p, d.test_macro_f1_mean, yerr=d.test_macro_f1_std, color=COLORS[model],
                    lw=3, marker='o', ms=7, capsize=3, zorder=3)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '1'])
    ax.set_xlabel('Probability p that a training label is randomized')
    ax.set_ylabel('Clean-test Macro-F1')
    ax.grid(axis='x', visible=False)
    save(fig, 'slide5_macro_f1.png')


def fit_vs_test(summary):
    d = summary[summary.p == 1.0].set_index('model').loc[['LR', 'RF', 'MLP']]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = range(3)
    w = 0.36
    for i, model in enumerate(d.index):
        fit, test = d.loc[model, 'train_acc_noisy_mean'], d.loc[model, 'test_accuracy_mean']
        ax.bar(i - w / 2, fit, w, color=COLORS[model])
        ax.bar(i + w / 2, test, w, color=COLORS[model], alpha=0.35)
        ax.text(i - w / 2, fit + 0.02, f'{fit:.0%}', ha='center', va='bottom', fontsize=17,
                fontweight='bold', color=COLORS[model])
        ax.text(i + w / 2, test + 0.02, f'{test:.0%}', ha='center', va='bottom', fontsize=15, color=MUTED)
    ax.axhline(1 / 7, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.98, 1 / 7 + 0.015, 'chance (1/7)', va='bottom', ha='right', fontsize=12, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([NAMES[m] for m in d.index], fontsize=15, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlim(-0.6, 3.0)
    ax.grid(axis='x', visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#555555', label='Fits the random training labels'),
                       Patch(color='#555555', alpha=0.35, label='Accuracy on clean test labels')],
              loc='upper right', frameon=False, fontsize=13)
    save(fig, 'slide6_fit_vs_test.png')


def demo_probabilities():
    classes = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
    clean = [33.4, 0.0, 0.0, 1.2, 0.5, 1.8, 63.2]
    random = [14.9, 14.8, 15.3, 13.9, 14.1, 12.6, 14.4]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, values, color, title in [(axes[0], clean, '#2e6b4f', 'Trained on clean labels'),
                                     (axes[1], random, '#b0603a', 'Trained on random labels')]:
        y = range(len(classes))[::-1]
        ax.barh(list(y), values, color=color, height=0.62)
        for yi, v in zip(y, values):
            ax.text(v + 1.5, yi, f'{v:.0f}%', va='center', fontsize=13, color=INK)
        ax.set_title(title, fontsize=16, color=INK, loc='left', pad=10, fontweight='bold')
        ax.set_xlim(0, 80)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(list(range(len(classes))[::-1]))
    axes[0].set_yticklabels(classes, fontsize=13, color=INK)
    axes[1].axvline(100 / 7, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    axes[1].text(100 / 7, -0.9, 'chance 1/7', fontsize=12, color=MUTED, ha='center', va='top')
    fig.subplots_adjust(wspace=0.08)
    save(fig, 'slide7_demo_probabilities.png')


def capacity():
    s = pd.read_csv(ROOT / 'results' / 'exploratory' / 'mlp_capacity_summary.csv')
    order = [('32', '32', 775), ('128x64', '128-64', 10887), ('512x256', '512-256', 141831)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (key, label, params) in enumerate(order):
        r = s[(s.width == key) & (s.p == 1)].iloc[0]
        ax.bar(i, r.train_acc_noisy_mean, 0.55, color=COLORS['MLP'], alpha=0.45 + 0.27 * i)
        ax.text(i, r.train_acc_noisy_mean + 0.03, f'{r.train_acc_noisy_mean:.0%}', ha='center',
                fontsize=20, fontweight='bold', color=COLORS['MLP'])
        ax.text(i, -0.04, f'{label}\n{params:,} params', ha='center', va='top', fontsize=13, color=INK)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylabel('Random training labels fitted')
    ax.grid(axis='x', visible=False)
    save(fig, 'slide_capacity.png')


def loss_curves():
    d = pd.read_csv(ROOT / 'paper' / 'generated' / 'mlp_loss_seed17.csv')
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    shades = {0.0: '#245b84', 0.25: '#4f7fa3', 0.5: '#8b7aa5', 0.75: '#a6789f', 1.0: '#805a8d'}
    for p, g in d.groupby('p'):
        ax.plot(g.epoch, g.training_objective, color=shades[p], lw=2.6)
        nudge = {1.0: 0.05, 0.75: -0.05}.get(p, 0)
        ax.text(g.epoch.iloc[-1] + 8, g.training_objective.iloc[-1] + nudge, f'p = {p:g}', va='center',
                fontsize=13, color=shades[p], fontweight='bold')
    ax.set_xlim(0, 560)
    ax.set_ylim(0, 2.1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss (seed 17)')
    ax.grid(axis='x', visible=False)
    save(fig, 'slideB3_loss.png')


def main():
    summary = pd.read_csv(ROOT / 'results' / 'model_noise_summary.csv')
    macro_f1_curve(summary)
    fit_vs_test(summary)
    demo_probabilities()
    capacity()
    loss_curves()


if __name__ == '__main__':
    main()
