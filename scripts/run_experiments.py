#!/usr/bin/env python3
"""End-to-end driver for the PriTrust-FL experiment section (1 seed,
plaintext, CIFAR Non-IID lr=0.03).

This single file does four things:
    1. sweep    -- runs every (dataset, iid, defense, attack, ratio) config
                   on the GPUs you specify, resumable
    2. aggregate-- scans save/objects/*.pkl and prints LaTeX rows for
                   Tables 1, 2, 3 to stdout, also writes a flat CSV
    3. plot     -- writes the four figure PDFs (signflip / minmax /
                   labelflip / backdoor sweeps)
    4. fill     -- replaces XX.XX placeholders inside an experiment-section
                   .tex file with the aggregated numbers

Defaults run all four steps in order. Run from the repo root.

Usage:
    # Full pipeline on two GPUs, also fill section.tex in-place:
    python scripts/run_experiments.py --gpus 0 1 --fill section.tex

    # Just dispatch the sweep (e.g. on a remote server):
    python scripts/run_experiments.py sweep --gpus 0 1

    # Reaggregate, replot and refill after the sweep finishes:
    python scripts/run_experiments.py report --fill section.tex

    # Preview the matrix without running anything:
    python scripts/run_experiments.py sweep --dry_run

The experiment matrix follows the latex section verbatim:
    seeds              = {1}
    settings           = MNIST IID, MNIST Non-IID, CIFAR IID, CIFAR Non-IID
    defenses           = fedavg, krum, trimmed_mean, shieldfl, pdfl, pritrust_fl
    attacks (ratios)   = none (0), sign_flip/min_max/label_flip/backdoor
                         at 0.1, 0.2, 0.3
    pritrust_fl hyper  = K_t = ceil(0.5 L); alpha=[0.5,1.5];
                         theta_tem=theta_spa=1.5; gamma=1.5; rho=0.8; kappa=0.5
"""

import argparse
import csv
import multiprocessing as mp
import pickle
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_OBJECTS = PROJECT_ROOT / 'save' / 'objects'
CSV_PATH = PROJECT_ROOT / 'save' / 'results_summary.csv'
FIG_DIR = PROJECT_ROOT / 'fig'
SWEEP_LOG_DIR = PROJECT_ROOT / 'logs' / 'sweep'
PYTHON = sys.executable

DEFENSES = ['fedavg', 'krum', 'trimmed_mean', 'shieldfl', 'pdfl', 'pritrust_fl']
DEFENSE_LABELS = {
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'trimmed_mean': 'Trimmed Mean',
    'shieldfl': 'ShieldFL',
    'pdfl': 'PDFL',
    'pritrust_fl': 'PriTrust-FL',
}
SETTINGS = [('mnist', 1), ('mnist', 0), ('cifar', 1), ('cifar', 0)]
ATTACKS = [
    ('none', [0.0]),
    ('sign_flip', [0.1, 0.2, 0.3]),
    ('min_max', [0.1, 0.2, 0.3]),
    ('label_flip', [0.1, 0.2, 0.3]),
    ('backdoor', [0.1, 0.2, 0.3]),
]
SEED = 1

# K_t = ceil(0.5 * L); MNIST CNN has L=8, CIFAR ResNet18 has L=102.
PRITRUST_AUDIT_LAYERS = {'mnist': 4, 'cifar': 51}
PRITRUST_COMMON = [
    '--pritrust_alpha_min=0.5',
    '--pritrust_alpha_max=1.5',
    '--pritrust_theta_tem=1.5',
    '--pritrust_theta_spa=1.5',
    '--pritrust_gamma=1.5',
    '--pritrust_rho=0.8',
    '--pritrust_kappa=0.5',
]

NA = 'N/A'


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------
def make_configs(only_dataset=None):
    configs = []
    for dataset, iid in SETTINGS:
        if only_dataset and dataset != only_dataset:
            continue
        for defense in DEFENSES:
            for attack, ratios in ATTACKS:
                for ratio in ratios:
                    configs.append({
                        'dataset': dataset,
                        'iid': iid,
                        'defense': defense,
                        'attack': attack,
                        'malicious_ratio': ratio,
                        'seed': SEED,
                    })
    return configs


def config_key(args_or_cfg):
    return (
        args_or_cfg['dataset'],
        int(args_or_cfg['iid']),
        args_or_cfg['defense'],
        args_or_cfg['attack'],
        round(float(args_or_cfg['malicious_ratio']), 4),
        int(args_or_cfg['seed']),
    )


def build_command(cfg, gpu_id=None):
    cmd = [
        PYTHON, str(PROJECT_ROOT / 'src' / 'federated_main.py'),
        f'--dataset={cfg["dataset"]}',
        f'--iid={cfg["iid"]}',
        f'--defense={cfg["defense"]}',
        f'--attack={cfg["attack"]}',
        f'--malicious_ratio={cfg["malicious_ratio"]}',
        f'--seed={cfg["seed"]}',
    ]
    if cfg['defense'] == 'pritrust_fl':
        cmd.append(
            f'--pritrust_audit_layers={PRITRUST_AUDIT_LAYERS[cfg["dataset"]]}')
        cmd.extend(PRITRUST_COMMON)
    if gpu_id is not None:
        cmd.append(f'--gpu={gpu_id}')
    return cmd


def scan_completed():
    completed = {}
    if not SAVE_OBJECTS.exists():
        return completed
    for pkl_path in SAVE_OBJECTS.glob('*.pkl'):
        record = _load_pkl(pkl_path)
        if record is None:
            continue
        key = config_key(record['args'])
        existing = completed.get(key)
        if existing is None or record['mtime'] > existing['mtime']:
            completed[key] = record
    return completed


def _load_pkl(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    args = data.get('args')
    accuracy = data.get('mta_accuracy') or data.get('test_accuracy')
    asr = data.get('attack_success_rates') or data.get('asr')
    if not args or not accuracy:
        return None
    try:
        _ = (args['dataset'], int(args['iid']), args['defense'],
             args['attack'], float(args['malicious_ratio']), int(args['seed']))
    except (KeyError, TypeError, ValueError):
        return None
    return {
        'args': args,
        'final_mta': float(accuracy[-1]),
        'final_asr': (None if not asr or asr[-1] is None
                      else float(asr[-1])),
        'pkl': pkl_path,
        'mtime': pkl_path.stat().st_mtime,
    }


def _gpu_worker(gpu_id, job_queue, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            idx, total, cfg = job_queue.get_nowait()
        except queue.Empty:
            return
        cmd = build_command(cfg, gpu_id=gpu_id)
        tag = (f'{cfg["dataset"]}_iid{cfg["iid"]}_{cfg["defense"]}_'
               f'{cfg["attack"]}_mr{cfg["malicious_ratio"]}_s{cfg["seed"]}')
        log_path = log_dir / f'{tag}.gpu{gpu_id}.log'
        start = time.time()
        print(f'[gpu{gpu_id}] [{idx:3d}/{total}] start: {tag}', flush=True)
        with open(log_path, 'w') as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        status = 'OK ' if proc.returncode == 0 else f'FAIL({proc.returncode})'
        print(f'[gpu{gpu_id}] [{idx:3d}/{total}] {status} {elapsed:6.1f}s : '
              f'{tag}', flush=True)


def run_sweep(gpus, only_dataset=None, dry_run=False, list_pending=False):
    configs = make_configs(only_dataset=only_dataset)
    completed = scan_completed()
    pending = [c for c in configs if config_key(c) not in completed]

    print(f'Total configs:    {len(configs)}')
    print(f'Already complete: {len(configs) - len(pending)}')
    print(f'Pending:          {len(pending)}')
    print()

    if dry_run or list_pending:
        for cfg in pending:
            print(' '.join(build_command(cfg)))
        return

    if not pending:
        print('All configs already complete; nothing to dispatch.')
        return

    job_queue = mp.Queue()
    for idx, cfg in enumerate(pending, start=1):
        job_queue.put((idx, len(pending), cfg))

    workers = [
        mp.Process(target=_gpu_worker, args=(gpu, job_queue, SWEEP_LOG_DIR))
        for gpu in gpus
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def load_runs():
    runs = {}
    if not SAVE_OBJECTS.exists():
        return runs
    for pkl_path in sorted(SAVE_OBJECTS.glob('*.pkl')):
        record = _load_pkl(pkl_path)
        if record is None:
            continue
        key = config_key(record['args'])
        existing = runs.get(key)
        if existing is None or record['mtime'] > existing['mtime']:
            runs[key] = record
    return runs


def lookup(runs, dataset, iid, defense, attack, ratio, metric='mta'):
    """Return formatted percentage string, or NA when missing."""
    key = (dataset, int(iid), defense, attack, round(float(ratio), 4), SEED)
    record = runs.get(key)
    if record is None:
        # 0% point can come from attack=none.
        if abs(ratio) < 1e-9:
            key = (dataset, int(iid), defense, 'none', 0.0, SEED)
            record = runs.get(key)
    if record is None:
        return NA
    value = record['final_mta'] if metric == 'mta' else record['final_asr']
    if value is None:
        return NA
    return '{:.2f}'.format(100 * value)


def write_csv(runs, csv_path=CSV_PATH):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'iid', 'defense', 'attack',
                  'malicious_ratio', 'seed', 'final_mta', 'final_asr']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(runs):
            r = runs[key]
            writer.writerow({
                'dataset': r['args']['dataset'],
                'iid': int(r['args']['iid']),
                'defense': r['args']['defense'],
                'attack': r['args']['attack'],
                'malicious_ratio': float(r['args']['malicious_ratio']),
                'seed': int(r['args']['seed']),
                'final_mta': r['final_mta'],
                'final_asr': r['final_asr'],
            })
    print(f'wrote {csv_path}  ({len(runs)} rows)')


def report_missing(runs):
    expected = make_configs()
    expected_keys = {config_key(c) for c in expected}
    present = expected_keys & set(runs.keys())
    missing = sorted(expected_keys - set(runs.keys()))
    print(f'expected configs: {len(expected_keys)}')
    print(f'present:          {len(present)}')
    print(f'missing:          {len(missing)}')
    if missing:
        print('first 10 missing:')
        for k in missing[:10]:
            print(' ', k)
    print()


def _row_label(defense):
    return ('\\textbf{PriTrust-FL}' if defense == 'pritrust_fl'
            else DEFENSE_LABELS[defense])


def print_table_clean(runs):
    print('% Table 1 (clean_baseline) - clean MTA at malicious_ratio = 0')
    for defense in DEFENSES:
        cells = []
        for dataset in ['mnist', 'cifar']:
            for iid in [1, 0]:
                cells.append(lookup(
                    runs, dataset, iid, defense, 'none', 0.0))
        line = ' & '.join([_row_label(defense)] + cells) + ' \\\\'
        print(line)
    print()


def print_table_untargeted_30(runs):
    print('% Table 2 (untargeted_30) - clean MTA at malicious_ratio = 0.3')
    for defense in DEFENSES:
        cells = []
        for dataset in ['mnist', 'cifar']:
            for attack in ['sign_flip', 'min_max']:
                for iid in [1, 0]:
                    cells.append(lookup(
                        runs, dataset, iid, defense, attack, 0.3))
        line = ' & '.join([_row_label(defense)] + cells) + ' \\\\'
        print(line)
    print()


def print_table_targeted_30(runs):
    print('% Table 3 (targeted_30) - MTA + ASR at malicious_ratio = 0.3')
    for iid_label, iid in [('IID', 1), ('Non-IID', 0)]:
        print(f'% --- {iid_label} block ---')
        for defense in DEFENSES:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['label_flip', 'backdoor']:
                    cells.append(lookup(
                        runs, dataset, iid, defense, attack, 0.3, 'mta'))
                    cells.append(lookup(
                        runs, dataset, iid, defense, attack, 0.3, 'asr'))
            line = ' & '.join([_row_label(defense)] + cells) + ' \\\\'
            print(line)
        print()


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def _plot_panels(runs, attack, ratios, metric, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        ('mnist', 1, 'MNIST IID'),
        ('mnist', 0, 'MNIST Non-IID'),
        ('cifar', 1, 'CIFAR-10 IID'),
        ('cifar', 0, 'CIFAR-10 Non-IID'),
    ]
    styles = {
        'fedavg':       ('#888888', 'o'),
        'krum':         ('#1f77b4', 's'),
        'trimmed_mean': ('#2ca02c', '^'),
        'shieldfl':     ('#ff7f0e', 'D'),
        'pdfl':         ('#9467bd', 'v'),
        'pritrust_fl':  ('#d62728', '*'),
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    secondary = (metric == 'asr_with_mta')
    for ax, (dataset, iid, title) in zip(axes.flat, panels):
        ax2 = ax.twinx() if secondary else None
        for defense in DEFENSES:
            color, marker = styles[defense]
            xs_a, ys_a, xs_b, ys_b = [], [], [], []
            for ratio in ratios:
                if metric == 'mta':
                    cell = lookup(runs, dataset, iid, defense, attack,
                                  ratio, 'mta')
                    if cell != NA:
                        xs_a.append(100 * ratio)
                        ys_a.append(float(cell))
                else:
                    asr_cell = lookup(runs, dataset, iid, defense, attack,
                                      ratio, 'asr')
                    mta_cell = lookup(runs, dataset, iid, defense, attack,
                                      ratio, 'mta')
                    if asr_cell != NA:
                        xs_a.append(100 * ratio)
                        ys_a.append(float(asr_cell))
                    if mta_cell != NA:
                        xs_b.append(100 * ratio)
                        ys_b.append(float(mta_cell))
            if xs_a:
                ax.plot(xs_a, ys_a, color=color, marker=marker,
                        label=DEFENSE_LABELS[defense],
                        linewidth=1.5, markersize=5)
            if secondary and xs_b:
                ax2.plot(xs_b, ys_b, color=color, marker=marker,
                         linestyle='--', linewidth=1.0, markersize=4,
                         alpha=0.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Malicious client ratio (%)')
        if metric == 'mta':
            ax.set_ylabel('MTA Acc (%)')
        else:
            ax.set_ylabel('ASR (%)')
            ax2.set_ylabel('MTA Acc (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=6,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'wrote {out_path}')


def make_plots(runs):
    _plot_panels(runs, 'sign_flip', [0.0, 0.1, 0.2, 0.3], 'mta',
                 FIG_DIR / 'signflip_sweep.pdf')
    _plot_panels(runs, 'min_max', [0.0, 0.1, 0.2, 0.3], 'mta',
                 FIG_DIR / 'minmax_sweep.pdf')
    _plot_panels(runs, 'label_flip', [0.1, 0.2, 0.3], 'asr_with_mta',
                 FIG_DIR / 'labelflip_sweep.pdf')
    _plot_panels(runs, 'backdoor', [0.1, 0.2, 0.3], 'asr_with_mta',
                 FIG_DIR / 'backdoor_sweep.pdf')


# ---------------------------------------------------------------------
# LaTeX placeholder filling
# ---------------------------------------------------------------------
DEFENSE_LINE_PATTERNS = {
    'fedavg':       re.compile(r'(^|&\s*)FedAvg\b'),
    'krum':         re.compile(r'(^|&\s*)Krum\b'),
    'trimmed_mean': re.compile(r'(^|&\s*)Trimmed Mean\b'),
    'shieldfl':     re.compile(r'(^|&\s*)ShieldFL\b'),
    'pdfl':         re.compile(r'(^|&\s*)PDFL\b'),
    'pritrust_fl':  re.compile(r'\\textbf\{PriTrust-FL\}'),
}
XX_RE = re.compile(r'XX\.XX')


def _replace_xx_sequence(line, values):
    out = line
    for v in values:
        out, n = XX_RE.subn(v, out, count=1)
        if n == 0:
            break
    return out


def _detect_defense(line):
    for defense, pattern in DEFENSE_LINE_PATTERNS.items():
        if pattern.search(line):
            return defense
    return None


def _slice_table(text, label):
    label_re = re.compile(r'\\label\{tab:' + re.escape(label) + r'\}')
    m = label_re.search(text)
    if not m:
        return None
    end_match = re.search(r'\\end\{table\*?\}', text[m.end():])
    if not end_match:
        return None
    start = m.start()
    end = m.end() + end_match.end()
    return start, end, text[start:end]


def _fill_clean_baseline(text, runs):
    sliced = _slice_table(text, 'clean_baseline')
    if sliced is None:
        return text
    start, end, block = sliced
    new_lines = []
    for line in block.splitlines(keepends=True):
        defense = _detect_defense(line)
        if defense and 'XX.XX' in line:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for iid in [1, 0]:
                    cells.append(lookup(
                        runs, dataset, iid, defense, 'none', 0.0))
            line = _replace_xx_sequence(line, cells)
        new_lines.append(line)
    return text[:start] + ''.join(new_lines) + text[end:]


def _fill_untargeted_30(text, runs):
    sliced = _slice_table(text, 'untargeted_30')
    if sliced is None:
        return text
    start, end, block = sliced
    new_lines = []
    for line in block.splitlines(keepends=True):
        defense = _detect_defense(line)
        if defense and 'XX.XX' in line:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['sign_flip', 'min_max']:
                    for iid in [1, 0]:
                        cells.append(lookup(
                            runs, dataset, iid, defense, attack, 0.3))
            line = _replace_xx_sequence(line, cells)
        new_lines.append(line)
    return text[:start] + ''.join(new_lines) + text[end:]


def _fill_targeted_30(text, runs):
    sliced = _slice_table(text, 'targeted_30')
    if sliced is None:
        return text
    start, end, block = sliced

    iid_block_re = re.compile(r'\\multirow\{6\}\{\*\}\{(IID|Non-IID)\}')
    new_lines = []
    current_iid = None
    for line in block.splitlines(keepends=True):
        m = iid_block_re.search(line)
        if m:
            current_iid = 1 if m.group(1) == 'IID' else 0
        defense = _detect_defense(line)
        if defense and 'XX.XX' in line and current_iid is not None:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['label_flip', 'backdoor']:
                    cells.append(lookup(
                        runs, dataset, current_iid, defense, attack,
                        0.3, 'mta'))
                    cells.append(lookup(
                        runs, dataset, current_iid, defense, attack,
                        0.3, 'asr'))
            line = _replace_xx_sequence(line, cells)
        new_lines.append(line)
    return text[:start] + ''.join(new_lines) + text[end:]


def fill_tex(in_path, out_path, runs):
    text = Path(in_path).read_text()
    text = _fill_clean_baseline(text, runs)
    text = _fill_untargeted_30(text, runs)
    text = _fill_targeted_30(text, runs)
    Path(out_path).write_text(text)
    remaining = text.count('XX.XX')
    print(f'wrote {out_path}  (remaining XX.XX placeholders: {remaining})')
    if remaining:
        print('  -> placeholders left in tables this script does not fill: '
              'tab:upload_size, tab:audit_time, tab:e2e_time '
              '(these need the privacy-preserving protocols).')


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def cmd_sweep(args):
    run_sweep(args.gpus, only_dataset=args.only,
              dry_run=args.dry_run, list_pending=args.list_pending)


def cmd_report(args):
    runs = load_runs()
    if not runs:
        print('No completed pickles found in', SAVE_OBJECTS)
        return
    report_missing(runs)
    print_table_clean(runs)
    print_table_untargeted_30(runs)
    print_table_targeted_30(runs)
    write_csv(runs)
    if not args.no_plots:
        make_plots(runs)
    if args.fill:
        out = args.fill_out or _filled_path(args.fill)
        fill_tex(args.fill, out, runs)


def cmd_all(args):
    cmd_sweep(args)
    if args.dry_run or args.list_pending:
        return
    cmd_report(args)


def _filled_path(in_path):
    p = Path(in_path)
    return p.with_name(p.stem + '_filled' + p.suffix)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['all', 'sweep', 'report'],
                        help='all: sweep then report (default); '
                        'sweep: only run pending experiments; '
                        'report: only aggregate, plot, and fill')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU ids to use during sweep (one worker per id)')
    parser.add_argument('--only', choices=['mnist', 'cifar'], default=None,
                        help='Restrict the sweep to one dataset')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print pending sweep commands and exit')
    parser.add_argument('--list_pending', action='store_true',
                        help='Same as --dry_run')
    parser.add_argument('--fill', default=None,
                        help='Path to your experiment-section .tex file. '
                        'A copy with XX.XX placeholders filled is written '
                        'next to it (default: <stem>_filled.tex).')
    parser.add_argument('--fill_out', default=None,
                        help='Override the output path for --fill')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip writing fig/*_sweep.pdf')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == 'sweep':
        cmd_sweep(args)
    elif args.mode == 'report':
        cmd_report(args)
    else:
        cmd_all(args)


if __name__ == '__main__':
    main()
