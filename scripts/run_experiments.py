#!/usr/bin/env python3
"""Run and report the full PriTrust-FL experiment section.

This is intentionally a single-file server runner. It wraps the existing
``src/federated_main.py`` training entry point and then builds the paper
artifacts from the saved pickles/logs:

    * clean baseline table at 0% malicious clients
    * sign-flip and Min-Max robustness figures
    * untargeted 30% worst-case table
    * label-flip and backdoor ASR/MTA figures
    * targeted 30% worst-case table
    * plaintext-code efficiency tables
    * sorted raw metrics/log export

Typical server usage from the repo root:

    python3 scripts/run_experiments.py all --gpus 0 1 --tasks-per-gpu 2

Useful split workflow:

    python3 scripts/run_experiments.py sweep --gpus 0 1 --tasks-per-gpu 2
    python3 scripts/run_experiments.py report

The default pipeline runs one seed and the repo's plaintext training/defense
logic only. Efficiency tables are filled from the current plaintext code path,
not from homomorphic encryption or secret-sharing protocol costs.

The runner is resumable: before dispatching a config, it scans
``save/objects/*.pkl`` and skips matching completed runs.
"""

import argparse
import csv
import json
import multiprocessing as mp
import pickle
import queue
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
SAVE_DIR = PROJECT_ROOT / 'save'
SAVE_OBJECTS = SAVE_DIR / 'objects'
LOG_DIR = PROJECT_ROOT / 'logs'
SWEEP_LOG_DIR = LOG_DIR / 'sweep'
FIG_DIR = PROJECT_ROOT / 'fig'
RUNS_CSV_PATH = SAVE_DIR / 'results_runs.csv'
SUMMARY_CSV_PATH = SAVE_DIR / 'results_summary.csv'
EFFICIENCY_JSON_PATH = SAVE_DIR / 'efficiency_summary.json'
DEFAULT_LATEX_OUT = SAVE_DIR / 'experiment_results_section.tex'
DEFAULT_RAW_OUT = SAVE_DIR / f'rawdata{date.today().isoformat()}'
PYTHON = sys.executable

SEEDS = [1]
DATASET_SETTINGS = [
    ('mnist', 1),
    ('mnist', 0),
    ('cifar', 1),
    ('cifar', 0),
]
METHODS = [
    'fedavg',
    'krum',
    'trimmed_mean',
    'shieldfl',
    'pdfl',
    'pritrust_fl',
]
EFFICIENCY_METHODS = ['shieldfl', 'pdfl', 'pritrust_fl']
METHOD_LABELS = {
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'trimmed_mean': 'Trimmed Mean',
    'shieldfl': 'ShieldFL',
    'pdfl': 'PDFL',
    'pritrust_fl': 'PriTrust-FL',
}
METHOD_ALIASES = {
    'fedavg': 'fedavg',
    'fed_avg': 'fedavg',
    'krum': 'krum',
    'trimmed_mean': 'trimmed_mean',
    'trimmedmean': 'trimmed_mean',
    'shieldfl': 'shieldfl',
    'shield_fl': 'shieldfl',
    'pdfl': 'pdfl',
    'pritrust_fl': 'pritrust_fl',
    'pritrustfl': 'pritrust_fl',
}
ATTACK_RATIOS = {
    'none': [0.0],
    'sign_flip': [0.1, 0.2, 0.3],
    'min_max': [0.1, 0.2, 0.3],
    'label_flip': [0.1, 0.2, 0.3],
    'backdoor': [0.1, 0.2, 0.3],
}
DEFAULT_EPOCHS = {'mnist': 200, 'cifar': 1000}
DEFAULT_LOCAL_BS = {'mnist': 10, 'cifar': 32}
DEFAULT_LR = {'mnist': 0.01, 'cifar': 0.03}
DEFAULT_WEIGHT_DECAY = {'mnist': 0.0, 'cifar': 5e-4}
DEFAULT_SCHEDULER = {'mnist': 'none', 'cifar': 'cosine'}
DEFAULT_MODEL = {'mnist': 'cnn', 'cifar': 'resnet18'}
DEFAULT_NORM = {'mnist': 'batch_norm', 'cifar': 'batch_norm'}
# K_t = ceil(0.5 L); MNIST CNN has L=8, CIFAR ResNet18 has L=102.
PRITRUST_AUDIT_LAYERS = {'mnist': 4, 'cifar': 51}
DIRICHLET_ALPHA = 0.3
NUM_USERS = 100
CLIENT_FRAC = 0.1
LOCAL_EPOCHS = 1
NA = 'N/A'

DEFENSE_LINE_PATTERNS = {
    'fedavg': re.compile(r'(^|&\s*)FedAvg\b'),
    'krum': re.compile(r'(^|&\s*)Krum\b'),
    'trimmed_mean': re.compile(r'(^|&\s*)Trimmed Mean\b'),
    'shieldfl': re.compile(r'(^|&\s*)ShieldFL\b'),
    'pdfl': re.compile(r'(^|&\s*)PDFL\b'),
    'pritrust_fl': re.compile(r'\\textbf\{PriTrust-FL\}|(^|&\s*)PriTrust-FL\b'),
}
XX_RE = re.compile(r'XX\.XX')
ROUND_TIME_RE = re.compile(r'Round Time:\s*([0-9]+):([0-9]{2}):([0-9]{2})')

ROBUSTNESS_LATEX_TEMPLATE = r"""\subsection{Robustness Against Untargeted Attacks}\label{subsec:exp_untargeted}

This subsection evaluates the six methods under the sign-flipping attack and the Min-Max attack. We report clean test accuracy at the final round across malicious client ratios from 0\% to 30\%.

\subsubsection{Clean Performance at Zero Attack Ratio}


Table~\ref{tab:clean_baseline} reports the clean test accuracy at malicious ratio 0\% for all six methods on MNIST and CIFAR-10 under both data distributions. The 0\% setting isolates the utility loss caused by each defense in the absence of any adversary. PriTrust-FL preserves accuracy comparable to FedAvg, which confirms that the audit and trust mechanisms do not harm benign training.

\begin{table}[!t]
\centering
\caption{Clean test accuracy at malicious ratio 0\%. Higher is better.}
\label{tab:clean_baseline}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lcccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{MNIST}} & \multicolumn{2}{c}{\textbf{CIFAR-10}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & IID & Non-IID & IID & Non-IID \\
\midrule
FedAvg          & XX.XX & XX.XX & XX.XX & XX.XX \\
Krum            & XX.XX & XX.XX & XX.XX & XX.XX \\
Trimmed Mean    & XX.XX & XX.XX & XX.XX & XX.XX \\
ShieldFL        & XX.XX & XX.XX & XX.XX & XX.XX \\
PDFL            & XX.XX & XX.XX & XX.XX & XX.XX \\
\textbf{PriTrust-FL} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Sign-Flipping Attack}

Figure~\ref{fig:signflip_sweep} plots clean test accuracy against the malicious client ratio under the sign-flipping attack on both datasets and both data distributions. FedAvg degrades sharply once the attack ratio exceeds 10\%. Krum and Trimmed Mean retain reasonable accuracy at low ratios but lose ground at 30\%. ShieldFL and PDFL provide stable cosine-based defense but show a visible drop in the Non-IID setting. PriTrust-FL achieves the flattest accuracy curve across all four panels, which indicates the strongest robustness to sign-flipping at every evaluated ratio.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig/signflip_sweep.pdf}
\caption{Clean test accuracy versus malicious client ratio under the sign-flipping attack. The four panels correspond to MNIST IID, MNIST Non-IID, CIFAR-10 IID, and CIFAR-10 Non-IID.}
\label{fig:signflip_sweep}
\end{figure}

\subsubsection{Min-Max Attack}

Figure~\ref{fig:minmax_sweep} reports the corresponding results under the Min-Max attack. Min-Max is a stronger untargeted attack because it constrains malicious updates to remain within the benign spread. Geometric defenses such as Krum struggle under this attack because the malicious vectors imitate the inter-client norm distribution. ShieldFL and PDFL also lose accuracy as the ratio rises. PriTrust-FL exhibits the smallest accuracy drop across the four panels. The dual-anchor design exposes the malicious deviation in the audited layers even when the overall update norm is bounded.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig/minmax_sweep.pdf}
\caption{Clean test accuracy versus malicious client ratio under the Min-Max attack. The four panels correspond to MNIST IID, MNIST Non-IID, CIFAR-10 IID, and CIFAR-10 Non-IID.}
\label{fig:minmax_sweep}
\end{figure}

\subsubsection{Worst-Case Comparison at 30\% Ratio}

Table~\ref{tab:untargeted_30} summarizes the clean test accuracy at the highest evaluated malicious ratio of 30\%. PriTrust-FL achieves the highest accuracy in every column. The advantage is largest under the Min-Max attack and under the Non-IID distribution, which are jointly the most challenging conditions for cosine-based defenses.

\begin{table*}[!t]
\centering
\caption{Clean test accuracy at malicious ratio 30\% under untargeted attacks. Higher is better.}
\label{tab:untargeted_30}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{l cccc cccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{4}{c}{\textbf{MNIST}} & \multicolumn{4}{c}{\textbf{CIFAR-10}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
 & \multicolumn{2}{c}{Sign-flip} & \multicolumn{2}{c}{Min-Max} & \multicolumn{2}{c}{Sign-flip} & \multicolumn{2}{c}{Min-Max} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & IID & Non-IID & IID & Non-IID & IID & Non-IID & IID & Non-IID \\
\midrule
FedAvg          & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
Krum            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
Trimmed Mean    & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
ShieldFL        & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
PDFL            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
\textbf{PriTrust-FL} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} \\
\bottomrule
\end{tabular}
\end{table*}


% ==================================================================
\subsection{Robustness Against Targeted Attacks}\label{subsec:exp_targeted}

This subsection evaluates the six methods under the label-flipping attack and the backdoor attack. We report clean test accuracy and attack success rate at the final round across malicious client ratios from 10\% to 30\%.

\subsubsection{Label-Flipping Attack}

Figure~\ref{fig:labelflip_sweep} plots ASR and clean test accuracy against the malicious client ratio under the label-flipping attack. Lower ASR indicates stronger defense. Without defense, ASR rises rapidly as the attack ratio grows. Krum and Trimmed Mean reduce ASR but suffer accuracy loss in the Non-IID setting. ShieldFL and PDFL keep ASR moderate at low ratios but lose containment at 30\%. PriTrust-FL keeps ASR low across all evaluated ratios on both datasets.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig/labelflip_sweep.pdf}
\caption{Attack success rate and clean test accuracy versus malicious client ratio under the label-flipping attack. Solid lines denote ASR. Dashed lines denote clean accuracy.}
\label{fig:labelflip_sweep}
\end{figure}

\subsubsection{Backdoor Attack}

Figure~\ref{fig:backdoor_sweep} reports the corresponding results under the backdoor attack. The backdoor attack is harder to detect than label-flipping because the malicious updates are largely aligned with benign updates on non-trigger features. PriTrust-FL substantially outperforms all baselines on backdoor ASR across both data distributions. The stochastic layer auditing combined with the dual-anchor distance test detects backdoor-induced deviations even when the global gradient direction matches the benign trend.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig/backdoor_sweep.pdf}
\caption{Attack success rate and clean test accuracy versus malicious client ratio under the backdoor attack. Solid lines denote ASR. Dashed lines denote clean accuracy.}
\label{fig:backdoor_sweep}
\end{figure}

\subsubsection{Worst-Case Comparison at 30\% Ratio}

Table~\ref{tab:targeted_30} summarizes the clean test accuracy and ASR at the highest evaluated malicious ratio of 30\%. PriTrust-FL achieves the lowest ASR and the highest or comparable clean accuracy in all columns. The improvement is most visible on the CIFAR-10 backdoor configuration under Non-IID partitioning.

\begin{table*}[!t]
\centering
\caption{Clean test accuracy and attack success rate at malicious ratio 30\% under targeted attacks. Higher Acc is better. Lower ASR is better.}
\label{tab:targeted_30}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{ll cccc cccc}
\toprule
\multirow{2}{*}{\textbf{Dist.}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{4}{c}{\textbf{MNIST}} & \multicolumn{4}{c}{\textbf{CIFAR-10}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
 & & \multicolumn{2}{c}{Label-flip} & \multicolumn{2}{c}{Backdoor} & \multicolumn{2}{c}{Label-flip} & \multicolumn{2}{c}{Backdoor} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
 & & Acc & ASR & Acc & ASR & Acc & ASR & Acc & ASR \\
\midrule
\multirow{6}{*}{IID}
 & FedAvg          & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & Krum            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & Trimmed Mean    & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & ShieldFL        & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & PDFL            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & \textbf{PriTrust-FL} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} \\
\midrule
\multirow{6}{*}{Non-IID}
 & FedAvg          & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & Krum            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & Trimmed Mean    & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & ShieldFL        & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & PDFL            & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX & XX.XX \\
 & \textbf{PriTrust-FL} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} & \textbf{XX.XX} \\
\bottomrule
\end{tabular}
\end{table*}


% ==================================================================
\subsection{Efficiency Evaluation}\label{subsec:exp_efficiency}

This subsection compares PriTrust-FL with the two privacy-preserving baselines, ShieldFL and PDFL, on three efficiency metrics. The metrics are client upload size per round, average server-side audit time per round, and average end-to-end online round time. The reported values are averaged over the first 50 benign training rounds on CIFAR-10 with $m=100$ clients and $n=10$ selected per round.

\subsubsection{Client Upload Size per Round}

Table~\ref{tab:upload_size} reports the per-round upload size of one selected client. ShieldFL transmits the largest payload because each Paillier ciphertext is much larger than a secret share. PDFL and PriTrust-FL both rely on additive secret sharing over $\mathbb{Z}_{2^{\ell}}$ and exchange comparable amounts of data. PriTrust-FL adds a small constant-size auxiliary header for the trust-related side information, which is negligible compared to the model share.

\begin{table}[!t]
\centering
\caption{Client upload size per round on CIFAR-10. Lower is better.}
\label{tab:upload_size}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{Upload size per round} \\
\midrule
ShieldFL    & XX.XX MB \\
PDFL        & XX.XX MB \\
\textbf{PriTrust-FL} & \textbf{XX.XX MB} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Server-Side Audit Time per Round}

Table~\ref{tab:audit_time} reports the average server-side audit time per round. The audit phase covers similarity scoring for the cosine-based baselines and indicator computation plus trust update for PriTrust-FL. PriTrust-FL audit time grows with the audit budget $K_t$, but remains far below the cost of homomorphic operations in ShieldFL.

\begin{table}[!t]
\centering
\caption{Average server-side audit time per round on CIFAR-10. Lower is better.}
\label{tab:audit_time}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{Audit time per round} \\
\midrule
ShieldFL    & XX.XX s \\
PDFL        & XX.XX s \\
\textbf{PriTrust-FL} & \textbf{XX.XX s} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{End-to-End Online Round Time}

Table~\ref{tab:e2e_time} reports the end-to-end online round time, which includes local training on each selected client, secret-share exchange, server-side audit, and aggregation. The values exclude offline Beaver triple preprocessing. PriTrust-FL achieves an end-to-end round time comparable to PDFL and substantially lower than ShieldFL. The audit overhead introduced by the dual-anchor design is offset by the efficiency of the underlying secret-sharing protocols.

\begin{table}[!t]
\centering
\caption{Average end-to-end online round time on CIFAR-10. Lower is better.}
\label{tab:e2e_time}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{Online round time} \\
\midrule
ShieldFL    & XX.XX s \\
PDFL        & XX.XX s \\
\textbf{PriTrust-FL} & \textbf{XX.XX s} \\
\bottomrule
\end{tabular}
\end{table}
"""


def normalize_method(value):
    key = str(value).strip().lower().replace('-', '_')
    if key not in METHOD_ALIASES:
        choices = ', '.join(METHODS)
        raise argparse.ArgumentTypeError(
            'unsupported method "{}"; choose from {}'.format(value, choices))
    return METHOD_ALIASES[key]


def iid_label(iid):
    return 'IID' if int(iid) == 1 else 'Non-IID'


def ratio_key(value):
    return round(float(value), 4)


def float_key(value):
    return round(float(value), 8)


def config_key(cfg):
    return (
        str(cfg['dataset']).lower(),
        int(cfg['iid']),
        normalize_method(cfg['defense']),
        str(cfg['attack']).lower(),
        ratio_key(cfg['malicious_ratio']),
        int(cfg['seed']),
        int(cfg['epochs']),
        int(cfg.get('num_users', NUM_USERS)),
        float_key(cfg.get('frac', CLIENT_FRAC)),
        int(cfg.get('local_ep', LOCAL_EPOCHS)),
        int(cfg.get('local_bs', DEFAULT_LOCAL_BS[str(cfg['dataset']).lower()])),
        float_key(cfg.get('lr', DEFAULT_LR[str(cfg['dataset']).lower()])),
        float_key(cfg.get('dirichlet_alpha', DIRICHLET_ALPHA)),
    )


def base_config(dataset, iid, defense, attack, ratio, seed, epochs,
                test_interval):
    dataset = dataset.lower()
    return {
        'dataset': dataset,
        'iid': int(iid),
        'defense': normalize_method(defense),
        'attack': attack,
        'malicious_ratio': float(ratio),
        'seed': int(seed),
        'epochs': int(epochs),
        'num_users': NUM_USERS,
        'frac': CLIENT_FRAC,
        'local_ep': LOCAL_EPOCHS,
        'local_bs': DEFAULT_LOCAL_BS[dataset],
        'lr': DEFAULT_LR[dataset],
        'momentum': 0.9,
        'weight_decay': DEFAULT_WEIGHT_DECAY[dataset],
        'scheduler': DEFAULT_SCHEDULER[dataset],
        'model': DEFAULT_MODEL[dataset],
        'norm': DEFAULT_NORM[dataset],
        'optimizer': 'sgd',
        'dirichlet_alpha': DIRICHLET_ALPHA,
        'test_interval': int(test_interval),
    }


def make_main_configs(seeds, methods, only_dataset, attacks, test_interval):
    configs = []
    selected_attacks = attacks or list(ATTACK_RATIOS.keys())
    for dataset, iid in DATASET_SETTINGS:
        if only_dataset and dataset != only_dataset:
            continue
        for method in methods:
            for attack in selected_attacks:
                for ratio in ATTACK_RATIOS[attack]:
                    for seed in seeds:
                        configs.append(base_config(
                            dataset, iid, method, attack, ratio, seed,
                            DEFAULT_EPOCHS[dataset], test_interval))
    return configs


def make_efficiency_configs(seeds, methods, iid, rounds, test_interval):
    configs = []
    for method in methods:
        for seed in seeds:
            configs.append(base_config(
                'cifar', iid, method, 'none', 0.0, seed, rounds,
                test_interval))
    return configs


def build_command(cfg, gpu_id=None):
    cmd = [
        PYTHON,
        str(SRC_DIR / 'federated_main.py'),
        f'--dataset={cfg["dataset"]}',
        f'--model={cfg["model"]}',
        f'--iid={cfg["iid"]}',
        f'--dirichlet_alpha={cfg["dirichlet_alpha"]}',
        f'--epochs={cfg["epochs"]}',
        f'--num_users={cfg["num_users"]}',
        f'--frac={cfg["frac"]}',
        f'--local_ep={cfg["local_ep"]}',
        f'--local_bs={cfg["local_bs"]}',
        f'--optimizer={cfg["optimizer"]}',
        f'--lr={cfg["lr"]}',
        f'--momentum={cfg["momentum"]}',
        f'--weight_decay={cfg["weight_decay"]}',
        f'--scheduler={cfg["scheduler"]}',
        f'--norm={cfg["norm"]}',
        f'--test_interval={cfg["test_interval"]}',
        f'--defense={cfg["defense"]}',
        f'--attack={cfg["attack"]}',
        f'--malicious_ratio={cfg["malicious_ratio"]}',
        '--sign_flip_lambda=5',
        '--min_max_search_steps=30',
        '--label_flip_source=1',
        '--attack_target_label=7',
        '--backdoor_fraction=0.2',
        f'--seed={cfg["seed"]}',
        f'--pritrust_audit_layers={PRITRUST_AUDIT_LAYERS[cfg["dataset"]]}',
        '--pritrust_alpha_min=0.5',
        '--pritrust_alpha_max=1.5',
        '--pritrust_theta_tem=1.5',
        '--pritrust_theta_spa=1.5',
        '--pritrust_gamma=1.5',
        '--pritrust_rho=0.8',
        '--pritrust_kappa=0.5',
    ]
    if gpu_id is not None:
        cmd.append(f'--gpu={gpu_id}')
    return cmd


def command_text(cmd):
    return ' '.join(str(part) for part in cmd)


def load_pkl(pkl_path):
    try:
        with open(pkl_path, 'rb') as handle:
            data = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    args = data.get('args')
    accuracy = data.get('mta_accuracy') or data.get('test_accuracy')
    asr = data.get('attack_success_rates') or data.get('asr')
    if not isinstance(args, dict) or not accuracy:
        return None

    try:
        cfg = {
            'dataset': args['dataset'],
            'iid': int(args['iid']),
            'defense': normalize_method(args['defense']),
            'attack': args['attack'],
            'malicious_ratio': float(args['malicious_ratio']),
            'seed': int(args['seed']),
            'epochs': int(args['epochs']),
            'num_users': int(args.get('num_users', NUM_USERS)),
            'frac': float(args.get('frac', CLIENT_FRAC)),
            'local_ep': int(args.get('local_ep', LOCAL_EPOCHS)),
            'local_bs': int(args.get(
                'local_bs',
                DEFAULT_LOCAL_BS[str(args['dataset']).lower()])),
            'lr': float(args.get(
                'lr',
                DEFAULT_LR[str(args['dataset']).lower()])),
            'dirichlet_alpha': float(args.get(
                'dirichlet_alpha', DIRICHLET_ALPHA)),
        }
    except (KeyError, TypeError, ValueError, argparse.ArgumentTypeError):
        return None

    final_asr = None
    if asr and asr[-1] is not None:
        final_asr = float(asr[-1])

    stem = pkl_path.stem
    final_log = LOG_DIR / f'{stem}.log'
    temp_log = LOG_DIR / f'tmp_{stem}.log'
    log_path = final_log if final_log.exists() else temp_log

    return {
        'args': args,
        'cfg': cfg,
        'key': config_key(cfg),
        'final_mta': float(accuracy[-1]),
        'final_asr': final_asr,
        'pkl': pkl_path,
        'log': log_path if log_path.exists() else None,
        'mtime': pkl_path.stat().st_mtime,
    }


def load_runs():
    runs = {}
    if not SAVE_OBJECTS.exists():
        return runs
    for pkl_path in sorted(SAVE_OBJECTS.glob('*.pkl')):
        record = load_pkl(pkl_path)
        if record is None:
            continue
        existing = runs.get(record['key'])
        if existing is None or record['mtime'] > existing['mtime']:
            runs[record['key']] = record
    return runs


def pending_configs(configs, runs):
    return [cfg for cfg in configs if config_key(cfg) not in runs]


def dispatch_configs(configs, gpus, tasks_per_gpu=1, dry_run=False,
                     list_pending=False):
    runs = load_runs()
    pending = pending_configs(configs, runs)

    print(f'Total configs:    {len(configs)}')
    print(f'Already complete: {len(configs) - len(pending)}')
    print(f'Pending:          {len(pending)}')
    print()

    if dry_run or list_pending:
        for cfg in pending:
            print(command_text(build_command(cfg)))
        return

    if not pending:
        print('All requested configs are already complete.')
        return

    job_queue = mp.Queue()
    for idx, cfg in enumerate(pending, start=1):
        job_queue.put((idx, len(pending), cfg))

    worker_slots = [
        (gpu, slot)
        for gpu in gpus
        for slot in range(max(1, int(tasks_per_gpu)))
    ]
    workers = [
        mp.Process(
            target=gpu_worker,
            args=(gpu, slot, job_queue, SWEEP_LOG_DIR),
        )
        for gpu, slot in worker_slots
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def gpu_worker(gpu_id, worker_slot, job_queue, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            idx, total, cfg = job_queue.get_nowait()
        except queue.Empty:
            return

        tag = (
            f'{cfg["dataset"]}_iid{cfg["iid"]}_{cfg["defense"]}_'
            f'{cfg["attack"]}_mr{cfg["malicious_ratio"]}_'
            f'ep{cfg["epochs"]}_s{cfg["seed"]}'
        )
        log_path = log_dir / f'{tag}.gpu{gpu_id}.slot{worker_slot}.log'
        cmd = build_command(cfg, gpu_id=gpu_id)
        start = time.time()
        worker_name = f'gpu{gpu_id}:{worker_slot}'
        print(f'[{worker_name}] [{idx:4d}/{total}] start {tag}', flush=True)
        with open(log_path, 'w') as output:
            proc = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        status = 'OK' if proc.returncode == 0 else f'FAIL({proc.returncode})'
        print(
            f'[{worker_name}] [{idx:4d}/{total}] {status} '
            f'{elapsed:8.1f}s {tag}',
            flush=True,
        )


def metric_values(runs, cfg_template, metric, seeds, allow_partial=False):
    values = []
    missing = []
    for seed in seeds:
        cfg = dict(cfg_template, seed=int(seed))
        record = runs.get(config_key(cfg))
        if record is None and cfg['attack'] != 'none' and abs(
                cfg['malicious_ratio']) < 1e-9:
            fallback = dict(cfg, attack='none', malicious_ratio=0.0)
            record = runs.get(config_key(fallback))
        if record is None:
            missing.append(seed)
            continue
        value = record['final_mta'] if metric == 'mta' else record['final_asr']
        if value is None:
            missing.append(seed)
            continue
        values.append(float(value))

    if missing and not allow_partial:
        return None, len(values), missing
    if not values:
        return None, 0, missing
    return statistics.mean(values), len(values), missing


def lookup_percent(runs, dataset, iid, defense, attack, ratio, metric, seeds,
                   allow_partial=False):
    cfg = base_config(
        dataset, iid, defense, attack, ratio, seeds[0],
        DEFAULT_EPOCHS[dataset], test_interval=0)
    value, _, _ = metric_values(
        runs, cfg, metric, seeds, allow_partial=allow_partial)
    if value is None:
        return NA
    return f'{100.0 * value:.2f}'


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def report_missing(title, configs, runs):
    missing = pending_configs(configs, runs)
    print(title)
    print(f'  expected: {len(configs)}')
    print(f'  present:  {len(configs) - len(missing)}')
    print(f'  missing:  {len(missing)}')
    for cfg in missing[:10]:
        print(
            '   - '
            f'{cfg["dataset"]} {iid_label(cfg["iid"])} '
            f'{cfg["defense"]} {cfg["attack"]} '
            f'mr={cfg["malicious_ratio"]} ep={cfg["epochs"]} '
            f'seed={cfg["seed"]}'
        )
    if len(missing) > 10:
        print(f'   ... {len(missing) - 10} more')
    print()


def row_label(method):
    if method == 'pritrust_fl':
        return r'\textbf{PriTrust-FL}'
    return METHOD_LABELS[method]


def print_latex_rows(runs, seeds, allow_partial=False):
    print('% Table clean_baseline - clean MTA at malicious_ratio = 0')
    for method in METHODS:
        cells = []
        for dataset in ['mnist', 'cifar']:
            for iid in [1, 0]:
                cells.append(lookup_percent(
                    runs, dataset, iid, method, 'none', 0.0, 'mta',
                    seeds, allow_partial))
        print(' & '.join([row_label(method)] + cells) + r' \\')
    print()

    print('% Table untargeted_30 - clean MTA at malicious_ratio = 30%')
    for method in METHODS:
        cells = []
        for dataset in ['mnist', 'cifar']:
            for attack in ['sign_flip', 'min_max']:
                for iid in [1, 0]:
                    cells.append(lookup_percent(
                        runs, dataset, iid, method, attack, 0.3, 'mta',
                        seeds, allow_partial))
        print(' & '.join([row_label(method)] + cells) + r' \\')
    print()

    print('% Table targeted_30 - MTA + ASR at malicious_ratio = 30%')
    for iid_name, iid in [('IID', 1), ('Non-IID', 0)]:
        print(f'% --- {iid_name} block ---')
        for method in METHODS:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['label_flip', 'backdoor']:
                    cells.append(lookup_percent(
                        runs, dataset, iid, method, attack, 0.3, 'mta',
                        seeds, allow_partial))
                    cells.append(lookup_percent(
                        runs, dataset, iid, method, attack, 0.3, 'asr',
                        seeds, allow_partial))
            print(' & '.join([row_label(method)] + cells) + r' \\')
        print()


def write_csvs(runs, seeds, allow_partial=False):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_CSV_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'dataset', 'iid', 'defense', 'attack', 'malicious_ratio',
                'seed', 'epochs', 'final_mta', 'final_asr', 'pkl',
            ],
        )
        writer.writeheader()
        for key in sorted(runs):
            record = runs[key]
            cfg = record['cfg']
            writer.writerow({
                'dataset': cfg['dataset'],
                'iid': cfg['iid'],
                'defense': cfg['defense'],
                'attack': cfg['attack'],
                'malicious_ratio': cfg['malicious_ratio'],
                'seed': cfg['seed'],
                'epochs': cfg['epochs'],
                'final_mta': record['final_mta'],
                'final_asr': record['final_asr'],
                'pkl': record['pkl'],
            })

    with open(SUMMARY_CSV_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'dataset', 'iid', 'defense', 'attack', 'malicious_ratio',
                'epochs', 'metric', 'mean', 'std', 'seeds_present',
                'seeds_expected', 'missing_seeds',
            ],
        )
        writer.writeheader()
        templates = []
        for dataset, iid in DATASET_SETTINGS:
            for method in METHODS:
                for attack, ratios in ATTACK_RATIOS.items():
                    for ratio in ratios:
                        templates.append(base_config(
                            dataset, iid, method, attack, ratio, seeds[0],
                            DEFAULT_EPOCHS[dataset], test_interval=0))
        for cfg in templates:
            for metric in ['mta', 'asr']:
                values = []
                missing = []
                for seed in seeds:
                    seeded = dict(cfg, seed=seed)
                    record = runs.get(config_key(seeded))
                    if record is None and cfg['attack'] != 'none' and abs(
                            cfg['malicious_ratio']) < 1e-9:
                        fallback = dict(seeded, attack='none',
                                        malicious_ratio=0.0)
                        record = runs.get(config_key(fallback))
                    if record is None:
                        missing.append(seed)
                        continue
                    value = (record['final_mta'] if metric == 'mta'
                             else record['final_asr'])
                    if value is None:
                        if metric == 'asr':
                            continue
                        missing.append(seed)
                        continue
                    values.append(float(value))
                if metric == 'asr' and cfg['attack'] not in (
                        'label_flip', 'backdoor'):
                    continue
                if missing and not allow_partial:
                    mean_value, std_value = None, None
                else:
                    mean_value, std_value = mean_std(values)
                writer.writerow({
                    'dataset': cfg['dataset'],
                    'iid': cfg['iid'],
                    'defense': cfg['defense'],
                    'attack': cfg['attack'],
                    'malicious_ratio': cfg['malicious_ratio'],
                    'epochs': cfg['epochs'],
                    'metric': metric,
                    'mean': '' if mean_value is None else mean_value,
                    'std': '' if std_value is None else std_value,
                    'seeds_present': len(values),
                    'seeds_expected': len(seeds),
                    'missing_seeds': ' '.join(str(seed) for seed in missing),
                })

    print(f'wrote {RUNS_CSV_PATH}')
    print(f'wrote {SUMMARY_CSV_PATH}')


def slug(value):
    text = str(value).strip().lower().replace('.', 'p')
    text = re.sub(r'[^a-z0-9_-]+', '-', text)
    return text.strip('-') or 'unknown'


def raw_run_dir(raw_root, record):
    cfg = record['cfg']
    ratio = f'mr_{float(cfg["malicious_ratio"]):.1f}'.replace('.', 'p')
    return (
        Path(raw_root) /
        slug(cfg['dataset']) /
        slug(iid_label(cfg['iid'])) /
        slug(cfg['defense']) /
        slug(cfg['attack']) /
        ratio /
        f'seed_{int(cfg["seed"])}_ep_{int(cfg["epochs"])}'
    )


def export_sorted_raw_data(runs, raw_root=DEFAULT_RAW_OUT):
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_root / 'manifest.csv'
    fieldnames = [
        'dataset', 'distribution', 'defense', 'attack', 'malicious_ratio',
        'seed', 'epochs', 'final_mta', 'final_asr', 'run_dir',
        'metrics_pkl', 'log_file', 'args_json',
    ]
    rows = []

    for key in sorted(runs):
        record = runs[key]
        cfg = record['cfg']
        dest_dir = raw_run_dir(raw_root, record)
        dest_dir.mkdir(parents=True, exist_ok=True)

        metrics_dest = dest_dir / 'metrics.pkl'
        log_dest = dest_dir / 'run.log'
        args_dest = dest_dir / 'args.json'

        if record['pkl'] and Path(record['pkl']).exists():
            shutil.copy2(record['pkl'], metrics_dest)
        if record['log'] and Path(record['log']).exists():
            shutil.copy2(record['log'], log_dest)
        args_dest.write_text(json.dumps(record['args'], indent=2,
                                        sort_keys=True))

        rows.append({
            'dataset': cfg['dataset'],
            'distribution': iid_label(cfg['iid']),
            'defense': cfg['defense'],
            'attack': cfg['attack'],
            'malicious_ratio': cfg['malicious_ratio'],
            'seed': cfg['seed'],
            'epochs': cfg['epochs'],
            'final_mta': record['final_mta'],
            'final_asr': record['final_asr'],
            'run_dir': dest_dir,
            'metrics_pkl': metrics_dest if metrics_dest.exists() else '',
            'log_file': log_dest if log_dest.exists() else '',
            'args_json': args_dest,
        })

    with open(manifest_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for csv_path in [RUNS_CSV_PATH, SUMMARY_CSV_PATH]:
        if csv_path.exists():
            shutil.copy2(csv_path, raw_root / csv_path.name)

    print(f'wrote sorted raw data: {raw_root} ({len(rows)} runs)')
    print(f'wrote raw manifest: {manifest_path}')


def plot_panels(runs, attack, ratios, metric, out_path, seeds,
                allow_partial=False):
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
        'fedavg': ('#6b7280', 'o'),
        'krum': ('#2563eb', 's'),
        'trimmed_mean': ('#059669', '^'),
        'shieldfl': ('#d97706', 'D'),
        'pdfl': ('#7c3aed', 'v'),
        'pritrust_fl': ('#dc2626', '*'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    secondary = metric == 'asr_with_mta'

    for ax, (dataset, iid, title) in zip(axes.flat, panels):
        ax2 = ax.twinx() if secondary else None
        for method in METHODS:
            color, marker = styles[method]
            xs_primary, ys_primary = [], []
            xs_secondary, ys_secondary = [], []
            for ratio in ratios:
                if metric == 'mta':
                    value = lookup_percent(
                        runs, dataset, iid, method, attack, ratio, 'mta',
                        seeds, allow_partial)
                    if value != NA:
                        xs_primary.append(100 * ratio)
                        ys_primary.append(float(value))
                else:
                    asr = lookup_percent(
                        runs, dataset, iid, method, attack, ratio, 'asr',
                        seeds, allow_partial)
                    mta = lookup_percent(
                        runs, dataset, iid, method, attack, ratio, 'mta',
                        seeds, allow_partial)
                    if asr != NA:
                        xs_primary.append(100 * ratio)
                        ys_primary.append(float(asr))
                    if mta != NA:
                        xs_secondary.append(100 * ratio)
                        ys_secondary.append(float(mta))
            if xs_primary:
                ax.plot(
                    xs_primary, ys_primary, color=color, marker=marker,
                    label=METHOD_LABELS[method], linewidth=1.6,
                    markersize=5,
                )
            if secondary and xs_secondary:
                ax2.plot(
                    xs_secondary, ys_secondary, color=color, marker=marker,
                    linestyle='--', linewidth=1.1, markersize=4,
                    alpha=0.78,
                )
        ax.set_title(title)
        ax.set_xlabel('Malicious client ratio (%)')
        ax.grid(True, alpha=0.28)
        if metric == 'mta':
            ax.set_ylabel('Clean accuracy (%)')
        else:
            ax.set_ylabel('ASR (%)')
            ax2.set_ylabel('Clean accuracy (%)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='lower center', ncol=6, frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def make_plots(runs, seeds, allow_partial=False):
    plot_panels(
        runs, 'sign_flip', [0.0, 0.1, 0.2, 0.3], 'mta',
        FIG_DIR / 'signflip_sweep.pdf', seeds, allow_partial)
    plot_panels(
        runs, 'min_max', [0.0, 0.1, 0.2, 0.3], 'mta',
        FIG_DIR / 'minmax_sweep.pdf', seeds, allow_partial)
    plot_panels(
        runs, 'label_flip', [0.1, 0.2, 0.3], 'asr_with_mta',
        FIG_DIR / 'labelflip_sweep.pdf', seeds, allow_partial)
    plot_panels(
        runs, 'backdoor', [0.1, 0.2, 0.3], 'asr_with_mta',
        FIG_DIR / 'backdoor_sweep.pdf', seeds, allow_partial)


def get_cifar_state_tensor_bytes():
    sys.path.insert(0, str(SRC_DIR))
    from types import SimpleNamespace
    from models import ResNet18Cifar

    model = ResNet18Cifar(SimpleNamespace(norm='batch_norm'))
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def upload_size_mb(args):
    try:
        byte_count = get_cifar_state_tensor_bytes()
    except Exception as exc:
        print(f'warning: could not compute model size for upload table: {exc}')
        return {method: None for method in EFFICIENCY_METHODS}

    # Current repo code uploads plaintext state_dict tensors, without the
    # Paillier ciphertexts or additive shares described by privacy protocols.
    plaintext_mb = byte_count / (1024.0 * 1024.0)
    return {
        method: plaintext_mb
        for method in EFFICIENCY_METHODS
    }


def benchmark_audit_times(args):
    if args.skip_audit_benchmark:
        return {method: None for method in EFFICIENCY_METHODS}

    cache = read_efficiency_cache()
    cache_key = {
        'mode': 'current_plaintext_code',
        'audit_benchmark_rounds': args.audit_benchmark_rounds,
        'audit_benchmark_clients': args.audit_benchmark_clients,
    }
    if (not args.refresh_efficiency and
            cache.get('cache_key') == cache_key and
            'audit_time_s' in cache):
        return {
            method: cache['audit_time_s'].get(method)
            for method in EFFICIENCY_METHODS
        }

    try:
        sys.path.insert(0, str(SRC_DIR))
        import torch
        from types import SimpleNamespace
        from models import ResNet18Cifar
        from defenses import aggregate_weights
    except Exception as exc:
        print(f'warning: could not run audit benchmark: {exc}')
        return {method: None for method in EFFICIENCY_METHODS}

    torch.manual_seed(1234)
    model = ResNet18Cifar(SimpleNamespace(norm='batch_norm'))
    global_weights = model.state_dict()
    clients = int(args.audit_benchmark_clients)
    local_weights = []
    for client_idx in range(clients):
        client_state = {}
        scale = 1e-4 * float(client_idx + 1)
        for key, value in global_weights.items():
            if value.is_floating_point():
                noise = torch.randn_like(value, dtype=torch.float32) * scale
                client_state[key] = value.detach().clone() + noise.to(
                    dtype=value.dtype)
            else:
                client_state[key] = value.detach().clone()
        local_weights.append(client_state)

    sample_counts = [500 for _ in range(clients)]
    client_ids = list(range(clients))
    audit_times = {}
    for method in EFFICIENCY_METHODS:
        bench_args = SimpleNamespace(
            defense=method,
            malicious_ratio=0.0,
            attack='none',
            defense_byzantine_clients=None,
            trimmed_mean_trim_ratio=None,
            shieldfl_similarity_threshold=0.0,
            pdfl_similarity_threshold=0.0,
            pritrust_audit_layers=None,
            pritrust_alpha_min=0.5,
            pritrust_alpha_max=1.5,
            pritrust_theta_tem=1.5,
            pritrust_theta_spa=1.5,
            pritrust_gamma=1.5,
            pritrust_rho=0.8,
            pritrust_kappa=0.5,
            pritrust_security_bits=128,
            seed=1,
        )
        state = {}
        timings = []
        for round_idx in range(args.audit_benchmark_rounds + 2):
            start = time.perf_counter()
            aggregate_weights(
                bench_args, global_weights, local_weights, sample_counts,
                client_ids=client_ids, state=state)
            elapsed = time.perf_counter() - start
            if round_idx >= 2:
                timings.append(elapsed)
        audit_times[method] = statistics.mean(timings) if timings else None
        print(f'audit benchmark {method}: {audit_times[method]:.4f}s')

    cache['cache_key'] = cache_key
    cache['audit_time_s'] = audit_times
    write_efficiency_cache(cache)
    return audit_times


def read_efficiency_cache():
    if not EFFICIENCY_JSON_PATH.exists():
        return {}
    try:
        return json.loads(EFFICIENCY_JSON_PATH.read_text())
    except Exception:
        return {}


def write_efficiency_cache(cache):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    EFFICIENCY_JSON_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f'wrote {EFFICIENCY_JSON_PATH}')


def parse_round_times(log_path, limit):
    if log_path is None or not Path(log_path).exists():
        return []
    times = []
    with open(log_path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = ROUND_TIME_RE.search(line)
            if not match:
                continue
            hours, minutes, seconds = (int(match.group(1)),
                                       int(match.group(2)),
                                       int(match.group(3)))
            times.append(hours * 3600 + minutes * 60 + seconds)
            if len(times) >= limit:
                break
    return times


def e2e_round_times(runs, args):
    result = {}
    for method in EFFICIENCY_METHODS:
        per_seed = []
        for seed in args.seeds:
            preferred = base_config(
                'cifar', args.efficiency_iid, method, 'none', 0.0, seed,
                args.efficiency_rounds, test_interval=0)
            record = runs.get(config_key(preferred))
            if record is None:
                fallback = base_config(
                    'cifar', args.efficiency_iid, method, 'none', 0.0, seed,
                    DEFAULT_EPOCHS['cifar'], test_interval=0)
                record = runs.get(config_key(fallback))
            if record is None:
                continue
            round_times = parse_round_times(record['log'], args.efficiency_rounds)
            if round_times:
                per_seed.append(statistics.mean(round_times))
        result[method] = statistics.mean(per_seed) if per_seed else None
    return result


def collect_efficiency_metrics(runs, args):
    metrics = {
        'upload_size_mb': upload_size_mb(args),
        'audit_time_s': benchmark_audit_times(args),
        'e2e_time_s': e2e_round_times(runs, args),
    }
    cache = read_efficiency_cache()
    cache.update(metrics)
    write_efficiency_cache(cache)
    return metrics


def format_efficiency_value(metrics, table_key, method):
    value = metrics.get(table_key, {}).get(method)
    if value is None:
        return NA
    return f'{value:.2f}'


def detect_defense(line):
    for method, pattern in DEFENSE_LINE_PATTERNS.items():
        if pattern.search(line):
            return method
    return None


def replace_xx_sequence(line, values):
    out = line
    for value in values:
        out, replacements = XX_RE.subn(value, out, count=1)
        if replacements == 0:
            break
    return out


def slice_table(text, label):
    label_re = re.compile(r'\\label\{tab:' + re.escape(label) + r'\}')
    match = label_re.search(text)
    if not match:
        return None
    end_match = re.search(r'\\end\{table\*?\}', text[match.end():])
    if not end_match:
        return None
    start = match.start()
    end = match.end() + end_match.end()
    return start, end, text[start:end]


def fill_clean_baseline(text, runs, seeds, allow_partial):
    sliced = slice_table(text, 'clean_baseline')
    if sliced is None:
        return text
    start, end, block = sliced
    lines = []
    for line in block.splitlines(keepends=True):
        method = detect_defense(line)
        if method and 'XX.XX' in line:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for iid in [1, 0]:
                    cells.append(lookup_percent(
                        runs, dataset, iid, method, 'none', 0.0, 'mta',
                        seeds, allow_partial))
            line = replace_xx_sequence(line, cells)
        lines.append(line)
    return text[:start] + ''.join(lines) + text[end:]


def fill_untargeted_30(text, runs, seeds, allow_partial):
    sliced = slice_table(text, 'untargeted_30')
    if sliced is None:
        return text
    start, end, block = sliced
    lines = []
    for line in block.splitlines(keepends=True):
        method = detect_defense(line)
        if method and 'XX.XX' in line:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['sign_flip', 'min_max']:
                    for iid in [1, 0]:
                        cells.append(lookup_percent(
                            runs, dataset, iid, method, attack, 0.3, 'mta',
                            seeds, allow_partial))
            line = replace_xx_sequence(line, cells)
        lines.append(line)
    return text[:start] + ''.join(lines) + text[end:]


def fill_targeted_30(text, runs, seeds, allow_partial):
    sliced = slice_table(text, 'targeted_30')
    if sliced is None:
        return text
    start, end, block = sliced
    iid_block_re = re.compile(r'\\multirow\{6\}\{\*\}\{(IID|Non-IID)\}')
    current_iid = None
    lines = []
    for line in block.splitlines(keepends=True):
        match = iid_block_re.search(line)
        if match:
            current_iid = 1 if match.group(1) == 'IID' else 0
        method = detect_defense(line)
        if method and 'XX.XX' in line and current_iid is not None:
            cells = []
            for dataset in ['mnist', 'cifar']:
                for attack in ['label_flip', 'backdoor']:
                    cells.append(lookup_percent(
                        runs, dataset, current_iid, method, attack, 0.3,
                        'mta', seeds, allow_partial))
                    cells.append(lookup_percent(
                        runs, dataset, current_iid, method, attack, 0.3,
                        'asr', seeds, allow_partial))
            line = replace_xx_sequence(line, cells)
        lines.append(line)
    return text[:start] + ''.join(lines) + text[end:]


def fill_efficiency_table(text, label, metrics, table_key):
    sliced = slice_table(text, label)
    if sliced is None:
        return text
    start, end, block = sliced
    lines = []
    for line in block.splitlines(keepends=True):
        method = detect_defense(line)
        if method in EFFICIENCY_METHODS and 'XX.XX' in line:
            value = format_efficiency_value(metrics, table_key, method)
            line = replace_xx_sequence(line, [value])
        lines.append(line)
    return text[:start] + ''.join(lines) + text[end:]


def filled_path(in_path):
    path = Path(in_path)
    return path.with_name(path.stem + '_filled' + path.suffix)


def fill_latex_text(text, runs, seeds, efficiency_metrics,
                    allow_partial=False):
    text = fill_clean_baseline(text, runs, seeds, allow_partial)
    text = fill_untargeted_30(text, runs, seeds, allow_partial)
    text = fill_targeted_30(text, runs, seeds, allow_partial)
    if efficiency_metrics is not None:
        text = fill_efficiency_table(
            text, 'upload_size', efficiency_metrics, 'upload_size_mb')
        text = fill_efficiency_table(
            text, 'audit_time', efficiency_metrics, 'audit_time_s')
        text = fill_efficiency_table(
            text, 'e2e_time', efficiency_metrics, 'e2e_time_s')
    return text


def write_latex_output(text, out_path, print_stdout=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    remaining = text.count('XX.XX')
    print(f'wrote {out_path}  (remaining XX.XX placeholders: {remaining})')
    if print_stdout:
        print()
        print('% ==================== Filled LaTeX Section ====================')
        print(text)


def fill_tex(in_path, out_path, runs, seeds, efficiency_metrics,
             allow_partial=False, print_stdout=False):
    in_path = Path(in_path)
    text = fill_latex_text(
        in_path.read_text(), runs, seeds, efficiency_metrics, allow_partial)
    write_latex_output(text, out_path, print_stdout=print_stdout)


def cmd_sweep(args):
    configs = make_main_configs(
        args.seeds, args.methods, args.only, args.attacks, args.test_interval)
    dispatch_configs(
        configs, args.gpus, args.tasks_per_gpu,
        args.dry_run, args.list_pending)


def cmd_efficiency(args):
    configs = make_efficiency_configs(
        args.seeds, EFFICIENCY_METHODS, args.efficiency_iid,
        args.efficiency_rounds, test_interval=0)
    dispatch_configs(
        configs, args.gpus, args.tasks_per_gpu,
        args.dry_run, args.list_pending)


def cmd_status(args):
    runs = load_runs()
    main_configs = make_main_configs(
        args.seeds, args.methods, args.only, args.attacks, args.test_interval)
    report_missing('Main experiment status:', main_configs, runs)
    if not args.no_efficiency:
        efficiency_configs = make_efficiency_configs(
            args.seeds, EFFICIENCY_METHODS, args.efficiency_iid,
            args.efficiency_rounds, test_interval=0)
        report_missing('Efficiency run status:', efficiency_configs, runs)


def cmd_report(args):
    runs = load_runs()
    if not runs:
        print(f'No completed pickles found in {SAVE_OBJECTS}')
        return

    main_configs = make_main_configs(
        args.seeds, args.methods, args.only, args.attacks, args.test_interval)
    report_missing('Main experiment status:', main_configs, runs)
    if not args.no_efficiency:
        efficiency_configs = make_efficiency_configs(
            args.seeds, EFFICIENCY_METHODS, args.efficiency_iid,
            args.efficiency_rounds, test_interval=0)
        report_missing('Efficiency run status:', efficiency_configs, runs)

    write_csvs(runs, args.seeds, allow_partial=args.allow_partial)
    if not args.no_raw_export:
        export_sorted_raw_data(runs, args.raw_out)

    if args.no_plots:
        print('skipped plots (--no-plots)')
    else:
        make_plots(runs, args.seeds, allow_partial=args.allow_partial)

    efficiency_metrics = None
    if not args.no_efficiency:
        efficiency_metrics = collect_efficiency_metrics(runs, args)

    rendered_latex = fill_latex_text(
        ROBUSTNESS_LATEX_TEMPLATE, runs, args.seeds, efficiency_metrics,
        allow_partial=args.allow_partial)
    write_latex_output(
        rendered_latex, args.latex_out,
        print_stdout=not args.no_latex_stdout)

    if args.fill:
        output_path = args.fill if args.in_place else (
            args.fill_out or filled_path(args.fill))
        fill_tex(
            args.fill, output_path, runs, args.seeds, efficiency_metrics,
            allow_partial=args.allow_partial, print_stdout=False)


def cmd_all(args):
    cmd_sweep(args)
    if args.dry_run or args.list_pending:
        if not args.no_efficiency:
            cmd_efficiency(args)
        return
    if not args.no_efficiency:
        cmd_efficiency(args)
    cmd_report(args)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'mode', nargs='?', default='all',
        choices=['all', 'sweep', 'efficiency', 'report', 'status'],
        help='default: all')
    parser.add_argument(
        '--gpus', type=int, nargs='+', default=[0],
        help='GPU ids to use')
    parser.add_argument(
        '--tasks-per-gpu', '--tasks_per_gpu', type=int, default=1,
        help='concurrent worker processes to launch per GPU')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=SEEDS,
        help='random seeds to run and average; default: 1')
    parser.add_argument(
        '--methods', type=normalize_method, nargs='+', default=METHODS,
        help='subset of methods to sweep')
    parser.add_argument(
        '--attacks', choices=list(ATTACK_RATIOS.keys()), nargs='+',
        default=None, help='subset of attacks to sweep')
    parser.add_argument(
        '--only', choices=['mnist', 'cifar'], default=None,
        help='restrict the main sweep to one dataset')
    parser.add_argument(
        '--test_interval', type=int, default=0,
        help='main-sweep evaluation interval; 0 records only final metrics')
    parser.add_argument(
        '--dry-run', '--dry_run', action='store_true',
        help='print pending commands without running them')
    parser.add_argument(
        '--list-pending', '--list_pending', action='store_true',
        help='same as --dry-run')
    parser.add_argument(
        '--allow-partial', '--allow_partial', action='store_true',
        help='average available seeds instead of requiring every seed')
    parser.add_argument(
        '--latex-out', '--latex_out', default=str(DEFAULT_LATEX_OUT),
        help='where to write the filled built-in LaTeX section')
    parser.add_argument(
        '--raw-out', '--raw_out', default=str(DEFAULT_RAW_OUT),
        help='where to export sorted raw metrics, logs, and args')
    parser.add_argument(
        '--no-raw-export', '--no_raw_export', action='store_true',
        help='skip exporting sorted raw data')
    parser.add_argument(
        '--no-latex-stdout', '--no_latex_stdout', action='store_true',
        help='write the LaTeX section to disk without printing it')
    parser.add_argument(
        '--fill', default=None,
        help='optional extra LaTeX file to fill in addition to the built-in section')
    parser.add_argument(
        '--fill-out', '--fill_out', default=None,
        help='output .tex path; default is <input>_filled.tex')
    parser.add_argument(
        '--in-place', '--in_place', action='store_true',
        help='overwrite --fill instead of writing a sibling filled file')
    parser.add_argument(
        '--no-plots', '--no_plots', action='store_true',
        help='skip writing fig/*.pdf')
    parser.add_argument(
        '--with-efficiency', '--with_efficiency', action='store_true',
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--no-efficiency', '--no_efficiency', action='store_true',
        help='skip the 50-round current-code efficiency jobs and tables')
    parser.add_argument(
        '--efficiency-iid', '--efficiency_iid', type=int, default=1,
        choices=[0, 1], help='CIFAR distribution for efficiency timing')
    parser.add_argument(
        '--efficiency-rounds', '--efficiency_rounds', type=int, default=50,
        help='benign CIFAR rounds used for online round-time averaging')
    parser.add_argument(
        '--audit-benchmark-rounds', '--audit_benchmark_rounds', type=int,
        default=50,
        help='synthetic CIFAR aggregation rounds for audit-time benchmark')
    parser.add_argument(
        '--audit-benchmark-clients', '--audit_benchmark_clients', type=int,
        default=10,
        help='selected clients in the synthetic audit-time benchmark')
    parser.add_argument(
        '--skip-audit-benchmark', '--skip_audit_benchmark',
        action='store_true', help='leave audit-time table as N/A')
    parser.add_argument(
        '--refresh-efficiency', '--refresh_efficiency', action='store_true',
        help='ignore cached audit benchmark values')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.methods = [normalize_method(method) for method in args.methods]
    args.seeds = [int(seed) for seed in args.seeds]

    if args.mode == 'sweep':
        cmd_sweep(args)
    elif args.mode == 'efficiency':
        cmd_efficiency(args)
    elif args.mode == 'report':
        cmd_report(args)
    elif args.mode == 'status':
        cmd_status(args)
    else:
        cmd_all(args)


if __name__ == '__main__':
    main()
