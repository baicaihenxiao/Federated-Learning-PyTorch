#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import hashlib
import logging
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid
from sampling import cifar_iid, cifar_noniid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FORMAT = (
    '%(asctime)s %(levelname)s %(filename)s-%(process)d-'
    '%(funcName)s:%(lineno)d %(message)s'
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_path():
    log_path = PROJECT_ROOT / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def _new_log_formatter():
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def _remove_file_handlers(log_file_path=None):
    root_logger = logging.getLogger()
    target_path = None
    if log_file_path is not None:
        target_path = Path(log_file_path).expanduser().resolve()

    for handler in list(root_logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if target_path is not None:
            handler_path = Path(handler.baseFilename).expanduser().resolve()
            if handler_path != target_path:
                continue
        handler.flush()
        root_logger.removeHandler(handler)
        handler.close()


def set_log_file(log_file_path, mode='a+'):
    """Route root logging to a specific file, replacing older file handlers."""
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = _new_log_formatter()

    _remove_file_handlers()

    has_console_handler = False
    for handler in root_logger.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                not isinstance(handler, logging.FileHandler)):
            has_console_handler = True
        handler.setFormatter(formatter)

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename=str(log_file_path),
                                       encoding='utf-8', mode=mode)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_file_path


def promote_log_file(temp_log_path, final_log_path):
    """Rename an active temporary log file after a successful run."""
    temp_log_path = Path(temp_log_path)
    final_log_path = Path(final_log_path)

    _remove_file_handlers(temp_log_path)
    final_log_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_log_path.exists():
        temp_log_path.replace(final_log_path)

    return set_log_file(final_log_path, mode='a+')


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    if not logging.getLogger().handlers:
        log_file_path = Path(get_log_path()) / (
            '%s-logfile.log' % datetime.now().strftime("%Y-%m-%d"))
        set_log_file(log_file_path)
    # format = '%(asctime)s %(levelname)s %(name)s %(module)s-%(funcName)s:%(lineno)d %(process)d-%(threadName)s msg = %(message)s')
    return logger


def format_run_time(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)


LOGGER = get_logger(__name__)


IMPORTANT_ARG_KEYS = [
    'dataset',
    'model',
    'iid',
    'dirichlet_alpha',
    'epochs',
    'num_users',
    'frac',
    'local_ep',
    'local_bs',
    'batch_size',
    'optimizer',
    'lr',
    'momentum',
    'weight_decay',
    'scheduler',
    'norm',
    'test_interval',
    'defense',
    'defense_byzantine_clients',
    'trimmed_mean_trim_ratio',
    'shieldfl_similarity_threshold',
    'pdfl_similarity_threshold',
    'pritrust_audit_layers',
    'pritrust_c_norm',
    'pritrust_zeta',
    'pritrust_theta_tem',
    'pritrust_theta_spa',
    'pritrust_gamma',
    'pritrust_r_max',
    'pritrust_rho',
    'pritrust_kappa',
    'pritrust_security_bits',
    'attack',
    'malicious_ratio',
    'sign_flip_lambda',
    'min_max_search_steps',
    'label_flip_source',
    'attack_target_label',
    'backdoor_fraction',
    'device',
    'gpu',
    'seed',
]


def log_args(args):
    """Log the fully resolved command-line arguments at run start."""
    args_dict = vars(args)
    important_keys = [key for key in IMPORTANT_ARG_KEYS if key in args_dict]
    other_keys = sorted(key for key in args_dict if key not in important_keys)

    LOGGER.info('\nResolved arguments:')
    LOGGER.info('Important and used parameters:')
    for key in important_keys:
        LOGGER.info('    %s: %s', key, args_dict[key])

    LOGGER.info('')
    LOGGER.info('Other parameters:')
    for key in other_keys:
        LOGGER.info('    %s: %s', key, args_dict[key])
    LOGGER.info('')


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs; return the concrete seed used."""
    if seed is None:
        return None

    if seed == 'random':
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        LOGGER.info('Generated random seed: %s', seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    LOGGER.info('Random seed set to %s', seed)
    return seed


def log_git_commit(stage, logger=None):
    """Log the current git branch, commit hash, author, time, and message."""
    logger = logger or LOGGER
    try:
        commit_log = subprocess.check_output(
            [
                'git', '-C', str(PROJECT_ROOT), 'log', '-1',
                '--format=%H%x1f%an%x1f%ad%x1f%s', '--date=iso-strict',
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning('Run %s code commit: unavailable (%s)', stage, exc)
        return

    try:
        branch_name = subprocess.check_output(
            [
                'git', '-C', str(PROJECT_ROOT), 'rev-parse',
                '--abbrev-ref', 'HEAD',
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        branch_name = 'unavailable'

    commit_parts = commit_log.split('\x1f', 3)
    if len(commit_parts) != 4:
        logger.warning('Run %s code commit: unavailable (%s)', stage,
                       commit_log or 'empty git output')
        return

    commit_hash, author_name, commit_time, commit_message = commit_parts
    if branch_name == 'HEAD':
        branch_name = 'detached'
    logger.info(
        'Run %s code commit: branch=%s | msg=%s | time=%s | hash=%s | name=%s',
        stage, branch_name, commit_message, commit_time, commit_hash,
        author_name)


def _format_filename_value(value):
    if value is None:
        return 'none'
    value = str(value).strip().lower().replace('.', 'p')
    value = re.sub(r'[^a-z0-9_-]+', '-', value)
    value = value.strip('-')
    return FILENAME_VALUE_ALIASES.get(value, value)


FILENAME_FIELD_ALIASES = {
    'dataset': 'ds',
    'model': 'm',
    'epochs': 'ep',
    'num_users': 'k',
    'frac': 'c',
    'local_ep': 'le',
    'local_bs': 'lb',
    'batch_size': 'bs',
    'optimizer': 'opt',
    'scheduler': 'sch',
    'test_interval': 'ti',
    'defense': 'def',
    'defense_byzantine_clients': 'f',
    'trimmed_mean_trim_ratio': 'tmr',
    'shieldfl_similarity_threshold': 'sft',
    'pdfl_similarity_threshold': 'pst',
    'pritrust_audit_layers': 'pal',
    'pritrust_c_norm': 'pcnorm',
    'pritrust_zeta': 'pzeta',
    'pritrust_theta_tem': 'pttem',
    'pritrust_theta_spa': 'ptspa',
    'pritrust_gamma': 'pgam',
    'pritrust_r_max': 'prmax',
    'pritrust_rho': 'prho',
    'pritrust_kappa': 'pkap',
    'pritrust_security_bits': 'psec',
    'dirichlet_alpha': 'a',
    'attack': 'atk',
    'malicious_ratio': 'mr',
    'sign_flip_lambda': 'sfl',
    'min_max_search_steps': 'mmsteps',
    'label_flip_source': 'lfs',
    'attack_target_label': 'tgt',
    'backdoor_fraction': 'bdf',
}


FILENAME_VALUE_ALIASES = {
    'batch_norm': 'bn',
    'group_norm': 'gn',
    'layer_norm': 'ln',
}


def _format_filename_field(field):
    return FILENAME_FIELD_ALIASES.get(field, field)


def get_run_name(args, prefix, fields, max_length=180):
    """Build a timestamped, argument-rich name for run artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [timestamp, prefix]
    for field in fields:
        field_name = _format_filename_field(field)
        field_value = _format_filename_value(getattr(args, field))
        parts.append(f'{field_name}-{field_value}')
    run_name = '_'.join(parts)

    if len(run_name) <= max_length:
        return run_name

    digest = hashlib.sha1(run_name.encode('utf-8')).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return f'{run_name[:prefix_length].rstrip("_-")}-{digest}'


def get_device(args):
    """Return the best available torch device for this run."""
    # Reuse the already resolved device when helper classes are created later.
    configured_device = getattr(args, 'device', None)
    if configured_device is not None:
        return torch.device(configured_device)

    requested_gpu = getattr(args, 'gpu', None)

    # Prefer CUDA whenever it is available, defaulting to GPU 0 when the user
    # does not request a specific device.
    if torch.cuda.is_available():
        gpu_index = 0 if requested_gpu is None else requested_gpu
        torch.cuda.set_device(gpu_index)
        return torch.device(f'cuda:{gpu_index}')

    # Treat --gpu as an explicit CUDA request; fall back cleanly if unavailable.
    if requested_gpu is not None:
        LOGGER.warning('CUDA GPU %s requested, but CUDA is not available.',
                       requested_gpu)

    # On Apple Silicon, MPS is much faster than CPU when the PyTorch build has it.
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def get_optimizer(args, model):
    """Build the configured optimizer for centralized and local training."""
    lr = getattr(args, 'current_lr', args.lr)
    if args.optimizer == 'sgd':
        # CIFAR ResNet baselines typically need momentum and weight decay.
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=args.weight_decay)

    raise ValueError(f'Unrecognized optimizer: {args.optimizer}')


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'mnist':
        data_dir = PROJECT_ROOT / 'data' / 'mnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(str(data_dir), train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(str(data_dir), train=False, download=True,
                                      transform=apply_transform)

        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            user_groups = mnist_noniid(
                train_dataset, args.num_users, args.dirichlet_alpha)

    elif args.dataset == 'cifar':
        data_dir = PROJECT_ROOT / 'data' / 'cifar'
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # CIFAR-10 channel statistics; more stable than generic 0.5 scaling.
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # Match train-time normalization during evaluation.
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))])

        train_dataset = datasets.CIFAR10(str(data_dir), train=True,
                                         download=True,
                                         transform=train_transform)

        test_dataset = datasets.CIFAR10(str(data_dir), train=False,
                                        download=True,
                                        transform=test_transform)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups = cifar_noniid(
                train_dataset, args.num_users, args.dirichlet_alpha)

    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    return train_dataset, test_dataset, user_groups


def average_weights(w, sample_counts=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])

    if sample_counts is not None:
        if len(sample_counts) != len(w):
            raise ValueError('sample_counts must match number of weight sets')
        total_count = float(sum(sample_counts))
        if total_count <= 0:
            raise ValueError('sample_counts must sum to a positive value')
        for key in w_avg.keys():
            if torch.is_floating_point(w_avg[key]):
                w_avg[key] = (
                    w[0][key].clone() * (sample_counts[0] / total_count))
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * (sample_counts[i] / total_count)
            else:
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], len(w)).to(w[0][key].dtype)
        return w_avg

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    LOGGER.info('\nExperimental details:')
    LOGGER.info(f'    Model     : {args.model}')
    LOGGER.info(f'    Optimizer : {args.optimizer}')
    LOGGER.info(f'    Learning  : {args.lr}')
    LOGGER.info(f'    Momentum  : {args.momentum}')
    LOGGER.info(f'    Weight decay : {args.weight_decay}')
    LOGGER.info(f'    Scheduler : {args.scheduler}')
    LOGGER.info(f'    Batch size: {args.batch_size}')
    LOGGER.info(f'    Global Rounds   : {args.epochs}\n')

    LOGGER.info('    Federated parameters:')
    if args.iid:
        LOGGER.info('    IID')
    else:
        LOGGER.info('    Non-IID')
    LOGGER.info(f'    Fraction of users  : {args.frac}')
    LOGGER.info(f'    Local Batch size   : {args.local_bs}')
    LOGGER.info(f'    Local Epochs       : {args.local_ep}\n')
    return
