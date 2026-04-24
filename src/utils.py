#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import logging
import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_log_path():
    log_path = PROJECT_ROOT / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def get_logger(log_name):
    log_file_path = get_log_path()
    log_file_path = os.path.join(
        log_file_path,
        '%s-logfile.log' % datetime.now().strftime("%Y-%m-%d"))

    logger = logging.getLogger(log_name)
    cur_format = '%(asctime)s %(levelname)s %(filename)s-%(process)d-%(funcName)s:%(lineno)d %(message)s'
    if not logging.getLogger().handlers:
        logging.basicConfig(handlers=[logging.FileHandler(filename=log_file_path,
                                                          encoding='utf-8',
                                                          mode='a+'),
                                      logging.StreamHandler()
                                      ],
                            level=logging.INFO,
                            # datefmt="%H:%M:%S",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            format=cur_format)
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
    'unequal',
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
    'test_interval',
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
    return value.strip('-')


def get_run_name(args, prefix, fields):
    """Build a timestamped, argument-rich name for run artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [timestamp, prefix]
    for field in fields:
        parts.append(f'{field}-{_format_filename_value(getattr(args, field))}')
    return '_'.join(parts)


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
    if args.optimizer == 'sgd':
        # CIFAR ResNet baselines typically need momentum and weight decay.
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    raise ValueError(f'Unrecognized optimizer: {args.optimizer}')


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
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

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = PROJECT_ROOT / 'data' / 'mnist'
        else:
            data_dir = PROJECT_ROOT / 'data' / 'fmnist'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(str(data_dir), train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(str(data_dir), train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
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
