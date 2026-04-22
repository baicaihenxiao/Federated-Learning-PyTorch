#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


MAX_RANDOM_SEED = 2**32 - 1


def seed_value(value):
    """Parse --seed as a fixed integer or the string 'random'."""
    value = str(value).strip().lower()
    if value == 'random':
        return value

    try:
        seed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            '--seed must be an integer or "random"')

    if seed < 0 or seed > MAX_RANDOM_SEED:
        raise argparse.ArgumentTypeError(
            f'--seed must be between 0 and {MAX_RANDOM_SEED}, or "random"')

    return seed


DEFAULT_MODELS = {
    'mnist': 'cnn',
    'fmnist': 'cnn',
    'cifar': 'resnet18',
}

DEFAULT_EPOCHS = {
    'mnist': 50,
    'fmnist': 50,
    'cifar': 150,
}


TRAINING_PRESETS = {
    ('cifar', 'resnet18'): {
        'sgd_lr': 0.1,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
        'scheduler': 'cosine',
    },
    ('cifar', 'cnn'): {
        'sgd_lr': 0.01,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
        'scheduler': 'cosine',
    },
    ('mnist', 'cnn'): {
        'sgd_lr': 0.01,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'batch_size': 64,
        'scheduler': 'none',
    },
    ('mnist', 'mlp'): {
        'sgd_lr': 0.01,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'batch_size': 64,
        'scheduler': 'none',
    },
    ('fmnist', 'cnn'): {
        'sgd_lr': 0.01,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'scheduler': 'none',
    },
    ('fmnist', 'mlp'): {
        'sgd_lr': 0.01,
        'adam_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'scheduler': 'none',
    },
}

FALLBACK_TRAINING_PRESET = {
    'sgd_lr': 0.01,
    'adam_lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0,
    'batch_size': 64,
    'scheduler': 'none',
}


DEFAULT_FEDERATED_ARGS = {
    'iid': 1,
    'local_ep': 10,
    'local_bs': 10,
    'test_interval': 1,
}


FEDERATED_DEFAULTS = {
    ('mnist', 1): {
        'iid': 1,
        'local_ep': 10,
        'local_bs': 10,
        'lr': 0.01,
        'test_interval': 1,
    },
    ('mnist', 0): {
        'iid': 0,
        'local_ep': 10,
        'local_bs': 10,
        'lr': 0.01,
        'test_interval': 1,
    },
    ('cifar', 1): {
        'iid': 1,
        'local_ep': 5,
        'local_bs': 32,
        'lr': 0.03,
        'test_interval': 1,
    },
    ('cifar', 0): {
        'iid': 0,
        'local_ep': 5,
        'local_bs': 32,
        'lr': 0.03,
        'test_interval': 1,
    },
}


def apply_experiment_defaults(args, experiment):
    """Fill experiment-specific defaults before optimizer presets."""
    defaults = DEFAULT_FEDERATED_ARGS.copy()
    if experiment == 'federated':
        iid = defaults['iid'] if args.iid is None else args.iid
        defaults.update(FEDERATED_DEFAULTS.get((args.dataset, iid), {}))

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def apply_training_preset(args):
    """Fill unset optimizer defaults from the selected dataset/model preset."""
    args.dataset = args.dataset.lower()
    if args.model is None:
        args.model = DEFAULT_MODELS.get(args.dataset, 'cnn')
    args.model = args.model.lower()
    args.optimizer = args.optimizer.lower()

    preset = TRAINING_PRESETS.get(
        (args.dataset, args.model), FALLBACK_TRAINING_PRESET)

    # Keep explicit command-line values; only fill values the user omitted.
    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS.get(args.dataset, 50)
    if args.lr is None:
        if args.optimizer == 'adam':
            args.lr = preset['adam_lr']
        else:
            args.lr = preset['sgd_lr']
    if args.momentum is None:
        args.momentum = preset['momentum']
    if args.weight_decay is None:
        args.weight_decay = preset['weight_decay']
    if args.batch_size is None:
        args.batch_size = preset['batch_size']
    if args.scheduler is None:
        args.scheduler = preset['scheduler']

    return args


def args_parser(experiment=None):
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of rounds of training; default depends '
                        'on dataset')
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=None,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=None,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate; default depends on dataset/model')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum; default depends on dataset/model')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight decay for SGD/Adam optimizers')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size for centralized baseline training')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['none', 'cosine'],
                        help='learning rate scheduler for baseline training')
    parser.add_argument('--test_interval', type=int, default=None,
                        help='evaluate and print test accuracy every N epochs '
                        'or global rounds during training; set 0 to disable '
                        'intermediate test evaluation')

    # model arguments
    parser.add_argument('--model', type=str.lower, default=None,
                        choices=['mlp', 'cnn', 'resnet18'],
                        help='model name: mlp, cnn, or resnet18; default '
                        'depends on dataset')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str.lower, default='cifar',
                        choices=['mnist', 'fmnist', 'cifar'],
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', type=int, default=None, help="To use CUDA, set \
                        to a specific GPU ID. If omitted, MPS is used when \
                        available, otherwise CPU is used.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'], help="type of optimizer")
    parser.add_argument('--iid', type=int, default=None,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=seed_value, default=1,
                        help='random seed integer, or "random" to choose a '
                        'fresh seed for this run')
    args = apply_experiment_defaults(parser.parse_args(), experiment)
    args = apply_training_preset(args)
    if args.test_interval < 0:
        parser.error('--test_interval must be greater than or equal to 0')
    return args
