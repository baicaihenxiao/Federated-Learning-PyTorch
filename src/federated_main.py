#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import math
import time
import pickle
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter(object):
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet18Cifar
from utils import (
    get_dataset, get_device, average_weights, get_logger, get_run_name,
    log_args, log_git_commit, set_seed, format_run_time,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / 'logs'
SAVE_DIR = PROJECT_ROOT / 'save'
SAVE_OBJECTS_DIR = SAVE_DIR / 'objects'
LOGGER = get_logger(__name__)


def format_interval(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return '{:02d}:{:02d}'.format(minutes, seconds)


def format_round_progress(current_round, total_rounds, elapsed_time):
    if current_round <= 0 or total_rounds <= 0:
        return '0% | 0/{} [00:00<?, 0.00s/it]'.format(total_rounds)

    current_round = min(current_round, total_rounds)
    percent_complete = 100.0 * current_round / total_rounds
    seconds_per_round = elapsed_time / current_round
    remaining_time = seconds_per_round * (total_rounds - current_round)

    return '{:.0f}% | {}/{} [{}<{}, {:.2f}s/it]'.format(
        percent_complete, current_round, total_rounds,
        format_interval(elapsed_time), format_interval(remaining_time),
        seconds_per_round)


def get_round_lr(args, current_round):
    if args.scheduler != 'cosine':
        return args.lr

    if args.epochs <= 1:
        return args.lr

    progress = (current_round - 1) / (args.epochs - 1)
    return 0.5 * args.lr * (1 + math.cos(math.pi * progress))


if __name__ == '__main__':
    start_time = time.time()
    log_git_commit('begin', LOGGER)

    # define paths
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    tb_logger = SummaryWriter(str(LOG_DIR))

    args = args_parser(experiment='federated')

    # Resolve the device once and share it with LocalUpdate instances.
    device = get_device(args)
    args.device = str(device)
    args.seed = set_seed(args.seed)
    LOGGER.info('Using device: %s', device)
    log_args(args)
    run_name = get_run_name(
        args,
        'fed',
        ['dataset', 'model', 'epochs', 'num_users', 'frac', 'iid',
         'norm', 'cifar_partition', 'dirichlet_alpha',
         'dirichlet_min_size', 'dirichlet_balance', 'cifar_shards_per_user',
         'local_ep', 'local_bs', 'optimizer', 'lr', 'test_interval'],
    )

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    local_updates = {
        int(user_id): LocalUpdate(args=args, dataset=train_dataset, idxs=idxs,
                                  logger=tb_logger)
        for user_id, idxs in user_groups.items()
    }

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'resnet18':
        if args.dataset != 'cifar':
            exit('Error: resnet18 is configured for cifar')
        global_model = ResNet18Cifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # LOGGER.info('%s', global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    test_epochs, test_accuracy, test_losses = [], [], []
    best_test_acc, best_test_epoch = 0.0, 0

    for epoch in range(args.epochs):
        round_start_time = time.time()
        current_epoch = epoch + 1
        local_weights = []
        local_sample_counts = []
        args.current_lr = get_round_lr(args, current_epoch)
        # LOGGER.info(f'\n | Global Training Round : {current_epoch}/{args.epochs} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        selected_user_ids = sorted(int(idx) for idx in idxs_users)

        for user_position, idx in enumerate(idxs_users, start=1):
            user_start_time = time.time()
            local_model = local_updates[int(idx)]
            w, _ = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=current_epoch)
            local_weights.append(copy.deepcopy(w))
            local_sample_counts.append(len(user_groups[idx]))
            if args.verbose:
                LOGGER.info(
                    '| Global Round : %s/%s | User : %s/%s (idx: %s) | '
                    'User Time: %.2fs',
                    current_epoch, args.epochs, user_position, m, int(idx),
                    time.time() - user_start_time)

        # update global weights
        global_weights = average_weights(local_weights, local_sample_counts)

        # update global weights
        global_model.load_state_dict(global_weights)

        should_test = (
            args.test_interval > 0 and
            (current_epoch % args.test_interval == 0 or
             current_epoch == args.epochs)
        )
        if should_test:
            test_acc, test_loss = test_inference(
                args, global_model, test_dataset)
            test_epochs.append(current_epoch)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = current_epoch

        now = time.time()
        round_time = now - round_start_time
        elapsed_time = now - start_time
        progress_summary = format_round_progress(
            current_epoch, args.epochs, elapsed_time)

        round_summary = (
            'Round Summary : {}/{} | Selected Users: {}/{} {}'.format(
                current_epoch, args.epochs, m, args.num_users,
                selected_user_ids)
        )
        if should_test:
            round_summary += (
                ' | Test Loss: {:.4f} | Test Accuracy: {:.2f}% | '
                'Best Accuracy: {:.2f}% @ Round {}'.format(
                    test_loss, 100*test_acc, 100*best_test_acc,
                    best_test_epoch)
            )
        round_summary += (
            ' | LR: {:.6f} | Progress: {} | Round Time: {} | Elapsed Time: {}'
            .format(
                args.current_lr, progress_summary, format_run_time(round_time),
                format_run_time(elapsed_time))
        )
        LOGGER.info(round_summary)

    if not test_epochs or test_epochs[-1] != args.epochs:
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_epochs.append(args.epochs)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = args.epochs
    else:
        test_acc = test_accuracy[-1]
        test_loss = test_losses[-1]

    test_accuracy_percent = [100*acc for acc in test_accuracy]

    LOGGER.info(f' \n Results after {args.epochs} global rounds of training:')
    LOGGER.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    LOGGER.info("|---- Best Test Accuracy: {:.2f}% @ Round {}".format(
        100*best_test_acc, best_test_epoch))

    # Saving the test metrics:
    file_name = SAVE_OBJECTS_DIR / f'{run_name}.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([test_epochs, test_losses, test_accuracy], f)

    # Plot Test Loss curve
    plt.figure()
    plt.title('Test Loss vs Communication Rounds')
    plt.plot(test_epochs, test_losses, color='r', marker='o')
    plt.ylabel('Test loss')
    plt.xlabel('Communication rounds')
    plt.tight_layout()
    loss_plot_path = SAVE_DIR / f'{run_name}_loss.png'
    plt.savefig(loss_plot_path)
    plt.close()
    LOGGER.info('Saved loss figure: %s', loss_plot_path)

    # Plot Test Accuracy curve
    plt.figure()
    plt.title('Test Accuracy vs Communication Rounds')
    plt.plot(test_epochs, test_accuracy_percent, marker='o', label='Test')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Communication rounds')
    plt.legend()
    plt.tight_layout()
    acc_plot_path = SAVE_DIR / f'{run_name}_acc.png'
    plt.savefig(acc_plot_path)
    plt.close()
    LOGGER.info('Saved accuracy figure: %s', acc_plot_path)

    LOGGER.info('Test epochs array: %s', test_epochs)
    LOGGER.info('Test accuracy percent array by test epoch: %s',
                test_accuracy_percent)
    LOGGER.info('Test loss array by test epoch: %s', test_losses)

    LOGGER.info('\n Total Run Time: %s', format_run_time(time.time()-start_time))
    log_git_commit('end', LOGGER)
