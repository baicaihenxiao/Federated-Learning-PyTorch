#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import math
import time
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from attacks import (
    apply_update_attack, has_attack, sample_round_clients,
    select_malicious_clients,
)
from update import LocalUpdate, test_attack_success_rate, test_inference
from models import MLP, CNNMnist, CNNCifar, ResNet18Cifar, NUM_CLASSES
from utils import (
    get_dataset, get_device, average_weights, get_logger, get_run_name,
    log_args, log_git_commit, promote_log_file, set_log_file, set_seed,
    format_run_time,
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


def get_federated_run_name(args):
    return get_run_name(
        args,
        'fed',
        ['dataset', 'model', 'epochs', 'num_users', 'frac', 'iid',
         'dirichlet_alpha', 'norm', 'local_ep', 'local_bs', 'optimizer',
         'lr', 'attack', 'malicious_ratio', 'test_interval'],
    )


def get_run_paths(run_name):
    return {
        'metrics': SAVE_OBJECTS_DIR / f'{run_name}.pkl',
        'loss_plot': SAVE_DIR / f'{run_name}_loss.png',
        'acc_plot': SAVE_DIR / f'{run_name}_acc.png',
        'asr_plot': SAVE_DIR / f'{run_name}_asr.png',
        'temp_log': LOG_DIR / f'tmp_{run_name}.log',
        'final_log': LOG_DIR / f'{run_name}.log',
    }


def main():
    start_time = time.time()
    run_completed = False
    tb_logger = None

    # define paths
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    args = args_parser(experiment='federated')
    run_name = get_federated_run_name(args)
    run_paths = get_run_paths(run_name)
    set_log_file(run_paths['temp_log'], mode='w')

    try:
        LOGGER.info('Run artifacts name: %s', run_name)
        LOGGER.info('Active run log: %s', run_paths['temp_log'])
        log_git_commit('begin', LOGGER)
        tb_logger = SummaryWriter(str(LOG_DIR))

        # Resolve the device once and share it with LocalUpdate instances.
        device = get_device(args)
        args.device = str(device)
        args.seed = set_seed(args.seed)
        LOGGER.info('Using device: %s', device)
        log_args(args)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = get_dataset(args)
        malicious_clients = select_malicious_clients(args)
        LOGGER.info(
            'Attack configuration: attack=%s | malicious_ratio=%.2f | '
            'malicious_clients=%s/%s %s',
            args.attack, args.malicious_ratio, len(malicious_clients),
            args.num_users, sorted(malicious_clients))
        local_updates = {
            int(user_id): LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=idxs, logger=tb_logger,
                                      client_id=int(user_id),
                                      is_malicious=(
                                          int(user_id) in malicious_clients))
            for user_id, idxs in user_groups.items()
        }

        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
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
                               dim_out=NUM_CLASSES)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        # LOGGER.info('%s', global_model)

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        test_epochs, mta_accuracy, test_losses = [], [], []
        attack_success_rates = []
        best_mta_acc, best_mta_epoch = 0.0, 0

        for epoch in range(args.epochs):
            round_start_time = time.time()
            current_epoch = epoch + 1
            local_weights = []
            local_sample_counts = []
            args.current_lr = get_round_lr(args, current_epoch)
            # LOGGER.info(f'\n | Global Training Round : {current_epoch}/{args.epochs} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = sample_round_clients(args, m, malicious_clients)
            selected_user_ids = sorted(int(idx) for idx in idxs_users)
            selected_malicious_flags = []
            selected_malicious_ids = []

            for user_position, idx in enumerate(idxs_users, start=1):
                user_start_time = time.time()
                user_id = int(idx)
                local_model = local_updates[user_id]
                is_malicious = user_id in malicious_clients
                selected_malicious_flags.append(is_malicious)
                if is_malicious:
                    selected_malicious_ids.append(user_id)
                w, _ = local_model.update_weights(
                    model=copy.deepcopy(global_model),
                    global_round=current_epoch)
                local_weights.append(copy.deepcopy(w))
                local_sample_counts.append(len(user_groups[idx]))
                if args.verbose:
                    LOGGER.info(
                        '| Global Round : %s/%s | User : %s/%s (idx: %s) | '
                        'User Time: %.2fs',
                        current_epoch, args.epochs, user_position, m,
                        int(idx), time.time() - user_start_time)

            local_weights = apply_update_attack(
                args, global_weights, local_weights, local_sample_counts,
                selected_malicious_flags)

            # update global weights
            global_weights = average_weights(local_weights,
                                             local_sample_counts)

            # update global weights
            global_model.load_state_dict(global_weights)

            should_test = (
                args.test_interval > 0 and
                (current_epoch % args.test_interval == 0 or
                 current_epoch == args.epochs)
            )
            if should_test:
                mta_acc, test_loss = test_inference(
                    args, global_model, test_dataset)
                asr = test_attack_success_rate(
                    args, global_model, test_dataset)
                test_epochs.append(current_epoch)
                mta_accuracy.append(mta_acc)
                test_losses.append(test_loss)
                attack_success_rates.append(asr)
                if mta_acc > best_mta_acc:
                    best_mta_acc = mta_acc
                    best_mta_epoch = current_epoch

            now = time.time()
            round_time = now - round_start_time
            elapsed_time = now - start_time
            progress_summary = format_round_progress(
                current_epoch, args.epochs, elapsed_time)

            selected_users_summary = 'Selected Users: {}/{}'.format(
                m, args.num_users)
            if m < args.num_users:
                selected_users_summary += ' {}'.format(selected_user_ids)
            if has_attack(args):
                selected_users_summary += (
                    ' | Malicious Selected: {} {}'.format(
                        len(selected_malicious_ids),
                        sorted(selected_malicious_ids))
                )

            round_summary = (
                'Round Summary : {}/{} | {}'.format(
                    current_epoch, args.epochs, selected_users_summary)
            )
            if should_test:
                round_summary += (
                    ' | Test Loss: {:.4f} | MTA Acc: {:.2f}% | '
                    'Best MTA Acc: {:.2f}% @ Round {}'.format(
                        test_loss, 100*mta_acc, 100*best_mta_acc,
                        best_mta_epoch)
                )
                if asr is not None:
                    round_summary += ' | ASR: {:.2f}%'.format(100*asr)
            round_summary += (
                ' | LR: {:.6f} | Progress: {} | Round Time: {} | '
                'Elapsed Time: {}'.format(
                    args.current_lr, progress_summary,
                    format_run_time(round_time),
                    format_run_time(elapsed_time))
            )
            LOGGER.info(round_summary)

        if not test_epochs or test_epochs[-1] != args.epochs:
            mta_acc, test_loss = test_inference(
                args, global_model, test_dataset)
            asr = test_attack_success_rate(args, global_model, test_dataset)
            test_epochs.append(args.epochs)
            mta_accuracy.append(mta_acc)
            test_losses.append(test_loss)
            attack_success_rates.append(asr)
            if mta_acc > best_mta_acc:
                best_mta_acc = mta_acc
                best_mta_epoch = args.epochs
        else:
            mta_acc = mta_accuracy[-1]
            test_loss = test_losses[-1]
            asr = attack_success_rates[-1]

        mta_accuracy_percent = [100*acc for acc in mta_accuracy]
        asr_percent = [
            None if rate is None else 100*rate
            for rate in attack_success_rates
        ]

        LOGGER.info(' \n Results after %s global rounds of training:',
                    args.epochs)
        LOGGER.info("|---- MTA Acc: {:.2f}%".format(100*mta_acc))
        LOGGER.info("|---- Best MTA Acc: {:.2f}% @ Round {}".format(
            100*best_mta_acc, best_mta_epoch))
        if asr is not None:
            LOGGER.info("|---- ASR: {:.2f}%".format(100*asr))

        # Saving the test metrics:
        file_name = run_paths['metrics']

        with open(file_name, 'wb') as f:
            pickle.dump({
                'epochs': test_epochs,
                'test_losses': test_losses,
                'test_accuracy': mta_accuracy,
                'mta_accuracy': mta_accuracy,
                'attack_success_rates': attack_success_rates,
                'asr': attack_success_rates,
            }, f)
        LOGGER.info('Saved test metrics: %s', file_name)

        # Plot Test Loss curve
        plt.figure()
        plt.title('Test Loss vs Communication Rounds')
        plt.plot(test_epochs, test_losses, color='r', marker='o')
        plt.ylabel('Test loss')
        plt.xlabel('Communication rounds')
        plt.tight_layout()
        loss_plot_path = run_paths['loss_plot']
        plt.savefig(loss_plot_path)
        plt.close()
        LOGGER.info('Saved loss figure: %s', loss_plot_path)

        # Plot Test Accuracy curve
        plt.figure()
        plt.title('MTA Accuracy vs Communication Rounds')
        plt.plot(test_epochs, mta_accuracy_percent, marker='o',
                 label='MTA')
        plt.ylabel('MTA accuracy (%)')
        plt.xlabel('Communication rounds')
        plt.legend()
        plt.tight_layout()
        acc_plot_path = run_paths['acc_plot']
        plt.savefig(acc_plot_path)
        plt.close()
        LOGGER.info('Saved MTA accuracy figure: %s', acc_plot_path)

        if any(rate is not None for rate in attack_success_rates):
            plt.figure()
            plt.title('ASR vs Communication Rounds')
            plt.plot(test_epochs, asr_percent, color='m', marker='o',
                     label='ASR')
            plt.ylabel('ASR (%)')
            plt.xlabel('Communication rounds')
            plt.legend()
            plt.tight_layout()
            asr_plot_path = run_paths['asr_plot']
            plt.savefig(asr_plot_path)
            plt.close()
            LOGGER.info('Saved ASR figure: %s', asr_plot_path)

        LOGGER.info('Test epochs array: %s', test_epochs)
        LOGGER.info('MTA accuracy percent array by test epoch: %s',
                    mta_accuracy_percent)
        LOGGER.info('ASR percent array by test epoch: %s', asr_percent)
        LOGGER.info('Test loss array by test epoch: %s', test_losses)

        LOGGER.info('\n Total Run Time: %s',
                    format_run_time(time.time()-start_time))
        log_git_commit('end', LOGGER)
        LOGGER.info('Run completed successfully; final log file: %s',
                    run_paths['final_log'])
        run_completed = True
    except BaseException:
        LOGGER.exception('Run ended prematurely; keeping log file as: %s',
                         run_paths['temp_log'])
        raise
    finally:
        if tb_logger is not None:
            tb_logger.close()
        if run_completed:
            promote_log_file(run_paths['temp_log'], run_paths['final_log'])


if __name__ == '__main__':
    main()
