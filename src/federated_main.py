#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import time
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

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
    log_args,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / 'logs'
SAVE_DIR = PROJECT_ROOT / 'save'
SAVE_OBJECTS_DIR = SAVE_DIR / 'objects'
LOGGER = get_logger(__name__)


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    tb_logger = SummaryWriter(str(LOG_DIR))

    args = args_parser()

    # Resolve the device once and share it with LocalUpdate instances.
    device = get_device(args)
    args.device = str(device)
    LOGGER.info('Using device: %s', device)
    log_args(args)
    run_name = get_run_name(
        args,
        'fed',
        ['dataset', 'model', 'epochs', 'num_users', 'frac', 'iid',
         'local_ep', 'local_bs', 'optimizer', 'lr', 'test_interval'],
    )

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

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
    train_loss, train_accuracy = [], []
    test_epochs, test_accuracy, test_losses = [], [], []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        LOGGER.info(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=tb_logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            # Evaluate every client's held-out split, not just the last sampled one.
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=tb_logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            LOGGER.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
            LOGGER.info(f'Training Loss : {np.mean(np.array(train_loss))}')
            LOGGER.info('Train Accuracy: {:.2f}% \n'.format(
                100*train_accuracy[-1]))

        current_epoch = epoch + 1
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
            LOGGER.info(
                'Test after global round %s/%s: Loss: %.4f | Accuracy: %.2f%%',
                current_epoch, args.epochs, test_loss, 100*test_acc)

    if not test_epochs or test_epochs[-1] != args.epochs:
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_epochs.append(args.epochs)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
    else:
        test_acc = test_accuracy[-1]
        test_loss = test_losses[-1]

    LOGGER.info(f' \n Results after {args.epochs} global rounds of training:')
    LOGGER.info("|---- Avg Train Accuracy: {:.2f}%".format(
        100*train_accuracy[-1]))
    LOGGER.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = SAVE_OBJECTS_DIR / (
        '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
            args.dataset, args.model, args.epochs, args.frac, args.iid,
            args.local_ep, args.local_bs)
    )

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    LOGGER.info('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication Rounds')
    plt.plot(range(1, len(train_loss)+1), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication rounds')
    plt.tight_layout()
    loss_plot_path = SAVE_DIR / f'{run_name}_loss.png'
    plt.savefig(loss_plot_path)
    plt.close()
    LOGGER.info('Saved loss figure: %s', loss_plot_path)

    # Plot Accuracy curve
    plt.figure()
    plt.title('Accuracy vs Communication Rounds')
    plt.plot(range(1, len(train_accuracy)+1),
             [100*acc for acc in train_accuracy], color='k',
             label='Avg train')
    plt.plot(test_epochs, [100*acc for acc in test_accuracy],
             marker='o', label='Test')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Communication rounds')
    plt.legend()
    plt.tight_layout()
    acc_plot_path = SAVE_DIR / f'{run_name}_acc.png'
    plt.savefig(acc_plot_path)
    plt.close()
    LOGGER.info('Saved accuracy figure: %s', acc_plot_path)
