#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import logging
import os
from datetime import datetime
from pathlib import Path

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


LOGGER = get_logger(__name__)


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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
