#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np


DEFAULT_DIRICHLET_ALPHA = 0.3


def _dataset_labels(dataset):
    """Return labels from torchvision datasets across API versions."""
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels
    else:
        raise AttributeError(
            f'{type(dataset).__name__} does not expose targets/train_labels')

    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()

    return np.asarray(labels)


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, alpha=DEFAULT_DIRICHLET_ALPHA):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param alpha: Dirichlet concentration; smaller values create more skew.
    :return:
    """
    return dirichlet_noniid(dataset, num_users, alpha)


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, alpha=DEFAULT_DIRICHLET_ALPHA):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param alpha: Dirichlet concentration; smaller values create more skew.
    :return:
    """
    return dirichlet_noniid(dataset, num_users, alpha)


def dirichlet_noniid(dataset, num_users, alpha=DEFAULT_DIRICHLET_ALPHA):
    """Sample non-IID client data with Dirichlet label skew."""
    if num_users < 1:
        raise ValueError('num_users must be at least 1')
    if num_users > len(dataset):
        raise ValueError('num_users must not exceed dataset size')
    if alpha <= 0:
        raise ValueError('alpha must be greater than 0')

    labels = _dataset_labels(dataset)
    num_classes = int(labels.max()) + 1
    idx_batch = [[] for _ in range(num_users)]

    for class_id in range(num_classes):
        idx_class = np.where(labels == class_id)[0]
        np.random.shuffle(idx_class)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        split_points = (np.cumsum(proportions)[:-1] * len(idx_class)).astype(int)

        for idx_user, idx_split in zip(
                idx_batch, np.split(idx_class, split_points)):
            idx_user.extend(idx_split.tolist())

    _ensure_nonempty_clients(idx_batch)
    for indices in idx_batch:
        np.random.shuffle(indices)

    return {
        user_id: np.asarray(indices, dtype=np.int64)
        for user_id, indices in enumerate(idx_batch)
    }


def _ensure_nonempty_clients(idx_batch):
    empty_users = [i for i, indices in enumerate(idx_batch) if not indices]
    for empty_user in empty_users:
        donor = max(range(len(idx_batch)), key=lambda i: len(idx_batch[i]))
        if len(idx_batch[donor]) <= 1:
            raise ValueError('cannot assign at least one sample to every user')
        idx_batch[empty_user].append(idx_batch[donor].pop())
