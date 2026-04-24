#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


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


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = _dataset_labels(dataset)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = _dataset_labels(dataset)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


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


def cifar_noniid(dataset, num_users, shards_per_user=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param shards_per_user: number of label-sorted shards per user. Larger
        values keep equal data volume per user while reducing label skew.
    :return:
    """
    if shards_per_user < 1:
        raise ValueError('shards_per_user must be at least 1')

    num_shards = num_users * shards_per_user
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = _dataset_labels(dataset)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    shards = np.array_split(idxs, num_shards)

    # Divide and assign label-sorted shards. With the CIFAR-10 default
    # num_users=100, shards_per_user=5 gives 100 images/shard and 500/user.
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shards_per_user,
                                        replace=False))
        idx_shard = [idx for idx in idx_shard if idx not in rand_set]
        for rand in sorted(rand_set):
            dict_users[i] = np.concatenate(
                (dict_users[i], shards[rand]), axis=0)
    return dict_users


def cifar_noniid_dirichlet(dataset, num_users, alpha=0.5, min_size=10,
                           balance=False):
    """
    Sample non-I.I.D. CIFAR10 client data with Dirichlet label skew.

    The initial allocation follows the standard per-class Dirichlet partitioning
    strategy used in many FL baselines. Set balance=True only when you
    explicitly want to rebalance clients back to near-equal sample counts after
    the Dirichlet draw.
    """
    if alpha <= 0:
        raise ValueError('alpha must be greater than 0')
    if min_size < 0:
        raise ValueError('min_size must be greater than or equal to 0')

    labels = _dataset_labels(dataset)
    num_classes = int(labels.max()) + 1
    min_client_size = 0

    while min_client_size < min_size:
        idx_batch = [[] for _ in range(num_users)]
        for class_id in range(num_classes):
            idx_class = np.where(labels == class_id)[0]
            np.random.shuffle(idx_class)

            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            split_points = (np.cumsum(proportions) * len(idx_class)).astype(int)
            split_points = split_points[:-1]

            for idx_user, idx_split in zip(
                    idx_batch, np.split(idx_class, split_points)):
                idx_user.extend(idx_split.tolist())

        min_client_size = min(len(idx_user) for idx_user in idx_batch)

    if balance:
        _rebalance_user_indices(idx_batch, len(dataset))

    for indices in idx_batch:
        np.random.shuffle(indices)

    return {
        user_id: np.asarray(indices, dtype=np.int64)
        for user_id, indices in enumerate(idx_batch)
    }


def _rebalance_user_indices(idx_batch, total_items):
    num_users = len(idx_batch)
    base_size = total_items // num_users
    remainder = total_items % num_users
    target_sizes = [
        base_size + (1 if user_id < remainder else 0)
        for user_id in range(num_users)
    ]

    for indices in idx_batch:
        np.random.shuffle(indices)

    underfull = [
        user_id for user_id, indices in enumerate(idx_batch)
        if len(indices) < target_sizes[user_id]
    ]
    overfull = [
        user_id for user_id, indices in enumerate(idx_batch)
        if len(indices) > target_sizes[user_id]
    ]

    under_pos, over_pos = 0, 0
    while under_pos < len(underfull) and over_pos < len(overfull):
        under_id = underfull[under_pos]
        over_id = overfull[over_pos]
        needed = target_sizes[under_id] - len(idx_batch[under_id])
        extra = len(idx_batch[over_id]) - target_sizes[over_id]
        move_count = min(needed, extra)

        idx_batch[under_id].extend(idx_batch[over_id][-move_count:])
        del idx_batch[over_id][-move_count:]

        if len(idx_batch[under_id]) == target_sizes[under_id]:
            under_pos += 1
        if len(idx_batch[over_id]) == target_sizes[over_id]:
            over_pos += 1


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
