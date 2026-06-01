#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import numpy as np
import torch


NO_ATTACK = 'none'
SIGN_FLIP = 'sign_flip'
MIN_MAX = 'min_max'
LABEL_FLIP = 'label_flip'
BACKDOOR = 'backdoor'

ATTACK_CHOICES = [NO_ATTACK, SIGN_FLIP, MIN_MAX, LABEL_FLIP, BACKDOOR]
UPDATE_ATTACKS = {SIGN_FLIP, MIN_MAX}
DATA_ATTACKS = {LABEL_FLIP, BACKDOOR}
TARGETED_ATTACKS = {LABEL_FLIP, BACKDOOR}

NORMALIZATION = {
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,),
        'trigger_size': 3,
    },
    'cifar': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'trigger_size': 5,
    },
}


def has_attack(args):
    return args.attack != NO_ATTACK and args.malicious_ratio > 0


def is_data_attack(args):
    return has_attack(args) and args.attack in DATA_ATTACKS


def is_update_attack(args):
    return has_attack(args) and args.attack in UPDATE_ATTACKS


def is_targeted_attack(args):
    return args.attack in TARGETED_ATTACKS


def select_malicious_clients(args):
    """Select a fixed malicious client set without perturbing round sampling."""
    num_malicious = int(round(args.malicious_ratio * args.num_users))
    num_malicious = min(max(num_malicious, 0), args.num_users)
    if args.attack == NO_ATTACK or num_malicious == 0:
        return set()

    seed = args.seed if isinstance(args.seed, int) else None
    rng = np.random.default_rng(seed)
    clients = rng.choice(args.num_users, num_malicious, replace=False)
    return set(int(client_id) for client_id in clients)


def sample_round_clients(args, num_selected, malicious_clients):
    """Sample a round with a fixed malicious-client quota."""
    num_selected = min(max(int(num_selected), 1), args.num_users)
    malicious_clients = set(int(client_id) for client_id in malicious_clients)
    benign_clients = [
        client_id for client_id in range(args.num_users)
        if client_id not in malicious_clients
    ]

    malicious_count = fixed_round_malicious_count(args, num_selected)
    malicious_count = min(malicious_count, len(malicious_clients),
                          num_selected)
    benign_count = min(num_selected - malicious_count, len(benign_clients))

    if malicious_count + benign_count < num_selected:
        remaining = num_selected - malicious_count - benign_count
        available_malicious = len(malicious_clients) - malicious_count
        extra_malicious = min(remaining, available_malicious)
        malicious_count += extra_malicious
        remaining -= extra_malicious
        if remaining > 0:
            raise ValueError('not enough clients to sample this round')

    selected = []
    if malicious_count > 0:
        selected.extend(
            np.random.choice(sorted(malicious_clients), malicious_count,
                             replace=False).tolist())
    if benign_count > 0:
        selected.extend(
            np.random.choice(benign_clients, benign_count,
                             replace=False).tolist())
    np.random.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def fixed_round_malicious_count(args, num_selected):
    if args.attack == NO_ATTACK or args.malicious_ratio <= 0:
        return 0
    malicious_count = int(np.floor(args.malicious_ratio * num_selected + 0.5))
    return min(max(malicious_count, 1), num_selected)


def sample_backdoor_indices(args, dataset_size, is_malicious):
    if (
        not is_malicious or
        args.attack != BACKDOOR or
        args.backdoor_fraction <= 0 or
        dataset_size <= 0
    ):
        return set()

    poison_count = int(round(args.backdoor_fraction * dataset_size))
    poison_count = min(max(poison_count, 0), dataset_size)
    if poison_count == 0:
        return set()

    selected = np.random.choice(dataset_size, poison_count, replace=False)
    return set(int(idx) for idx in selected)


def apply_data_attack(args, images, labels, is_malicious,
                      local_indices=None, backdoor_indices=None):
    if not is_malicious or args.attack not in DATA_ATTACKS:
        return images, labels

    if args.attack == LABEL_FLIP:
        return _apply_label_flip(args, images, labels)

    if args.attack == BACKDOOR:
        return _apply_backdoor(args, images, labels, local_indices,
                               backdoor_indices)

    return images, labels


def stamp_backdoor_trigger(args, images):
    triggered_images = images.clone()
    _stamp_white_trigger(args, triggered_images)
    return triggered_images


def apply_update_attack(args, global_weights, local_weights, sample_counts,
                        malicious_flags):
    if args.attack not in UPDATE_ATTACKS or not any(malicious_flags):
        return local_weights

    if args.attack == SIGN_FLIP:
        return [
            _sign_flip_weights(args, global_weights, weights)
            if is_malicious else weights
            for weights, is_malicious in zip(local_weights, malicious_flags)
        ]

    if args.attack == MIN_MAX:
        return _min_max_weights(args, global_weights, local_weights,
                                sample_counts, malicious_flags)

    return local_weights


def _apply_label_flip(args, images, labels):
    poisoned_labels = labels.clone()
    source = int(args.label_flip_source)
    target = int(args.attack_target_label)
    poisoned_labels[poisoned_labels == source] = target
    return images, poisoned_labels


def _apply_backdoor(args, images, labels, local_indices, backdoor_indices):
    if local_indices is None or not backdoor_indices:
        return images, labels

    local_indices = local_indices.detach().cpu().tolist()
    mask_values = [int(idx) in backdoor_indices for idx in local_indices]
    if not any(mask_values):
        return images, labels

    mask = torch.tensor(mask_values, dtype=torch.bool, device=images.device)
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    triggered_images = poisoned_images[mask].clone()
    _stamp_white_trigger(args, triggered_images)
    poisoned_images[mask] = triggered_images
    poisoned_labels[mask] = int(args.attack_target_label)
    return poisoned_images, poisoned_labels


def _stamp_white_trigger(args, images):
    if images.numel() == 0:
        return

    norm = NORMALIZATION[args.dataset]
    trigger_size = int(norm['trigger_size'])
    height, width = images.shape[-2], images.shape[-1]
    if trigger_size > height or trigger_size > width:
        raise ValueError('trigger size exceeds image dimensions')

    mean = torch.tensor(norm['mean'], device=images.device,
                        dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(norm['std'], device=images.device,
                       dtype=images.dtype).view(1, -1, 1, 1)
    white_value = (1.0 - mean) / std
    images[:, :, -trigger_size:, -trigger_size:] = white_value


def _sign_flip_weights(args, global_weights, local_weights):
    attacked = copy.deepcopy(local_weights)
    scale = float(args.sign_flip_lambda)
    for key, local_value in local_weights.items():
        if not torch.is_floating_point(local_value):
            attacked[key] = local_value.clone()
            continue
        global_value = global_weights[key].to(
            device=local_value.device, dtype=local_value.dtype)
        attacked[key] = global_value - scale * (local_value - global_value)
    return attacked


def _min_max_weights(args, global_weights, local_weights, sample_counts,
                     malicious_flags):
    benign_positions = [
        idx for idx, is_malicious in enumerate(malicious_flags)
        if not is_malicious
    ]
    malicious_positions = [
        idx for idx, is_malicious in enumerate(malicious_flags)
        if is_malicious
    ]
    if not benign_positions or not malicious_positions:
        return local_weights

    keys = _floating_keys(global_weights)
    if not keys:
        return local_weights

    benign_vectors = [
        _delta_vector(local_weights[idx], global_weights, keys)
        for idx in benign_positions
    ]
    spread = _max_pairwise_distance(benign_vectors)
    benign_mean = _weighted_mean_vector(
        benign_vectors, [sample_counts[idx] for idx in benign_positions])

    direction = -benign_mean
    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm.item() == 0.0:
        return local_weights
    direction = direction / direction_norm

    candidate_delta = _largest_in_spread_delta(
        args, benign_mean, direction, benign_vectors, spread)
    malicious_weights = _weights_from_delta_vector(
        global_weights, candidate_delta, keys)

    attacked_weights = list(local_weights)
    for idx in malicious_positions:
        attacked_weights[idx] = copy.deepcopy(malicious_weights)
    return attacked_weights


def _floating_keys(weights):
    return [
        key for key, value in weights.items()
        if torch.is_floating_point(value)
    ]


def _delta_vector(weights, global_weights, keys):
    return torch.cat([
        (weights[key].detach().to(device='cpu', dtype=torch.float32) -
         global_weights[key].detach().to(device='cpu', dtype=torch.float32)
         ).reshape(-1)
        for key in keys
    ])


def _weighted_mean_vector(vectors, counts):
    total_count = float(sum(counts))
    if total_count <= 0:
        raise ValueError('sample_counts must sum to a positive value')

    mean = torch.zeros_like(vectors[0])
    for vector, count in zip(vectors, counts):
        mean += vector * (float(count) / total_count)
    return mean


def _max_pairwise_distance(vectors):
    if len(vectors) < 2:
        return 0.0

    max_distance = 0.0
    for left_idx in range(len(vectors)):
        for right_idx in range(left_idx + 1, len(vectors)):
            distance = torch.linalg.vector_norm(
                vectors[left_idx] - vectors[right_idx]).item()
            max_distance = max(max_distance, distance)
    return max_distance


def _max_distance_to_benign(candidate, benign_vectors):
    max_distance = 0.0
    for benign_vector in benign_vectors:
        distance = torch.linalg.vector_norm(candidate - benign_vector).item()
        max_distance = max(max_distance, distance)
    return max_distance


def _largest_in_spread_delta(args, benign_mean, direction, benign_vectors,
                             spread):
    if spread <= 0:
        return benign_mean

    low, high = 0.0, float(spread)
    for _ in range(int(args.min_max_search_steps)):
        mid = (low + high) / 2.0
        candidate = benign_mean + mid * direction
        if _max_distance_to_benign(candidate, benign_vectors) <= spread:
            low = mid
        else:
            high = mid
    return benign_mean + low * direction


def _weights_from_delta_vector(global_weights, delta_vector, keys):
    key_set = set(keys)
    weights = {}
    offset = 0
    for key, global_value in global_weights.items():
        if key not in key_set:
            weights[key] = global_value.clone()
            continue

        numel = global_value.numel()
        delta = delta_vector[offset:offset + numel].view_as(global_value)
        delta = delta.to(device=global_value.device, dtype=global_value.dtype)
        weights[key] = global_value.clone() + delta
        offset += numel
    return weights
