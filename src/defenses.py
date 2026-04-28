#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math

import torch

from utils import average_weights


FEDAVG = 'fedavg'
KRUM = 'krum'
TRIMMED_MEAN = 'trimmed_mean'
SHIELDFL = 'shieldfl'
PDFL = 'pdfl'
PRITRUST_FL = 'pritrust_fl'

DEFENSE_CHOICES = [
    FEDAVG, KRUM, TRIMMED_MEAN, SHIELDFL, PDFL, PRITRUST_FL,
]

DEFENSE_ALIASES = {
    'fedavg': FEDAVG,
    'fed_avg': FEDAVG,
    'krum': KRUM,
    'trimmed_mean': TRIMMED_MEAN,
    'trimmedmean': TRIMMED_MEAN,
    'shieldfl': SHIELDFL,
    'shield_fl': SHIELDFL,
    'pdfl': PDFL,
    'pritrust_fl': PRITRUST_FL,
    'pritrustfl': PRITRUST_FL,
}


def normalize_defense_name(value):
    key = str(value).strip().lower().replace('-', '_')
    try:
        return DEFENSE_ALIASES[key]
    except KeyError:
        choices = ', '.join(DEFENSE_CHOICES)
        raise ValueError('unsupported defense "{}"; choose from {}'.format(
            value, choices))


def aggregate_weights(args, global_weights, local_weights, sample_counts,
                      client_ids=None, state=None):
    """Aggregate client models with the selected defense method.

    ShieldFL, PDFL, and PriTrust-FL are implemented as plaintext robust
    aggregators only. Their privacy-preserving cryptographic/DP protocols are
    intentionally omitted for baseline experiments.
    """
    if not local_weights:
        raise ValueError('local_weights must not be empty')
    if len(local_weights) != len(sample_counts):
        raise ValueError('sample_counts must match local_weights')

    defense = normalize_defense_name(getattr(args, 'defense', FEDAVG))
    state = {} if state is None else state
    if client_ids is None:
        client_ids = list(range(len(local_weights)))

    if defense == FEDAVG:
        return average_weights(local_weights, sample_counts), {
            'defense': FEDAVG,
            'selected_count': len(local_weights),
        }
    if defense == KRUM:
        return _krum(args, global_weights, local_weights, sample_counts,
                     client_ids)
    if defense == TRIMMED_MEAN:
        return _trimmed_mean(args, local_weights, sample_counts)
    if defense == SHIELDFL:
        return _shieldfl(args, global_weights, local_weights, sample_counts)
    if defense == PDFL:
        return _pdfl(args, global_weights, local_weights, sample_counts,
                     client_ids)
    if defense == PRITRUST_FL:
        return _pritrust_fl(args, global_weights, local_weights,
                            sample_counts, client_ids, state)

    raise ValueError('unsupported defense: {}'.format(defense))


def _resolve_byzantine_count(args, num_clients, require_krum_feasible=False):
    explicit = getattr(args, 'defense_byzantine_clients', None)
    if explicit is not None:
        count = int(explicit)
    else:
        ratio = float(getattr(args, 'malicious_ratio', 0.0))
        count = int(math.floor(ratio * num_clients + 0.5))
        if getattr(args, 'attack', 'none') != 'none' and ratio > 0:
            count = max(count, 1)

    count = min(max(count, 0), max(num_clients - 1, 0))
    requested = count
    if require_krum_feasible:
        max_feasible = max((num_clients - 3) // 2, 0)
        count = min(count, max_feasible)
    return count, requested


def _floating_keys(weights):
    return [
        key for key, value in weights.items()
        if torch.is_floating_point(value)
    ]


def _delta_matrix(local_weights, global_weights, keys):
    vectors = []
    for weights in local_weights:
        vectors.append(torch.cat([
            (weights[key].detach().to(device='cpu', dtype=torch.float32) -
             global_weights[key].detach().to(device='cpu',
                                             dtype=torch.float32)).reshape(-1)
            for key in keys
        ]))
    return torch.stack(vectors, dim=0)


def _safe_normalize(vectors, eps=1e-12):
    norms = torch.linalg.vector_norm(vectors, dim=1)
    safe_norms = torch.clamp(norms, min=eps)
    normalized = vectors / safe_norms.unsqueeze(1)
    normalized[norms <= eps] = 0.0
    return normalized, norms


def _cosine_matrix(vectors):
    normalized, norms = _safe_normalize(vectors)
    return normalized.matmul(normalized.t()).clamp(min=-1.0, max=1.0), norms


def _weighted_average_state(local_weights, coefficients):
    if len(local_weights) != len(coefficients):
        raise ValueError('coefficients must match local_weights')
    total = float(sum(coefficients))
    if total <= 0:
        raise ValueError('coefficients must sum to a positive value')

    result = copy.deepcopy(local_weights[0])
    max_position = max(range(len(coefficients)),
                       key=lambda idx: float(coefficients[idx]))
    for key in result.keys():
        if torch.is_floating_point(result[key]):
            result[key] = local_weights[0][key].clone() * (
                float(coefficients[0]) / total)
            for idx in range(1, len(local_weights)):
                result[key] += local_weights[idx][key] * (
                    float(coefficients[idx]) / total)
        else:
            result[key] = local_weights[max_position][key].clone()
    return result


def _subset(values, positions):
    return [values[position] for position in positions]


def _krum(args, global_weights, local_weights, sample_counts, client_ids):
    num_clients = len(local_weights)
    if num_clients == 1:
        return copy.deepcopy(local_weights[0]), {
            'defense': KRUM,
            'selected_count': 1,
            'selected_clients': [int(client_ids[0])],
        }

    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': KRUM,
            'selected_count': num_clients,
            'fallback': 'no floating parameters',
        }

    byzantine_count, requested = _resolve_byzantine_count(
        args, num_clients, require_krum_feasible=True)
    neighbor_count = max(num_clients - byzantine_count - 2, 1)

    vectors = _delta_matrix(local_weights, global_weights, keys)
    distances = torch.cdist(vectors, vectors, p=2).pow(2)
    sorted_distances, _ = torch.sort(distances, dim=1)
    scores = sorted_distances[:, 1:neighbor_count + 1].sum(dim=1)
    selected_position = int(torch.argmin(scores).item())
    info = {
        'defense': KRUM,
        'selected_count': 1,
        'selected_clients': [int(client_ids[selected_position])],
        'byzantine_count': byzantine_count,
    }
    if requested != byzantine_count:
        info['requested_byzantine_count'] = requested
    return copy.deepcopy(local_weights[selected_position]), info


def _trimmed_mean(args, local_weights, sample_counts):
    num_clients = len(local_weights)
    if num_clients == 1:
        return copy.deepcopy(local_weights[0]), {
            'defense': TRIMMED_MEAN,
            'selected_count': 1,
            'trim_count': 0,
        }

    trim_ratio = getattr(args, 'trimmed_mean_trim_ratio', None)
    if trim_ratio is None:
        trim_count, requested = _resolve_byzantine_count(args, num_clients)
    else:
        requested = int(math.floor(float(trim_ratio) * num_clients))
        trim_count = requested
    max_trim = max((num_clients - 1) // 2, 0)
    trim_count = min(max(trim_count, 0), max_trim)

    result = copy.deepcopy(local_weights[0])
    averaged_nonfloating = None
    for key in result.keys():
        if torch.is_floating_point(result[key]):
            values = torch.stack([
                weights[key].detach().to(dtype=torch.float32)
                for weights in local_weights
            ], dim=0)
            sorted_values, _ = torch.sort(values, dim=0)
            if trim_count > 0:
                sorted_values = sorted_values[trim_count:-trim_count]
            result[key] = sorted_values.mean(dim=0).to(
                device=local_weights[0][key].device,
                dtype=local_weights[0][key].dtype)
        else:
            if averaged_nonfloating is None:
                averaged_nonfloating = average_weights(
                    local_weights, sample_counts)
            result[key] = averaged_nonfloating[key]

    info = {
        'defense': TRIMMED_MEAN,
        'selected_count': num_clients,
        'trim_count': trim_count,
    }
    if requested != trim_count:
        info['requested_trim_count'] = requested
    return result, info


def _cosine_centrality(similarities):
    if similarities.size(0) == 1:
        return torch.ones(1, dtype=similarities.dtype)
    return (similarities.sum(dim=1) - torch.diag(similarities)) / (
        similarities.size(0) - 1)


def _shieldfl(args, global_weights, local_weights, sample_counts):
    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': SHIELDFL,
            'selected_count': len(local_weights),
            'fallback': 'no floating parameters',
        }

    vectors = _delta_matrix(local_weights, global_weights, keys)
    similarities, _ = _cosine_matrix(vectors)
    threshold = float(getattr(args, 'shieldfl_similarity_threshold', 0.0))
    centrality = _cosine_centrality(similarities)
    trust = torch.clamp(centrality - threshold, min=0.0)

    coefficients = [
        float(sample_count) * float(score)
        for sample_count, score in zip(sample_counts, trust)
    ]
    if sum(coefficients) <= 0:
        return average_weights(local_weights, sample_counts), {
            'defense': SHIELDFL,
            'selected_count': len(local_weights),
            'fallback': 'zero trust scores',
        }

    return _weighted_average_state(local_weights, coefficients), {
        'defense': SHIELDFL,
        'selected_count': len(local_weights),
        'min_trust': float(torch.min(trust).item()),
        'max_trust': float(torch.max(trust).item()),
    }


def _pdfl(args, global_weights, local_weights, sample_counts, client_ids):
    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': PDFL,
            'selected_count': len(local_weights),
            'fallback': 'no floating parameters',
        }

    vectors = _delta_matrix(local_weights, global_weights, keys)
    similarities, _ = _cosine_matrix(vectors)
    threshold = float(getattr(args, 'pdfl_similarity_threshold', 0.0))
    cluster = _largest_similarity_component(
        similarities, sample_counts, threshold)

    selected_weights = _subset(local_weights, cluster)
    selected_counts = _subset(sample_counts, cluster)
    selected_clients = _subset(client_ids, cluster)
    return average_weights(selected_weights, selected_counts), {
        'defense': PDFL,
        'selected_count': len(cluster),
        'selected_clients': [int(client_id) for client_id in selected_clients],
        'similarity_threshold': threshold,
    }


def _largest_similarity_component(similarities, sample_counts, threshold):
    num_clients = similarities.size(0)
    visited = [False] * num_clients
    best_component = []
    best_score = None

    for start in range(num_clients):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = torch.nonzero(
                similarities[current] >= threshold,
                as_tuple=False).view(-1).tolist()
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        sample_weight = sum(float(sample_counts[idx]) for idx in component)
        internal_similarity = _mean_internal_similarity(
            similarities, component)
        score = (sample_weight, internal_similarity, len(component))
        if best_score is None or score > best_score:
            best_score = score
            best_component = component

    return sorted(best_component)


def _mean_internal_similarity(similarities, component):
    if len(component) < 2:
        return -1.0
    total, count = 0.0, 0
    for left_pos, left_idx in enumerate(component):
        for right_idx in component[left_pos + 1:]:
            total += float(similarities[left_idx, right_idx].item())
            count += 1
    return total / count


def _pritrust_fl(args, global_weights, local_weights, sample_counts,
                 client_ids, state):
    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': PRITRUST_FL,
            'selected_count': len(local_weights),
            'fallback': 'no floating parameters',
        }

    vectors = _delta_matrix(local_weights, global_weights, keys)
    similarities, norms = _cosine_matrix(vectors)
    anchor_position = _cosine_medoid(similarities, sample_counts)
    unit_vectors, _ = _safe_normalize(vectors)
    anchor = unit_vectors[anchor_position]
    direction_trust = torch.clamp(unit_vectors.matmul(anchor), min=0.0)
    norm_trust = _norm_consistency_scores(norms)
    instant_trust = direction_trust * norm_trust

    trust_memory = state.setdefault('pritrust_client_trust', {})
    momentum = float(getattr(args, 'pritrust_momentum', 0.8))
    effective_trust = []
    for client_id, instant_score in zip(client_ids, instant_trust):
        client_id = int(client_id)
        instant_score = float(instant_score.item())
        previous_score = float(trust_memory.get(client_id, 1.0))
        updated_score = momentum * previous_score + (
            1.0 - momentum) * instant_score
        trust_memory[client_id] = updated_score
        effective_trust.append(instant_score * updated_score)

    coefficients = [
        float(sample_count) * float(score)
        for sample_count, score in zip(sample_counts, effective_trust)
    ]
    if sum(coefficients) <= 0:
        return average_weights(local_weights, sample_counts), {
            'defense': PRITRUST_FL,
            'selected_count': len(local_weights),
            'fallback': 'zero trust scores',
        }

    return _weighted_average_state(local_weights, coefficients), {
        'defense': PRITRUST_FL,
        'selected_count': len(local_weights),
        'anchor_client': int(client_ids[anchor_position]),
        'min_trust': min(effective_trust),
        'max_trust': max(effective_trust),
    }


def _cosine_medoid(similarities, sample_counts):
    sample_weights = torch.tensor(sample_counts, dtype=similarities.dtype)
    weighted = similarities * sample_weights.view(1, -1)
    weighted = weighted - torch.diag(torch.diag(weighted))
    denom = max(float(sum(sample_counts) - min(sample_counts)), 1.0)
    centrality = weighted.sum(dim=1) / denom
    return int(torch.argmax(centrality).item())


def _norm_consistency_scores(norms, eps=1e-12):
    positive_norms = norms[norms > eps]
    if positive_norms.numel() == 0:
        return torch.zeros_like(norms)
    median_norm = torch.median(positive_norms)
    ratio = torch.clamp(norms / torch.clamp(median_norm, min=eps), min=eps)
    return torch.minimum(ratio, 1.0 / ratio).clamp(min=0.0, max=1.0)
