#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import hashlib
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
        return _shieldfl(args, global_weights, local_weights, sample_counts,
                         client_ids, state)
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


def _plaintext_normalized_deltas(local_weights, global_weights, keys):
    """Adapt the papers' normalized-gradient inputs to local model states."""
    vectors = _delta_matrix(local_weights, global_weights, keys)
    normalized, norms = _safe_normalize(vectors)
    accepted = [
        int(position)
        for position in torch.nonzero(norms > 0.0, as_tuple=False)
        .view(-1).tolist()
    ]
    return vectors, normalized, norms, accepted


def _median_norm(norms, positions):
    selected_norms = torch.tensor(
        [float(norms[position].item()) for position in positions],
        dtype=torch.float32)
    return float(torch.median(selected_norms).item())


def _normalized_delta_update(global_weights, local_weights, keys, positions,
                             coefficients, normalized_vectors, norms):
    """Apply a robust normalized update direction at a median raw-delta scale."""
    if len(positions) != len(coefficients):
        raise ValueError('coefficients must match positions')
    total = float(sum(coefficients))
    if total <= 0:
        raise ValueError('coefficients must sum to a positive value')

    normalized_coefficients = torch.tensor(
        [float(coefficient) / total for coefficient in coefficients],
        dtype=normalized_vectors.dtype)
    aggregate_direction = torch.sum(
        normalized_vectors[positions] * normalized_coefficients.view(-1, 1),
        dim=0)
    aggregate_scale = _median_norm(norms, positions)
    aggregate_delta = aggregate_direction * aggregate_scale

    result = copy.deepcopy(global_weights)
    max_position = positions[
        max(range(len(coefficients)),
            key=lambda idx: float(coefficients[idx]))]
    offset = 0
    for key, global_value in global_weights.items():
        if torch.is_floating_point(global_value):
            size = global_value.numel()
            delta = aggregate_delta[offset:offset + size].view_as(
                global_value)
            result[key] = global_value.clone() + delta.to(
                device=global_value.device, dtype=global_value.dtype)
            offset += size
        else:
            result[key] = local_weights[max_position][key].clone()

    return result, aggregate_direction.detach().clone(), aggregate_scale


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


def _shieldfl(args, global_weights, local_weights, sample_counts,
              client_ids, state):
    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': SHIELDFL,
            'selected_count': len(local_weights),
            'fallback': 'no floating parameters',
        }

    _, normalized, norms, accepted = _plaintext_normalized_deltas(
        local_weights, global_weights, keys)
    if not accepted:
        return average_weights(local_weights, sample_counts), {
            'defense': SHIELDFL,
            'selected_count': len(local_weights),
            'fallback': 'zero update vectors',
        }

    previous = state.get('shieldfl_previous_aggregate')
    if previous is None or previous.numel() != normalized.size(1):
        coefficients = [1.0 for _ in accepted]
        aggregated, aggregate_direction, aggregate_scale = (
            _normalized_delta_update(global_weights, local_weights, keys,
                                     accepted, coefficients, normalized,
                                     norms))
        state['shieldfl_previous_aggregate'] = aggregate_direction
        return aggregated, {
            'defense': SHIELDFL,
            'selected_count': len(accepted),
            'selected_clients': [int(client_ids[position])
                                 for position in accepted],
            'rejected_count': len(local_weights) - len(accepted),
            'aggregation_scale': aggregate_scale,
            'initial_round': True,
        }

    previous_normalized, previous_norms = _safe_normalize(previous.view(1, -1))
    if float(previous_norms[0].item()) <= 0.0:
        coefficients = [1.0 for _ in accepted]
        baseline_position = accepted[0]
        fallback = 'zero previous aggregate'
    else:
        previous_vector = previous_normalized[0]
        previous_similarities = normalized.matmul(previous_vector).clamp(
            min=-1.0, max=1.0)
        baseline_position = min(
            accepted,
            key=lambda position: float(previous_similarities[position].item()))
        baseline_similarities = normalized.matmul(
            normalized[baseline_position]).clamp(min=-1.0, max=1.0)
        coefficients = [
            max(1.0 - float(baseline_similarities[position].item()), 0.0)
            for position in accepted
        ]
        fallback = None

    if sum(coefficients) <= 0.0:
        coefficients = [1.0 for _ in accepted]
        if fallback is None:
            fallback = 'zero confidence mass'

    aggregated, aggregate_direction, aggregate_scale = (
        _normalized_delta_update(global_weights, local_weights, keys,
                                 accepted, coefficients, normalized, norms))
    state['shieldfl_previous_aggregate'] = aggregate_direction

    info = {
        'defense': SHIELDFL,
        'selected_count': len(accepted),
        'selected_clients': [int(client_ids[position])
                             for position in accepted],
        'rejected_count': len(local_weights) - len(accepted),
        'poisonous_baseline_client': int(client_ids[baseline_position]),
        'min_confidence': min(coefficients),
        'max_confidence': max(coefficients),
        'aggregation_scale': aggregate_scale,
    }
    if fallback is not None:
        info['fallback'] = fallback
    return aggregated, info


def _pdfl(args, global_weights, local_weights, sample_counts, client_ids):
    keys = _floating_keys(global_weights)
    if not keys:
        return average_weights(local_weights, sample_counts), {
            'defense': PDFL,
            'selected_count': len(local_weights),
            'fallback': 'no floating parameters',
        }

    _, normalized, norms, accepted = _plaintext_normalized_deltas(
        local_weights, global_weights, keys)
    if not accepted:
        return average_weights(local_weights, sample_counts), {
            'defense': PDFL,
            'selected_count': len(local_weights),
            'fallback': 'zero update vectors',
        }

    similarities = normalized.matmul(normalized.t()).clamp(
        min=-1.0, max=1.0)
    threshold = float(getattr(args, 'pdfl_similarity_threshold', 0.0))
    cluster = _largest_similarity_component(
        similarities, accepted, threshold)

    selected_clients = _subset(client_ids, cluster)
    similarity_weights = [
        float(torch.mean(similarities[position, cluster]).item())
        for position in cluster
    ]
    fallback = None
    if sum(similarity_weights) <= 0.0:
        similarity_weights = [1.0 for _ in cluster]
        fallback = 'zero similarity weight mass'

    aggregated, _, aggregate_scale = _normalized_delta_update(
        global_weights, local_weights, keys, cluster, similarity_weights,
        normalized, norms)
    info = {
        'defense': PDFL,
        'selected_count': len(cluster),
        'selected_clients': [int(client_id) for client_id in selected_clients],
        'similarity_threshold': threshold,
        'rejected_count': len(local_weights) - len(accepted),
        'min_similarity_weight': min(similarity_weights),
        'max_similarity_weight': max(similarity_weights),
        'aggregation_scale': aggregate_scale,
    }
    if fallback is not None:
        info['fallback'] = fallback
    return aggregated, info


def _largest_similarity_component(similarities, positions, threshold):
    position_set = set(int(position) for position in positions)
    visited = {position: False for position in position_set}
    best_component = []
    best_score = None

    for start in sorted(position_set):
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
                neighbor = int(neighbor)
                if neighbor in position_set and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        internal_similarity = _mean_internal_similarity(
            similarities, component)
        score = (len(component), internal_similarity)
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

    trust_memory = state.setdefault('pritrust_client_trust', {})
    previous_trust = [
        float(trust_memory.get(int(client_id), 1.0))
        for client_id in client_ids
    ]
    participant_weights = _normalize_or_uniform(previous_trust)
    previous_global_weights = state.get('pritrust_previous_global_weights')
    round_number = int(state.get('pritrust_round', 0)) + 1
    audited_keys = _select_pritrust_audited_layers(
        args, keys, client_ids, round_number)

    client_scores = _pritrust_consistency_scores(
        args, global_weights, previous_global_weights, local_weights,
        audited_keys, participant_weights)
    retained_positions, threshold, median_score, mad_score = (
        _adaptive_filter_positions(args, client_scores))

    updated_trust = []
    rho = float(getattr(args, 'pritrust_rho', 0.8))
    kappa = float(getattr(args, 'pritrust_kappa', 0.5))
    retained_set = set(retained_positions)
    for position, (client_id, previous_score, current_score) in enumerate(
            zip(client_ids, previous_trust, client_scores)):
        if position in retained_set:
            new_score = rho * previous_score + (1.0 - rho) * current_score
        else:
            new_score = kappa * previous_score
        new_score = min(max(float(new_score), 0.0), 1.0)
        trust_memory[int(client_id)] = new_score
        updated_trust.append(new_score)

    retained_trust = [updated_trust[position] for position in retained_positions]
    aggregation_weights = _normalize_or_uniform(retained_trust)
    aggregated = _trust_weighted_delta_update(
        global_weights, local_weights, retained_positions, aggregation_weights)

    state['pritrust_previous_global_weights'] = copy.deepcopy(global_weights)
    state['pritrust_round'] = round_number
    state['pritrust_last_audited_layers'] = list(audited_keys)

    retained_clients = [int(client_ids[position])
                        for position in retained_positions]
    return aggregated, {
        'defense': PRITRUST_FL,
        'selected_count': len(retained_positions),
        'selected_clients': retained_clients,
        'audited_layers': list(audited_keys),
        'median_score': median_score,
        'mad_score': mad_score,
        'filter_threshold': threshold,
        'min_score': min(client_scores),
        'max_score': max(client_scores),
        'min_trust': min(updated_trust),
        'max_trust': max(updated_trust),
    }


def _normalize_or_uniform(values):
    values = [max(float(value), 0.0) for value in values]
    total = float(sum(values))
    if total > 0:
        return [value / total for value in values]
    if not values:
        return []
    return [1.0 / len(values) for _ in values]


def _select_pritrust_audited_layers(args, keys, client_ids, round_number):
    layer_count = len(keys)
    requested_budget = getattr(args, 'pritrust_audit_layers', None)
    if requested_budget is None:
        audit_budget = max(1, int(math.ceil(0.5 * layer_count)))
    else:
        audit_budget = int(requested_budget)
    audit_budget = min(max(audit_budget, 1), layer_count)

    seed_material = _pritrust_round_seed(args, client_ids, round_number)
    scored_layers = []
    for position, key in enumerate(keys):
        material = '{}|{}|{}'.format(seed_material, position, key)
        digest = hashlib.sha256(material.encode('utf-8')).digest()
        score = int.from_bytes(digest, byteorder='big', signed=False)
        scored_layers.append((score, position, key))

    selected = sorted(scored_layers, reverse=True)[:audit_budget]
    selected_positions = sorted(position for _, position, _ in selected)
    return [keys[position] for position in selected_positions]


def _pritrust_round_seed(args, client_ids, round_number):
    participant_serialization = ','.join(
        str(int(client_id)) for client_id in sorted(client_ids))
    base_seed = getattr(args, 'seed', None)
    security_bits = int(getattr(args, 'pritrust_security_bits', 128))
    material = '{}|{}|{}|{}'.format(
        base_seed, security_bits, round_number, participant_serialization)
    return hashlib.sha256(material.encode('utf-8')).hexdigest()


def _pritrust_consistency_scores(args, global_weights,
                                 previous_global_weights, local_weights,
                                 audited_keys, participant_weights):
    client_count = len(local_weights)
    score_sums = [0.0 for _ in range(client_count)]
    eps = 1e-12

    for key in audited_keys:
        deltas = _layer_delta_vectors(local_weights, global_weights, key)
        layer_norms = torch.sum(deltas * deltas, dim=1)
        weight_tensor = torch.tensor(
            participant_weights, dtype=deltas.dtype, device=deltas.device)
        avg_norm = float(torch.sum(weight_tensor * layer_norms).item())

        spatial_anchor = torch.sum(
            deltas * weight_tensor.view(-1, 1), dim=0)
        spatial_norm = float(torch.dot(spatial_anchor, spatial_anchor).item())

        temporal_anchor = None
        temporal_norm = 0.0
        if previous_global_weights is not None and key in previous_global_weights:
            temporal_anchor = (
                global_weights[key].detach().to(device='cpu',
                                                dtype=torch.float32) -
                previous_global_weights[key].detach().to(
                    device='cpu', dtype=torch.float32)
            ).reshape(-1)
            temporal_norm = float(torch.dot(temporal_anchor,
                                            temporal_anchor).item())

        for position in range(client_count):
            update = deltas[position]
            update_norm = float(layer_norms[position].item())
            indicators = []
            indicators.append(
                1.0 if _passes_amplitude_band(
                    args, update_norm, avg_norm) else 0.0)

            if temporal_anchor is not None and temporal_norm > eps:
                projection = float(torch.dot(update, temporal_anchor).item())
                distance = update_norm + temporal_norm - 2.0 * projection
                indicators.append(1.0 if projection >= 0.0 else 0.0)
                indicators.append(
                    1.0 if distance <= (
                        float(getattr(args, 'pritrust_theta_tem', 1.5)) *
                        temporal_norm) else 0.0)

            if spatial_norm > eps:
                projection = float(torch.dot(update, spatial_anchor).item())
                distance = update_norm + spatial_norm - 2.0 * projection
                indicators.append(1.0 if projection >= 0.0 else 0.0)
                indicators.append(
                    1.0 if distance <= (
                        float(getattr(args, 'pritrust_theta_spa', 1.5)) *
                        spatial_norm) else 0.0)

            score_sums[position] += sum(indicators) / len(indicators)

    audited_count = float(len(audited_keys))
    return [score_sum / audited_count for score_sum in score_sums]


def _layer_delta_vectors(local_weights, global_weights, key):
    global_value = global_weights[key].detach().to(
        device='cpu', dtype=torch.float32)
    return torch.stack([
        (weights[key].detach().to(device='cpu', dtype=torch.float32) -
         global_value).reshape(-1)
        for weights in local_weights
    ], dim=0)


def _passes_amplitude_band(args, update_norm, average_norm):
    alpha_min = float(getattr(args, 'pritrust_alpha_min', 0.5))
    alpha_max = float(getattr(args, 'pritrust_alpha_max', 1.5))
    lower = alpha_min * average_norm
    upper = alpha_max * average_norm
    return lower <= update_norm <= upper


def _adaptive_filter_positions(args, scores):
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    median_score = float(torch.median(score_tensor).item())
    deviations = torch.abs(score_tensor - median_score)
    mad_score = float(torch.median(deviations).item())
    gamma = float(getattr(args, 'pritrust_gamma', 1.5))
    if mad_score > 0.0:
        threshold = median_score - gamma * mad_score
    else:
        threshold = median_score

    retained = [
        position for position, score in enumerate(scores)
        if score >= threshold
    ]
    if not retained:
        retained = [
            position for position, score in enumerate(scores)
            if score >= median_score
        ]
    if not retained:
        best_position = max(range(len(scores)), key=lambda idx: scores[idx])
        retained = [best_position]
    return retained, threshold, median_score, mad_score


def _trust_weighted_delta_update(global_weights, local_weights,
                                 retained_positions, aggregation_weights):
    result = copy.deepcopy(global_weights)
    best_position = retained_positions[
        max(range(len(aggregation_weights)),
            key=lambda idx: aggregation_weights[idx])]
    for key, global_value in global_weights.items():
        if torch.is_floating_point(global_value):
            aggregate_delta = torch.zeros_like(global_value)
            for retained_position, weight in zip(retained_positions,
                                                 aggregation_weights):
                local_value = local_weights[retained_position][key].to(
                    device=global_value.device, dtype=global_value.dtype)
                aggregate_delta += (local_value - global_value) * float(weight)
            result[key] = global_value.clone() + aggregate_delta
        else:
            result[key] = local_weights[best_position][key].clone()
    return result
