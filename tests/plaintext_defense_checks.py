#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from defenses import (  # noqa: E402
    _pdfl,
    _pritrust_fl,
    _select_pritrust_audited_layers,
    _shieldfl,
)


def _state(values):
    return {'w': torch.tensor(values, dtype=torch.float32)}


def _assert_close(actual, expected):
    expected = torch.tensor(expected, dtype=torch.float32)
    if not torch.allclose(actual, expected, atol=1e-6):
        raise AssertionError('expected {}, got {}'.format(
            expected.tolist(), actual.tolist()))


def _assert_float_close(actual, expected):
    if abs(float(actual) - float(expected)) > 1e-6:
        raise AssertionError('expected {}, got {}'.format(expected, actual))


def _pritrust_args(**overrides):
    values = dict(
        pritrust_audit_layers=1,
        pritrust_c_norm=2.0,
        pritrust_zeta=0.1,
        pritrust_theta_tem=1.5,
        pritrust_theta_spa=1.5,
        pritrust_gamma=0.8,
        pritrust_r_max=0.3,
        pritrust_rho=0.7,
        pritrust_kappa=0.2,
        pritrust_security_bits=128,
        seed=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shieldfl_uses_previous_round_poisonous_baseline():
    args = SimpleNamespace()
    global_weights = _state([0.0, 0.0])
    state = {}

    first_round = [_state([1.0, 0.0]), _state([1.0, 0.0])]
    _shieldfl(args, global_weights, first_round, [1, 1], [0, 1], state)

    second_round = [
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
        _state([-1.0, 0.0]),
    ]
    aggregated, info = _shieldfl(
        args, global_weights, second_round, [1, 1, 1], [0, 1, 2], state)

    _assert_close(aggregated['w'], [1.0, 0.0])
    assert info['poisonous_baseline_client'] == 2
    assert info['min_confidence'] == 0.0
    assert info['max_confidence'] == 2.0


def test_shieldfl_removes_raw_update_magnitude_from_confidence():
    args = SimpleNamespace()
    global_weights = _state([0.0, 0.0])
    state = {'shieldfl_previous_aggregate': torch.tensor([1.0, 0.0])}
    local_weights = [
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
        _state([10.0, 0.0]),
    ]

    aggregated, info = _shieldfl(
        args, global_weights, local_weights, [1, 1, 1], [0, 1, 2], state)

    _assert_close(aggregated['w'], [1.0, 0.0])
    assert info['aggregation_scale'] == 1.0


def test_pdfl_uses_majority_cluster_similarity_weights():
    args = SimpleNamespace(pdfl_similarity_threshold=0.7)
    global_weights = _state([0.0, 0.0])
    local_weights = [
        _state([1.0, 0.0]),
        _state([0.8, 0.6]),
        _state([-1.0, 0.0]),
    ]

    aggregated, info = _pdfl(
        args, global_weights, local_weights, [1, 100, 1], [0, 1, 2])

    _assert_close(aggregated['w'], [0.9, 0.3])
    assert info['selected_clients'] == [0, 1]
    _assert_float_close(info['min_similarity_weight'], 0.9)
    _assert_float_close(info['max_similarity_weight'], 0.9)


def test_pritrust_prefilters_median_norm_outlier_before_aggregation():
    args = _pritrust_args()
    global_weights = _state([0.0, 0.0])
    local_weights = [
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
        _state([100.0, 0.0]),
    ]
    state = {}

    aggregated, info = _pritrust_fl(
        args, global_weights, local_weights, [1, 1, 1], [0, 1, 2], state)

    _assert_close(aggregated['w'], [1.0, 0.0])
    assert info['candidate_count'] == 2
    assert info['norm_prefiltered_count'] == 1
    assert info['norm_violation_counts'] == [0, 0, 1]
    assert info['selected_clients'] == [0, 1]
    _assert_float_close(state['pritrust_client_trust'][2], 0.2)


def test_pritrust_zero_mad_uses_top_r_tiebreakers():
    args = _pritrust_args(pritrust_r_max=0.5)
    global_weights = _state([0.0, 0.0])
    local_weights = [
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
        _state([1.0, 0.0]),
    ]
    state = {
        'pritrust_client_trust': {
            0: 0.1,
            1: 0.9,
            2: 0.4,
            3: 0.2,
        },
    }

    aggregated, info = _pritrust_fl(
        args, global_weights, local_weights, [1, 1, 1, 1],
        [0, 1, 2, 3], state)

    _assert_close(aggregated['w'], [1.0, 0.0])
    assert info['filter_mode'] == 'top_r_zero_mad'
    assert info['selected_clients'] == [1, 2]
    _assert_float_close(state['pritrust_client_trust'][0], 0.02)
    _assert_float_close(state['pritrust_client_trust'][1], 0.93)
    _assert_float_close(state['pritrust_client_trust'][2], 0.58)
    _assert_float_close(state['pritrust_client_trust'][3], 0.04)


def test_pritrust_weights_retained_clients_by_trust_and_sample_count():
    args = _pritrust_args()
    global_weights = _state([0.0, 0.0])
    local_weights = [
        _state([1.0, 0.0]),
        _state([1.5, 0.0]),
    ]
    state = {}

    aggregated, info = _pritrust_fl(
        args, global_weights, local_weights, [1, 3], [0, 1], state)

    _assert_close(aggregated['w'], [1.375, 0.0])
    assert info['selected_clients'] == [0, 1]


def test_pritrust_audit_selection_keeps_sentinel_tensors():
    args = _pritrust_args(pritrust_audit_layers=3)
    keys = [
        'conv1.weight',
        'layer1.0.conv1.weight',
        'layer2.0.conv1.weight',
        'linear.weight',
        'linear.bias',
    ]

    audited = _select_pritrust_audited_layers(args, keys, [4, 2, 9], 1)

    assert 'conv1.weight' in audited
    assert 'linear.weight' in audited
    assert 'linear.bias' in audited
    assert len(audited) == 3


if __name__ == '__main__':
    test_shieldfl_uses_previous_round_poisonous_baseline()
    test_shieldfl_removes_raw_update_magnitude_from_confidence()
    test_pdfl_uses_majority_cluster_similarity_weights()
    test_pritrust_prefilters_median_norm_outlier_before_aggregation()
    test_pritrust_zero_mad_uses_top_r_tiebreakers()
    test_pritrust_weights_retained_clients_by_trust_and_sample_count()
    test_pritrust_audit_selection_keeps_sentinel_tensors()
    print('plaintext defense checks passed')
