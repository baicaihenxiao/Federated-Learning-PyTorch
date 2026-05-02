#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from defenses import _pdfl, _shieldfl  # noqa: E402


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


if __name__ == '__main__':
    test_shieldfl_uses_previous_round_poisonous_baseline()
    test_shieldfl_removes_raw_update_magnitude_from_confidence()
    test_pdfl_uses_majority_cluster_similarity_weights()
    print('plaintext defense checks passed')
