from __future__ import annotations

import numpy as np
import pytest
from graphix.command import M, N, CommandKind
from graphix.noise_models.noise_model import ApplyNoise

from veriphix.gaussian_region_noise_model import GaussianRegionNoiseModel
from veriphix.malicious_noise_model import DephasingNoise


def _make_model(node_probs, seed=0):
    return GaussianRegionNoiseModel(node_probs, rng=np.random.default_rng(seed))


class TestGaussianRegionNoiseModel:
    def test_rejects_out_of_range_probability(self) -> None:
        with pytest.raises(ValueError):
            _make_model({0: 1.2})
        with pytest.raises(ValueError):
            _make_model({0: -0.1})

    def test_node_keys_and_probs_are_cast(self) -> None:
        model = _make_model({np.int64(3): np.float64(0.5)})
        assert model.node_probs == {3: 0.5}
        assert all(isinstance(k, int) for k in model.node_probs)
        assert all(isinstance(v, float) for v in model.node_probs.values())

    def test_unlisted_node_never_flipped(self) -> None:
        model = _make_model({0: 1.0})
        out = model.command(M(node=7))
        assert out == [M(node=7)]

    def test_non_measurement_command_untouched(self) -> None:
        model = _make_model({3: 1.0})
        cmd = N(node=3)
        assert model.command(cmd) == [cmd]

    def test_prob_one_always_flips_before_measurement(self) -> None:
        model = _make_model({3: 1.0})
        for _ in range(20):
            out = model.command(M(node=3))
            assert len(out) == 2
            noise, meas = out
            assert isinstance(noise, ApplyNoise)
            assert isinstance(noise.noise, DephasingNoise)
            assert noise.noise.prob == 1
            assert noise.nodes == [3]
            # the original measurement command comes *after* the noise
            assert meas.kind == CommandKind.M
            assert meas.node == 3

    def test_prob_zero_never_flips(self) -> None:
        model = _make_model({3: 0.0})
        for _ in range(20):
            assert model.command(M(node=3)) == [M(node=3)]

    def test_draws_are_independent_per_call(self) -> None:
        # With p=0.5, repeated calls must produce a mix of flip / no-flip,
        # confirming the Bernoulli draw happens live (not cached once).
        model = _make_model({3: 0.5}, seed=1234)
        flips = [len(model.command(M(node=3))) == 2 for _ in range(200)]
        assert any(flips) and not all(flips)

    def test_empirical_flip_rate_matches_probability(self) -> None:
        p = 0.3
        model = _make_model({3: p}, seed=2024)
        n = 5000
        flips = sum(len(model.command(M(node=3))) == 2 for _ in range(n))
        assert abs(flips / n - p) < 0.03

    def test_multiple_nodes_independent_probabilities(self) -> None:
        model = _make_model({1: 0.0, 2: 1.0}, seed=7)
        assert model.command(M(node=1)) == [M(node=1)]
        assert len(model.command(M(node=2))) == 2

    def test_confuse_result_is_identity(self) -> None:
        model = _make_model({3: 1.0})
        assert model.confuse_result(M(node=3), 1) == 1
        assert model.confuse_result(M(node=3), 0) == 0

    def test_input_nodes_returns_no_noise(self) -> None:
        model = _make_model({3: 1.0})
        assert model.input_nodes([3, 4, 5]) == []
