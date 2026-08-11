#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

import unittest

from signaloid.distributional.distributional import DistributionalValue

from signaloid.benchmarking.config import VariableTypes
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    TaggedDistributionalValue,
)
from signaloid.benchmarking.equivalent_mc.equivalent_monte_carlo import (
    EquivalentMonteCarlo,
)


def _make_emcc_with_distances(
    distances_by_count: dict[int, list[float]],
) -> EquivalentMonteCarlo:
    """Build an EquivalentMonteCarlo with only ``_adversary_distances``
    populated, bypassing __init__ for focused unit-testing of the
    closest-key fallback. The plot methods read ``self.n_adversaries``
    via ``math.log2``, so seed it too."""
    emcc = object.__new__(EquivalentMonteCarlo)
    emcc._adversary_distances = distances_by_count
    emcc.n_adversaries = 5
    return emcc


def _scalar_tagged(samples: list[float], mc_count: int) -> TaggedDistributionalValue:
    """A scalar adversary/ground-truth carrier, as produced by the scalar
    loaders in ``load.py`` (DistributionalValue.from_samples + mc_count)."""
    return TaggedDistributionalValue(
        dv=DistributionalValue.from_samples(samples),
        mc_count=mc_count,
    )


class TestResolveAdversaryDistances(unittest.TestCase):
    """``_resolve_adversary_distances`` falls back to the closest sampled key."""

    def test_resolve_adversary_distances_exact_match(self) -> None:
        """An exact key returns its list unchanged."""
        emcc = _make_emcc_with_distances({100: [1.0, 2.0], 500: [3.0]})

        self.assertEqual(emcc._resolve_adversary_distances(100), [1.0, 2.0])
        self.assertEqual(emcc._resolve_adversary_distances(500), [3.0])

    def test_resolve_adversary_distances_falls_back_to_closest_key(self) -> None:
        """A missing key resolves to the closest sampled adversary size.

        Regression for the bug surfaced on the Rendering Importance
        Sampling demo: a predicted EMCC of 1363 raised ``KeyError`` because
        the adaptive adversary stepping never sampled exactly 1363. The
        helper must pick the nearest sampled key instead.
        """
        emcc = _make_emcc_with_distances({1000: [1.0], 1500: [2.0], 2000: [3.0]})

        # 1363 is closer to 1500 than to 1000 or 2000.
        self.assertEqual(emcc._resolve_adversary_distances(1363), [2.0])

    def test_resolve_adversary_distances_ties_to_lower_key(self) -> None:
        """``min`` with ``abs`` keys ties to the first key encountered."""
        emcc = _make_emcc_with_distances({100: [1.0], 200: [2.0]})

        # 150 is equidistant from 100 and 200.
        # min() takes the first seen.
        result = emcc._resolve_adversary_distances(150)
        self.assertIn(result, ([1.0], [2.0]))

    def test_resolve_adversary_distances_empty_returns_none(self) -> None:
        """No adversaries have been computed yet → None, no exception."""
        emcc = _make_emcc_with_distances({})

        self.assertIsNone(emcc._resolve_adversary_distances(1))
        self.assertIsNone(emcc._resolve_adversary_distances(50000))


class TestPlotSkipsWhenNoDistances(unittest.TestCase):
    """Plot helpers bail before matplotlib work when no distances exist."""

    def test_plot_equiv_mc_vs_brownian_bridge_skips_when_no_distances(self) -> None:
        """``_plot_equiv_mc_vs_brownian_bridge`` must not raise when
        ``_adversary_distances`` is empty — it should bail before any
        matplotlib work."""
        emcc = _make_emcc_with_distances({})

        emcc._plot_equiv_mc_vs_brownian_bridge(
            mc_count=1363,
            brownian_bridge_integrals=[0.0, 1.0, 2.0],
            prefix="unused",
        )

    def test_plot_equiv_mc_vs_half_norm_skips_when_no_distances(self) -> None:
        """``_plot_equiv_mc_vs_half_norm`` must not raise when
        ``_adversary_distances`` is empty."""
        emcc = _make_emcc_with_distances({})

        emcc._plot_equiv_mc_vs_half_norm(mc_count=1363, prefix="unused")


class TestComputeDistanceDataScalar(unittest.TestCase):
    """Scalar variables with a single-sample adversary must not crash."""

    def test_compute_distance_data_scalar_single_sample_adversary_no_raise(
        self,
    ) -> None:
        """Regression: a scalar variable whose ``adversary_mc[0]`` carries a
        single sample must not crash ``compute_distance_data``.

        A single-sample adversary makes ``adversary_size_max`` collapse to
        ``len(positions) - 1 == 0``. The scalar branch of
        ``_compute_adversary_distances`` never reads the adversary-size array,
        but the array was still generated unconditionally — driving the
        adaptive-step path into ``np.log(0)`` (``math.ceil(-inf)`` raises
        ``OverflowError``). ``_generate_adversary_size_array`` now short-circuits
        to an empty array for non-distribution variables.
        """
        variable = BenchmarkingVariable(
            name="scalarOutput",
            description="Scalar Output",
            type=VariableTypes.SCALAR,
        )
        emcc = EquivalentMonteCarlo(
            ground_truth=_scalar_tagged([1.0], mc_count=1),
            uxhw_data=[_scalar_tagged([1.1], mc_count=1)],
            variable=variable,
            # adversary_mc[0] carries a single sample => adversary_size_max == 0
            adversary_mc=[
                _scalar_tagged([1.2], mc_count=1),
                _scalar_tagged([0.9, 1.1], mc_count=2),
            ],
            n_processes=1,
            n_adversaries=1,
        )

        # adversary_size_max is the degenerate value that previously broke the
        # adaptive-step log so the guard must tolerate it.
        self.assertEqual(emcc.adversary_size_max, 0)

        # Adaptive steps is the path that previously hit log(0).
        emcc.compute_distance_data(use_adaptive_steps=True)

        # The scalar branch ignores the size array (empty for scalars).
        self.assertEqual(len(emcc.adversary_size_array), 0)
        # The scalar distance computation still ran against each adversary.
        self.assertTrue(emcc._adversary_distances)


if __name__ == "__main__":
    unittest.main()
