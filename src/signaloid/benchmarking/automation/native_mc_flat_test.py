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
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from unittest.mock import patch

from signaloid.distributional.distributional import DistributionalValue

from signaloid.benchmarking.automation import sample_generator
from signaloid.benchmarking.automation.sample_generator import (
    _run_distribution_mc_flat,
    _run_scalar_mc_flat,
)
from signaloid.benchmarking.types import BenchmarkingVariable
from signaloid.benchmarking.config import EquivMC, VariableTypes
from signaloid.benchmarking.distribution_helpers.collapse import (
    _collapse_asymptotically_optimal_w1,
)

SAMPLE_GEN_MODULE = "signaloid.benchmarking.automation.sample_generator"


def _make_dist_var(name: str, cla: str) -> BenchmarkingVariable:
    return BenchmarkingVariable(
        name=name,
        description=name,
        type=VariableTypes.DISTRIBUTION,
        cla=cla,
    )


def _make_scalar_var(name: str, cla: str) -> BenchmarkingVariable:
    return BenchmarkingVariable(
        name=name,
        description=name,
        type=VariableTypes.SCALAR,
        cla=cla,
    )


class TestRunDistributionMcFlat(unittest.TestCase):
    """Per-variable aggregation, ordering, and error semantics of
    ``_run_distribution_mc_flat``."""

    def setUp(self) -> None:
        # Common kwargs the flat helpers consume.
        self.sample_gen_kwargs: dict[str, Any] = dict(
            n_processors=2,
            path_to_application="/tmp/fake-app",
            native_executable_name="fake-exec",
            native_executable_dir="/tmp/fake-app",
            demo_cli_args="--demo",
        )

    def test_aggregates_per_variable_unweighted(self) -> None:
        """Each variable should receive the union of its own chunks only."""
        var_a = _make_dist_var("a", "-S 1")
        var_b = _make_dist_var("b", "-S 2")
        benchmarking_variables = [var_a, var_b]

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            _index, sub_size, cla = work_item
            tag = 1.0 if "-S 1" in cla else 2.0
            return [tag] * sub_size

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_distribution_mc_flat(
                dist_vars=[(0, var_a), (1, var_b)],
                size=10,
                weighted_samples=False,
                num_weighted_samples=0,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        self.assertEqual(var_a.distribution_samples.values, [1.0] * 10)
        self.assertEqual(var_b.distribution_samples.values, [2.0] * 10)
        self.assertEqual(var_a.distribution_samples.weights, [])
        self.assertEqual(var_b.distribution_samples.weights, [])

    def test_chunks_large_size_and_orders_by_chunk_index(self) -> None:
        """Per-variable values must be concatenated in chunk-index order
        even though chunks complete out of order."""
        var = _make_dist_var("a", "-S 1")
        benchmarking_variables = [var]

        # Patch the chunk size down so the test exercises multi-chunk
        # ordering without allocating multi-million-element lists.
        chunk_size = 4
        total_size = 10  # 4 + 4 + 2 -> three chunks, tail smaller than full.

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            _index, sub_size, _cla = work_item
            # Encode chunk position via the sub_size: full/full/tail.
            if sub_size == chunk_size:
                return [1.0] * sub_size
            return [2.0] * sub_size

        with patch(f"{SAMPLE_GEN_MODULE}._MC_CHUNK_SIZE", chunk_size), patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_distribution_mc_flat(
                dist_vars=[(0, var)],
                size=total_size,
                weighted_samples=False,
                num_weighted_samples=0,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        self.assertEqual(len(var.distribution_samples.values), total_size)
        # First two chunks (chunk_idx 0 and 1) are full-sized, then the tail.
        head_len = 2 * chunk_size
        tail_len = total_size - head_len
        self.assertEqual(var.distribution_samples.values[:head_len], [1.0] * head_len)
        self.assertEqual(var.distribution_samples.values[head_len:], [2.0] * tail_len)

    def test_passes_globally_unique_index_to_simulation(self) -> None:
        """Temp-dir/error indices must disambiguate across variables."""
        var_a = _make_dist_var("a", "")
        scalar_filler = _make_scalar_var("filler", "")
        var_b = _make_dist_var("b", "")
        # Sparse var indices: dist vars sit at positions 0 and 2 in the
        # parent variable list, so the test catches code that confuses
        # ``var_idx`` with a dense enumeration.
        benchmarking_variables = [var_a, scalar_filler, var_b]

        seen_indices: list[Any] = []

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            index, sub_size, _cla = work_item
            seen_indices.append(index)
            return [0.0] * sub_size

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_distribution_mc_flat(
                dist_vars=[(0, var_a), (2, var_b)],
                size=5,
                weighted_samples=False,
                num_weighted_samples=0,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        # All indices distinct and var_idx is encoded so failures point at
        # the right variable even when chunk_idx repeats.
        self.assertEqual(sorted(seen_indices), ["v0_c0", "v2_c0"])

    def test_uses_demo_cli_args_prefix(self) -> None:
        """``demo_cli_args`` should prefix every variable's CLA."""
        var = _make_dist_var("a", "-S 1")
        benchmarking_variables = [var]
        seen_cla: list[Any] = []

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            _index, sub_size, cla = work_item
            seen_cla.append(cla)
            return [0.0] * sub_size

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_distribution_mc_flat(
                dist_vars=[(0, var)],
                size=3,
                weighted_samples=False,
                num_weighted_samples=0,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        self.assertEqual(seen_cla, ["--demo -S 1"])

    def test_zero_size_skips_with_warning(self) -> None:
        var = _make_dist_var("a", "")
        benchmarking_variables = [var]

        with patch(f"{SAMPLE_GEN_MODULE}._run_mc_simulation") as mock_sim, patch(
            f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor
        ), self.assertWarnsRegex(UserWarning, "must be > 0"):
            _run_distribution_mc_flat(
                dist_vars=[(0, var)],
                size=0,
                weighted_samples=False,
                num_weighted_samples=0,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        mock_sim.assert_not_called()
        self.assertEqual(var.distribution_samples.values, [])

    def test_raises_when_no_samples_returned(self) -> None:
        """If every chunk for a variable returns empty, finalize must raise."""
        var = _make_dist_var("a", "")
        benchmarking_variables = [var]

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            return []

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            with self.assertRaisesRegex(RuntimeError, "No MC samples collected"):
                _run_distribution_mc_flat(
                    dist_vars=[(0, var)],
                    size=5,
                    weighted_samples=False,
                    num_weighted_samples=0,
                    benchmarking_variables=benchmarking_variables,
                    **self.sample_gen_kwargs,
                )

    def test_raises_when_simulation_chunk_raises(self) -> None:
        """Partial-failure must propagate — silent under-sampling would
        corrupt the downstream database with strictly fewer samples than
        ``size`` for the affected variable."""
        var_a = _make_dist_var("a", "-S 1")
        var_b = _make_dist_var("b", "-S 2")
        benchmarking_variables = [var_a, var_b]

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            _index, sub_size, cla = work_item
            if "-S 2" in cla:
                raise RuntimeError("synthetic native-MC failure")
            return [1.0] * sub_size

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            with self.assertRaisesRegex(RuntimeError, "MC simulation failed"):
                _run_distribution_mc_flat(
                    dist_vars=[(0, var_a), (1, var_b)],
                    size=5,
                    weighted_samples=False,
                    num_weighted_samples=0,
                    benchmarking_variables=benchmarking_variables,
                    **self.sample_gen_kwargs,
                )

    def test_weighted_samples_collapse_per_variable(self) -> None:
        """With weighted_samples=True, each variable should end up with
        positions/weights set via ``set_weighted_values``, independently."""
        var_a = _make_dist_var("a", "-S 1")
        var_b = _make_dist_var("b", "-S 2")
        benchmarking_variables = [var_a, var_b]

        def fake_sim(work_item: Any, *_args: Any, **_kwargs: Any) -> list[float]:
            _index, sub_size, cla = work_item
            base = 10.0 if "-S 1" in cla else 20.0
            return [base + i * 0.001 for i in range(sub_size)]

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_mc_simulation", side_effect=fake_sim
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_distribution_mc_flat(
                dist_vars=[(0, var_a), (1, var_b)],
                size=200,
                weighted_samples=True,
                num_weighted_samples=8,
                benchmarking_variables=benchmarking_variables,
                **self.sample_gen_kwargs,
            )

        # Both variables converted independently. Sample magnitudes are disjoint
        # between var_a (~10) and var_b (~20), so verify each variable's
        # collapsed support comes from its own samples only.
        self.assertTrue(
            len(var_a.distribution_samples.values)
            == len(var_a.distribution_samples.weights)
            > 0
        )
        self.assertTrue(
            len(var_b.distribution_samples.values)
            == len(var_b.distribution_samples.weights)
            > 0
        )
        self.assertTrue(
            all(9.5 <= v <= 11.0 for v in var_a.distribution_samples.values)
        )
        self.assertTrue(
            all(19.5 <= v <= 21.0 for v in var_b.distribution_samples.values)
        )


class TestRunScalarMcFlat(unittest.TestCase):
    """Per-variable scalar aggregation, repetition, and error semantics of
    ``_run_scalar_mc_flat``."""

    def setUp(self) -> None:
        # Common kwargs the flat helpers consume.
        sample_gen_kwargs: dict[str, Any] = dict(
            n_processors=2,
            path_to_application="/tmp/fake-app",
            native_executable_name="fake-exec",
            native_executable_dir="/tmp/fake-app",
            demo_cli_args="--demo",
        )
        # Common kwargs for ``_run_scalar_mc_flat``.
        self.scalar_gen_kwargs: dict[str, Any] = dict(
            sample_gen_kwargs,
            n_adversaries=1,
            ground_truth_size=50,
            adversary_max_size_scalar=100,
            use_clt=False,
        )

    def test_aggregates_per_variable(self) -> None:
        """Each variable should accumulate only its own scalar outputs."""
        var_a = _make_scalar_var("a", "-S 1")
        var_b = _make_scalar_var("b", "-S 2")
        benchmarking_variables = [var_a, var_b]

        def fake_scalar(
            _path: Any,
            _exec: Any,
            _exec_dir: Any,
            cla: str,
            sample_size: int,
            _run_id: Any,
        ) -> tuple[int, float]:
            return sample_size, 1.0 if "-S 1" in cla else 2.0

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_scalar_native", side_effect=fake_scalar
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_scalar_mc_flat(
                scalar_vars=[(0, var_a), (1, var_b)],
                n_steps_scalar=3,
                ground_truth=False,
                benchmarking_variables=benchmarking_variables,
                **self.scalar_gen_kwargs,
            )

        # var_a should only see 1.0 outputs, var_b only 2.0
        for size, vals in var_a.distribution_samples.scalar_output_dict.items():
            self.assertTrue(all(v == 1.0 for v in vals), f"var_a size {size}: {vals}")
        for size, vals in var_b.distribution_samples.scalar_output_dict.items():
            self.assertTrue(all(v == 2.0 for v in vals), f"var_b size {size}: {vals}")

    def test_ground_truth_uses_ground_truth_size_once(self) -> None:
        var = _make_scalar_var("a", "")
        benchmarking_variables = [var]

        seen_sizes: list[int] = []

        def fake_scalar(
            _path: Any,
            _exec: Any,
            _exec_dir: Any,
            _cla: Any,
            sample_size: int,
            _run_id: Any,
        ) -> tuple[int, float]:
            seen_sizes.append(sample_size)
            return sample_size, 7.0

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_scalar_native", side_effect=fake_scalar
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_scalar_mc_flat(
                scalar_vars=[(0, var)],
                n_steps_scalar=10,
                ground_truth=True,
                benchmarking_variables=benchmarking_variables,
                **self.scalar_gen_kwargs,
            )

        self.assertEqual(seen_sizes, [self.scalar_gen_kwargs["ground_truth_size"]])
        self.assertEqual(
            var.distribution_samples.scalar_output_dict,
            {self.scalar_gen_kwargs["ground_truth_size"]: [7.0]},
        )

    def test_repetitions_match_n_adversaries(self) -> None:
        """For non-ground-truth runs, every size is repeated n_adversaries times."""
        var = _make_scalar_var("a", "")
        benchmarking_variables = [var]
        # Override the fixture default for this test.
        scalar_gen_kwargs = dict(self.scalar_gen_kwargs, n_adversaries=4)

        def fake_scalar(
            _path: Any,
            _exec: Any,
            _exec_dir: Any,
            _cla: Any,
            sample_size: int,
            run_id: int,
        ) -> tuple[int, float]:
            return sample_size, float(run_id)

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_scalar_native", side_effect=fake_scalar
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_scalar_mc_flat(
                scalar_vars=[(0, var)],
                n_steps_scalar=3,
                ground_truth=False,
                benchmarking_variables=benchmarking_variables,
                **scalar_gen_kwargs,
            )

        for size, vals in var.distribution_samples.scalar_output_dict.items():
            self.assertEqual(
                len(vals), 4, f"size {size}: expected 4 reps, got {len(vals)}"
            )

    def test_skips_none_values(self) -> None:
        """``_run_scalar_native`` returning ``None`` for the value
        should not be appended."""
        var = _make_scalar_var("a", "")
        benchmarking_variables = [var]
        scalar_gen_kwargs = dict(self.scalar_gen_kwargs, n_adversaries=2)

        call_count = {"n": 0}

        def fake_scalar(
            _path: Any,
            _exec: Any,
            _exec_dir: Any,
            _cla: Any,
            sample_size: int,
            _run_id: Any,
        ) -> tuple[int, float | None]:
            call_count["n"] += 1
            # Drop every other value.
            value = None if call_count["n"] % 2 == 0 else 5.0
            return sample_size, value

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_scalar_native", side_effect=fake_scalar
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            _run_scalar_mc_flat(
                scalar_vars=[(0, var)],
                n_steps_scalar=2,
                ground_truth=False,
                benchmarking_variables=benchmarking_variables,
                **scalar_gen_kwargs,
            )

        for vals in var.distribution_samples.scalar_output_dict.values():
            self.assertTrue(all(v == 5.0 for v in vals))
            self.assertTrue(len(vals) <= 2)

    def test_no_work_items_returns_early(self) -> None:
        """Empty scalar_vars should not invoke the simulator."""
        with patch(f"{SAMPLE_GEN_MODULE}._run_scalar_native") as mock_scalar, patch(
            f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor
        ):
            _run_scalar_mc_flat(
                scalar_vars=[],
                n_steps_scalar=3,
                ground_truth=False,
                benchmarking_variables=[],
                **self.scalar_gen_kwargs,
            )

        mock_scalar.assert_not_called()

    def test_raises_when_scalar_run_raises(self) -> None:
        """A raising scalar future must propagate. Silent failure
        previously caused the variable's ``scalar_output_dict`` to be missing
        entries without operator visibility."""
        var = _make_scalar_var("a", "")
        benchmarking_variables = [var]
        scalar_gen_kwargs = dict(self.scalar_gen_kwargs, n_adversaries=2)

        call_count = {"n": 0}

        def fake_scalar(
            _path: Any,
            _exec: Any,
            _exec_dir: Any,
            _cla: Any,
            sample_size: int,
            _run_id: Any,
        ) -> tuple[int, float]:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("synthetic scalar failure")
            return sample_size, 5.0

        with patch(
            f"{SAMPLE_GEN_MODULE}._run_scalar_native", side_effect=fake_scalar
        ), patch(f"{SAMPLE_GEN_MODULE}.ProcessPoolExecutor", ThreadPoolExecutor):
            with self.assertRaisesRegex(RuntimeError, "Scalar MC run failed"):
                _run_scalar_mc_flat(
                    scalar_vars=[(0, var)],
                    n_steps_scalar=2,
                    ground_truth=False,
                    benchmarking_variables=benchmarking_variables,
                    **scalar_gen_kwargs,
                )


class TestSampleGeneratorModuleExports(unittest.TestCase):
    """The public free-function surface of ``sample_generator`` is importable."""

    def test_sample_generator_module_exports(self) -> None:
        """Smoke: verify the public free-function surface is importable."""
        self.assertTrue(callable(sample_generator.generate_database_native_mc))
        self.assertTrue(callable(sample_generator.generate_scalar_samples))
        self.assertTrue(callable(sample_generator._run_distribution_mc_flat))
        self.assertTrue(callable(sample_generator._run_scalar_mc_flat))
        self.assertTrue(callable(sample_generator._determine_scalar_sample_sizes))
        self.assertIsInstance(sample_generator._MC_CHUNK_SIZE, int)


def _scalar_var_with_predictions(
    predicted_sizes: list[int],
) -> BenchmarkingVariable:
    """Scalar variable whose EMCC predictions drive the pre-compute path."""
    variable = _make_scalar_var("scalar", "-S 0")
    variable.emcc_results.emcc_data = [
        {EquivMC.EMCC_PREDICTED: size} for size in predicted_sizes
    ]
    return variable


class TestDetermineScalarSampleSizes(unittest.TestCase):
    """Clamping, warning, and geometric-schedule behaviour of
    ``_determine_scalar_sample_sizes``."""

    def test_determine_scalar_sample_sizes_clamps_to_ground_truth(self) -> None:
        """Pre-compute-EMCC sizes above ground_truth_size clamp to it."""
        cases: list[tuple[list[int], int, list[int]]] = [
            # A single over-INT_MAX prediction clamps down to the cap.
            ([3_750_665_567], 10_000_000, [10_000_000]),
            # Mixed: only the over-cap predictions are clamped. The small one
            # survives. Two distinct over-cap values both clamp to the cap and
            # collapse via the sorted-set dedup.
            ([500, 3_750_665_567, 2_152_521_315], 10_000_000, [500, 10_000_000]),
            # Everything at or below the cap is left untouched.
            ([100, 5_000, 9_999_999], 10_000_000, [100, 5_000, 9_999_999]),
            # Boundary: a size exactly equal to the cap is NOT clamped
            # (the comparison is strictly-greater).
            ([10_000_000], 10_000_000, [10_000_000]),
        ]
        for predicted, ground_truth_size, expected in cases:
            with self.subTest(
                predicted=predicted,
                ground_truth_size=ground_truth_size,
                expected=expected,
            ):
                variable = _scalar_var_with_predictions(predicted)
                sizes = sample_generator._determine_scalar_sample_sizes(
                    variable=variable,
                    n_steps_scalar=10,
                    use_clt=True,
                    adversary_max_size_scalar=100,
                    ground_truth_size=ground_truth_size,
                )
                self.assertEqual(sizes, expected)

    def test_determine_scalar_sample_sizes_warns_with_details(self) -> None:
        """A clamp warns, naming the variable, the prediction, and the cap."""
        variable = _scalar_var_with_predictions([3_750_665_567])
        variable.description = "gg_cross_section"
        with self.assertWarns(UserWarning) as cm:
            sample_generator._determine_scalar_sample_sizes(
                variable=variable,
                n_steps_scalar=10,
                use_clt=True,
                adversary_max_size_scalar=100,
                ground_truth_size=10_000_000,
            )
        message = str(cm.warning)
        self.assertIn("3750665567", message)  # original (pre-clamp) prediction
        self.assertIn("gg_cross_section", message)  # the variable named
        self.assertIn("10000000", message)  # the cap it was clamped to

    def test_determine_scalar_sample_sizes_no_warn_within_bound(self) -> None:
        """No clamp warning when every prediction is within the bound."""
        variable = _scalar_var_with_predictions([100, 5_000])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            sizes = sample_generator._determine_scalar_sample_sizes(
                variable=variable,
                n_steps_scalar=10,
                use_clt=True,
                adversary_max_size_scalar=100,
                ground_truth_size=10_000_000,
            )
        self.assertEqual(sizes, [100, 5_000])
        self.assertFalse(any("clamping" in str(w.message) for w in recorded))

    def test_determine_scalar_sample_sizes_geometric_branch_ignores_cap(
        self,
    ) -> None:
        """The non-pre-compute geometric schedule is unaffected by the cap."""
        variable = _make_scalar_var("scalar", "-S 0")
        sizes = sample_generator._determine_scalar_sample_sizes(
            variable=variable,
            n_steps_scalar=5,
            use_clt=False,
            adversary_max_size_scalar=100,
            ground_truth_size=10,
        )
        # Geometric schedule spans 1..100; a ground_truth_size of 10 must
        # NOT clamp it (clamping only applies to the pre-compute branch).
        self.assertEqual(sizes[0], 1)
        self.assertEqual(sizes[-1], 100)
        self.assertTrue(max(sizes) > 10)


# Golden captured from the legacy local Distribution.collapse() before the
# migration. That Distribution class is not part of this package.
#
# Frozen golden, captured from the legacy analyses-side oracle before the
# migration by running, on `np.random.default_rng(12345).normal(loc=3.0,
# scale=1.5, size=5_000)` with n_dirac_deltas=16:
#
#     reference = Distribution.from_samples(samples)
#     reference.representation_type = RepresentationTypes.ASYMPTOTICALLY_OPTIMAL_W1
#     reference.representation_size = 16
#     reference = reference.collapse()
#     # reference.positions, reference.masses
#
# The legacy free-function helper was verified to reproduce these arrays exactly
# (`np.array_equal` True), and the relocated
# `signaloid.benchmarking.distribution_helpers.collapse._collapse_asymptotically_optimal_w1`
# was then verified to reproduce the same golden. So this test pins the
# relocated helper against the pre-migration `Distribution.collapse()` behaviour
# without importing the (project-uxhw-only) `Distribution` class.
_COLLAPSE_GOLDEN_N_DIRAC_DELTAS = 16
_COLLAPSE_GOLDEN_POSITIONS = np.array(
    [
        0.15123892294603908,
        1.0001846373663956,
        1.4828055862879468,
        1.8585344179779364,
        2.146328987738469,
        2.4234668203514094,
        2.64603234008446,
        2.866208282199121,
        3.0956121716323417,
        3.339269867384286,
        3.588948140745383,
        3.869471210253865,
        4.166571099311518,
        4.534450957209472,
        4.987797195677714,
        5.836204379853931,
    ]
)
_COLLAPSE_GOLDEN_MASSES = np.array(
    [
        0.0581533088149018,
        0.06431726150274462,
        0.06288896876050541,
        0.0665813095373724,
        0.059637233508816945,
        0.06557286657028133,
        0.05696562926616555,
        0.07041828174894837,
        0.05839853287387953,
        0.06422738707839659,
        0.06245163653588759,
        0.060347899205485445,
        0.06329965714293706,
        0.06540079792140441,
        0.06357280075481142,
        0.05776642877746152,
    ]
)


class TestCollapseAsymptoticallyOptimalW1(unittest.TestCase):
    """The relocated ``_collapse_asymptotically_optimal_w1`` free function must
    reproduce the legacy ``Distribution.collapse()`` golden and reject too-few
    deltas."""

    def test_collapse_asymptotically_optimal_w1_matches_distribution_collapse(
        self,
    ) -> None:
        """The free-function helper must reproduce the legacy ``Distribution.collapse()``
        result for the ASYMPTOTICALLY_OPTIMAL_W1 representation, producing the same
        positions/masses on a representative MC-style sample input.

        The legacy ``Distribution`` oracle is NOT imported here. It is not
        part of the benchmarking package.
        Instead the expected positions/masses are pinned as a frozen golden
        (``_COLLAPSE_GOLDEN_*``) captured from that legacy path before the migration.
        """
        rng = np.random.default_rng(12345)
        samples = rng.normal(loc=3.0, scale=1.5, size=5_000)
        n_dirac_deltas = _COLLAPSE_GOLDEN_N_DIRAC_DELTAS

        # Under test: the new free-function helper on a plain DistributionalValue.
        dv = DistributionalValue.from_samples(samples)
        collapsed = _collapse_asymptotically_optimal_w1(
            dv, n_dirac_deltas=n_dirac_deltas
        )

        # Tight tolerance, not exact equality: the golden was captured on one
        # platform, and the pure-numpy collapse (argsort / cumsum / np.interp) can
        # differ by ~1 ULP across platforms/BLAS (~1e-16). 1e-12 is far above that
        # float noise yet far below any real algorithmic change.
        np.testing.assert_allclose(
            collapsed.positions, _COLLAPSE_GOLDEN_POSITIONS, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            collapsed.masses, _COLLAPSE_GOLDEN_MASSES, rtol=1e-12, atol=1e-12
        )
        # Sanity: the collapse actually produced the requested support size.
        self.assertEqual(len(collapsed.positions), n_dirac_deltas)

    def test_collapse_asymptotically_optimal_w1_rejects_too_few_deltas(
        self,
    ) -> None:
        """n_dirac_deltas < 2 raises a clear ValueError (not an opaque IndexError)."""
        dv = DistributionalValue.from_samples(np.random.default_rng(0).normal(size=100))
        for n in (0, 1):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "n_dirac_deltas"):
                    _collapse_asymptotically_optimal_w1(dv, n_dirac_deltas=n)


if __name__ == "__main__":
    unittest.main()
