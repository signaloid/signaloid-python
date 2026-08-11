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

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from signaloid.benchmarking.automation.analysis import compute_emcc_predictions
from signaloid.benchmarking.automation.database_generator import (
    generate_database_from_mc_samples,
    generate_database_from_weighted_samples,
)
from signaloid.benchmarking.config import (
    DistanceMetrics,
    VariableTypes,
    ReportingMethods,
    RepresentationTypes,
    BenchmarkingVariables,
    EquivMC,
)
from signaloid.benchmarking.equivalent_mc.equivalent_mc_main import (
    load_data_and_compute_equivalent_mc,
)
from signaloid.benchmarking.types import (
    BenchmarkingVariable,
    UxhwDistanceRecord,
)

from .conftest import (
    SyntheticUxhwConfig,
    SyntheticUxhwExpression,
    build_uxhw_database,
)

# Expressions used to synthesise the ground-truth + adversary databases. These
# mirror the (name, description) pairs the original committed ``inputs/*.db``
# fixtures carried.
_SYNTHETIC_EQUIV_MC_EXPRESSIONS = [
    ("outputDistributions[0]", "Stock Price at Maturity"),
    ("outputDistributions[1]", "Call Option"),
    ("outputDistributions[2]", "Put Option"),
]

test_1_dict = {
    "distance_type": DistanceMetrics.WASSERSTEIN_1,
    "gt_path": "inputs/ground-truth-weighted-samples.db",
    "pre_compute": True,
    "plots": True,
    "adaptive": True,
    "use_binned": True,
}
test_2_dict = {
    "distance_type": DistanceMetrics.WASSERSTEIN_2,
    "gt_path": "inputs/ground-truth-weighted-samples.db",
    "pre_compute": False,
    "plots": False,
    "adaptive": True,
    "use_binned": False,
}
test_3_dict = {
    "distance_type": DistanceMetrics.WASSERSTEIN_1,
    "gt_path": "inputs/ground-truth-weighted-samples.db",
    "pre_compute": False,
    "plots": False,
    "adaptive": False,
    "use_binned": True,
}
config_options: list[dict[str, Any]] = [test_1_dict, test_2_dict, test_3_dict]


def _build_synthetic_equiv_mc_dbs(directory: Path) -> dict[str, str]:
    """Generate tiny ground-truth + adversary SQLite DBs for the equiv-MC smoke
    test, replacing the large committed ``inputs/*.db`` fixtures.

    Uses the package's own ``database_generator`` writers so the schema always
    matches the loaders. The data is synthetic, seeded and small: the equiv-MC
    smoke test asserts only that the pipeline runs end to end and produces
    structurally valid EMCC output, not specific numbers, so fidelity to the
    original captured distributions is unnecessary.
    """
    rng = np.random.default_rng(20260623)
    gt_variables: list[BenchmarkingVariable] = []
    adversary_variables: list[BenchmarkingVariable] = []
    for index, (name, description) in enumerate(_SYNTHETIC_EQUIV_MC_EXPRESSIONS):
        centre = 100.0 + 10.0 * index
        # Ground truth: a small weighted (positions, masses) distribution.
        positions = np.linspace(centre - 45.0, centre + 45.0, 64)
        masses = np.exp(-0.5 * ((positions - centre) / 15.0) ** 2)
        masses = masses / masses.sum()
        gt = BenchmarkingVariable(
            name=name, description=description, value_id=f"synthetic-{index}"
        )
        gt.distribution_samples.set_weighted_values(
            [float(p) for p in positions], [float(m) for m in masses]
        )
        gt_variables.append(gt)
        # Adversary: a small Monte-Carlo sample pool for the same expression.
        samples = rng.normal(loc=centre, scale=15.0, size=5_000)
        adversary = BenchmarkingVariable(
            name=name, description=description, value_id=f"synthetic-{index}"
        )
        adversary.distribution_samples.set_values([float(s) for s in samples])
        adversary_variables.append(adversary)
    gt_path = directory / "ground-truth-weighted-samples.db"
    adversary_path = directory / "adversary.db"
    generate_database_from_weighted_samples(str(gt_path), gt_variables)
    generate_database_from_mc_samples(str(adversary_path), adversary_variables)
    return {"gt_path": str(gt_path), "adv_path": str(adversary_path)}


def _build_synthetic_tracing_db(directory: Path) -> str:
    """Generate a tiny Athens-16 UxHw tracing database for the equiv-MC smoke
    test, replacing the large committed ``inputs/tracing.db`` fixture.

    Carries the same three expressions (at the same Gaussian centres as the
    synthetic ground truth) under both correlation-tracking statuses, matching
    the shape the original fixture provided.
    """
    expressions = [
        SyntheticUxhwExpression(
            name=name,
            value_id=f"synthetic-{index}",
            centre=100.0 + 10.0 * index,
            scale=15.0,
        )
        for index, (name, _description) in enumerate(_SYNTHETIC_EQUIV_MC_EXPRESSIONS)
    ]
    configs = [
        SyntheticUxhwConfig(RepresentationTypes.ATHENS, 16, "Disabled"),
        SyntheticUxhwConfig(RepresentationTypes.ATHENS, 16, "Autocorrelation"),
    ]
    tracing_path = directory / "tracing.db"
    build_uxhw_database(str(tracing_path), EquivMC.TRACING_TABLE, expressions, configs)
    return str(tracing_path)


def _build_benchmarking_variables(
    uxhw_distance_file: str,
) -> list[BenchmarkingVariable]:
    """Build the three benchmarking variables and preload their UxHw distances
    from ``inputs/uxhw_distances.csv`` (the equiv-MC pipeline needs them)."""
    benchmarking_variables: list[BenchmarkingVariable] = [
        BenchmarkingVariable(
            name="outputDistributions[0]",
            type=VariableTypes.DISTRIBUTION,
            description="Stock Price at Maturity",
        ),
        BenchmarkingVariable(
            name="outputDistributions[1]",
            type=VariableTypes.DISTRIBUTION,
            description="Call Option",
        ),
        BenchmarkingVariable(
            name="outputDistributions[2]",
            type=VariableTypes.DISTRIBUTION,
            description="Put Option",
        ),
    ]

    for variable in benchmarking_variables:
        if not variable.uxhw_distances.records:
            print(
                "Warning: UxHw distance distribution data not found in Benchmark object."
            )
            print(f"Loading UxHw distance data from {uxhw_distance_file}")

            try:
                # Load and group data by variable name
                distance_df = pd.read_csv(uxhw_distance_file)
                distance_dict = distance_df.to_dict("records")

                # Create a lookup dict for faster access
                data_by_variable: dict = {}
                for record in distance_dict:
                    var_name = record[BenchmarkingVariables.VARIABLE_DESCRIPTION]
                    data_by_variable.setdefault(var_name, []).append(record)

                # Now USE the lookup dict to populate the variable
                variable_records = data_by_variable.get(variable.description, [])

                # Append all matching records to the list
                if variable_records:
                    for record in variable_records:
                        variable.uxhw_distances.records.append(
                            UxhwDistanceRecord(
                                uxhw_conf=record.get(BenchmarkingVariables.UXHW_CONF),
                                uxhw_distance=record.get(
                                    BenchmarkingVariables.UXHW_DISTANCE
                                ),
                                uxhw_binned_distance=record.get(
                                    BenchmarkingVariables.UXHW_BINNED_DISTANCE
                                ),
                            )
                        )
                else:
                    print(
                        f"Warning: No UxHw distance data found for '{variable.description}'"
                    )

            except FileNotFoundError:
                print(f"Error: File not found at {uxhw_distance_file}")
                print("Cannot continue without UxHw distances data. Terminating.")
                raise
    return benchmarking_variables


_ASYMPTOTIC_MEAN = 200.0
_ASYMPTOTIC_QUANTILE_95 = 200.0

# Deterministic fallback UxHw distance for any variable the CSV does not cover.
# Note `inputs/uxhw_distances.csv` lists "Stock Price At Maturity" (capital
# "At") which does not match the variable's "Stock Price at Maturity" (lowercase
# "at"), so that variable gets this record.
_FALLBACK_UXHW_CONF = "Athens-16"
_FALLBACK_UXHW_DISTANCE = 10.0


def _populate_emcc_prediction_inputs(
    benchmarking_variables: list[BenchmarkingVariable],
) -> None:
    """Populate the real inputs that ``compute_emcc_predictions`` consumes, so
    its loaders (``load_uxhw_distances`` → ``load_asymptotic_dist``) early-return
    without reading any file:

    - ``asymptotic_distribution`` — ``load_asymptotic_dist`` early-returns only
      when EVERY variable has ``quantile_95 is not None``, so set deterministic
      positive ``mean`` and ``quantile_95`` (the fields ``value_for`` reads for
      ``ReportingMethods.MEAN`` / ``ReportingMethods.QUANTILE_95``) on each.
    - ``uxhw_distances.records`` — already preloaded from the CSV by
      ``_build_benchmarking_variables``. For any variable the CSV does not cover
      (e.g. "Stock Price at Maturity"), append a single deterministic record so
      every variable has >= 1 record and ``load_uxhw_distances`` early-returns.

    With these populated, ``compute_emcc_predictions`` runs entirely on
    in-memory inputs and genuinely computes ``EMCC Predicted`` per
    variable x reporting method.
    """
    for variable in benchmarking_variables:
        variable.asymptotic_distribution.mean = _ASYMPTOTIC_MEAN
        variable.asymptotic_distribution.quantile_95 = _ASYMPTOTIC_QUANTILE_95
        if not variable.uxhw_distances.records:
            variable.uxhw_distances.records.append(
                UxhwDistanceRecord(
                    uxhw_conf=_FALLBACK_UXHW_CONF,
                    uxhw_distance=_FALLBACK_UXHW_DISTANCE,
                    uxhw_binned_distance=_FALLBACK_UXHW_DISTANCE,
                )
            )


class TestEquivMc(unittest.TestCase):
    """End-to-end smoke test for the equivalent-Monte-Carlo pipeline against
    synthetic ground-truth + adversary databases."""

    def test_equiv_mc(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        uxhw_distance_file = os.path.join(current_dir, "inputs/uxhw_distances.csv")

        for options in config_options:
            with self.subTest(
                distance_type=options["distance_type"],
                pre_compute=options["pre_compute"],
                plots=options["plots"],
                adaptive=options["adaptive"],
                use_binned=options["use_binned"],
            ):
                # Per-config isolation: a fresh temp dir + fresh synthetic GT /
                # adversary DBs + fresh benchmarking variables, with cwd moved
                # into the temp dir so PNG / CSV output from one config cannot
                # bleed into another. The input DB / CSV paths above are
                # absolute, so they are unaffected by the chdir.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    original_cwd = os.getcwd()
                    os.chdir(tmp_dir)
                    try:
                        dbs = _build_synthetic_equiv_mc_dbs(Path(tmp_dir))
                        gt_path = dbs["gt_path"]
                        adv_path = dbs["adv_path"]
                        tracing_path = _build_synthetic_tracing_db(Path(tmp_dir))
                        benchmarking_variables = _build_benchmarking_variables(
                            uxhw_distance_file
                        )
                        reporting_methods = [
                            ReportingMethods.MEAN,
                            ReportingMethods.QUANTILE_95,
                        ]
                        # Populate real inputs, then run the REAL
                        # compute_emcc_predictions so emcc_data (including "EMCC
                        # Predicted") is genuinely produced by production code.
                        # Its loaders early-return on the pre-populated
                        # asymptotic / UxHw data, so asymptotic_dist_file is
                        # never read (see _populate_emcc_prediction_inputs).
                        _populate_emcc_prediction_inputs(benchmarking_variables)
                        emcc_predictions_file = os.path.join(
                            tmp_dir, "emcc_predictions.csv"
                        )
                        asymptotic_dist_file = os.path.join(
                            tmp_dir, "asymptotic_distances.csv"
                        )
                        compute_emcc_predictions(
                            benchmarking_variables=benchmarking_variables,
                            use_binned_uxhw=options["use_binned"],
                            reporting_methods=reporting_methods,
                            output_data_file=emcc_predictions_file,
                            uxhw_distance_file=uxhw_distance_file,
                            asymptotic_dist_file=asymptotic_dist_file,
                            distance_type=options["distance_type"],
                        )
                        load_data_and_compute_equivalent_mc(
                            {
                                "ground_truth_database_path": gt_path,
                                "ground_truth_table_name": "WeightedSamples",
                                "benchmarking_variables": benchmarking_variables,
                                "adversary_database_path": adv_path,
                                "adversary_table_name": "MonteCarlo",
                                "adversary_size_step": 100,
                                "adversary_size_min": 1,
                                "adversary_size_max": 1000,
                                "uxhw_database_path": tracing_path,
                                "uxhw_table_names": [EquivMC.TRACING_TABLE],
                                "uxhw_ur_types": [RepresentationTypes.ATHENS],
                                "uxhw_ur_sizes": [16],
                                "correlations": [],
                                "distance_type": options["distance_type"],
                                "n_processes": 1,
                                "n_adversaries": 4,
                                "output_file": "equivalent_mc.csv",
                                "plot_comparison_distributions": options["plots"],
                                "use_clt": options["pre_compute"],
                                "auto_prefix": True,
                                "reporting_methods": reporting_methods,
                                "ground_truth_type": "WeightedSamples",
                                "plot_adversary_distances": options["plots"],
                                "use_adaptive_steps": options["adaptive"],
                                "plot_distributions": options["plots"],
                                "use_binned_uxhw": options["use_binned"],
                                "plots_dir": "",
                            }
                        )

                        self._assert_emcc_structure(benchmarking_variables)
                        if options["plots"]:
                            self._assert_plots_written(Path(tmp_dir))
                    finally:
                        os.chdir(original_cwd)

    def _assert_emcc_structure(
        self, benchmarking_variables: list[BenchmarkingVariable]
    ) -> None:
        """Assert that every variable produced structurally valid EMCC output:
        non-empty ``emcc_data`` with exactly one record per reporting method,
        each carrying a positive EMCC-predicted count and (when present) a
        well-formed ``% MC beats UxHw`` proportion."""
        # The pipeline was passed exactly these two reporting methods.
        reporting_methods = [ReportingMethods.MEAN, ReportingMethods.QUANTILE_95]
        for variable in benchmarking_variables:
            emcc_data = variable.emcc_results.emcc_data
            self.assertTrue(
                emcc_data,
                f"emcc_data is empty for variable '{variable.description}'",
            )
            # Exactly one record per reporting method.
            self.assertEqual(len(emcc_data), len(reporting_methods))
            observed_methods = sorted(
                str(record[EquivMC.REPORTING_METHOD]) for record in emcc_data
            )
            self.assertEqual(observed_methods, sorted(reporting_methods))

            for record in emcc_data:
                self.assertIn(EquivMC.EMCC_PREDICTED, record)
                emcc_predicted = record[EquivMC.EMCC_PREDICTED]
                self.assertGreaterEqual(float(emcc_predicted), 1.0)

                # The pipeline-under-test fills in the equivalent MC count.
                self.assertIn(EquivMC.EMCC, record)
                self.assertGreaterEqual(float(record[EquivMC.EMCC]), 1.0)

                if EquivMC.PERCENTAGE_MC_BEATS_UXHW in record:
                    proportion = float(record[EquivMC.PERCENTAGE_MC_BEATS_UXHW])
                    self.assertGreaterEqual(proportion, 0.0)
                    self.assertLessEqual(proportion, 1.0)

    def _assert_plots_written(self, output_dir: Path) -> None:
        """Assert at least one ``.png`` was written under the temp dir when
        plotting is enabled."""
        pngs = list(output_dir.rglob("*.png"))
        self.assertTrue(
            pngs,
            f"expected at least one .png under {output_dir}, found none",
        )


if __name__ == "__main__":
    unittest.main()
