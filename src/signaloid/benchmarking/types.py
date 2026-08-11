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

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from signaloid.distributional.distributional import DistributionalValue

from signaloid.benchmarking.config import Correlations, ReportingMethods, VariableTypes

# Only BenchmarkingVariable is part of the public API (it is the element type
# of LoadDataComputeEquivalentMCArgs["benchmarking_variables"] that callers
# construct). The remaining value/result types are populated on / read off a
# BenchmarkingVariable rather than constructed by callers, so they stay
# importable by explicit path but out of the advertised surface.
__all__ = [
    "BenchmarkingVariable",
]


@dataclass
class TaggedDistributionalValue:
    """
    A :class:`DistributionalValue` paired with the loader metadata the
    benchmarking surface needs.

    The metadata fields are populated directly from the SQLite columns by the
    ``equivalent_mc/load.py`` loaders, so nothing here reads them off a
    ``Distribution`` instance. ``dv`` is always a base
    :class:`DistributionalValue` (never the analyses-side ``Distribution``
    subclass), keeping this surface free of that dependency.

    Attributes:
        dv: The numeric distribution (positions, masses, mean, ...).
        representation_type: Uncertain-representation type string (e.g.
            ``"Athens"``, ``"MonteCarlo"``, ``"WeightedSamples"``).
        representation_size: Build size N for the representation. For Athens this
            is ``UR_Order``. Otherwise ``UR_Order_CoreLibrary``.
        correlation_tracking: Correlation-tracking status string from the DB
            (e.g. ``"Disabled"``, ``"Autocorrelation"``). ``None`` for
            ground-truth and adversary loads that carry no correlation metadata.
        mc_count: Monte Carlo sample count for scalar loads, ``None``
            otherwise.
    """

    dv: DistributionalValue
    representation_type: str | None = None
    representation_size: int | None = None
    correlation_tracking: str | None = None
    mc_count: int | None = None

    def __repr__(self) -> str:
        """
        Reproduce ``Distribution.__repr__`` so the config string (the
        ``UXHW_CONF`` key written to the EMCC CSV and used to join timing data)
        stays byte-identical.
        """
        base = f"{self.representation_type}-{self.representation_size}"
        # Disabled correlation tracking is the implicit default and is omitted
        # from the config string. Autocorrelation carries a suffix.
        if self.correlation_tracking == Correlations.AUTOCORRELATION:
            return f"{base}-{Correlations.AUTOCORRELATION}"
        return base


@dataclass
class DistributionSamples:
    """
    Per-variable distribution sample positions/weights and scalar Monte Carlo
    outputs. Owned by ``BenchmarkingVariable.distribution_samples``.
    """

    values: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    scalar_output_dict: dict[int, list[float]] = field(default_factory=dict)

    def set_values(self, values: list[float]) -> None:
        """
        Set distribution sample positions. Weights unchanged.

        Args:
            values: List of sample position values.
        """
        self.values = values

    def set_weighted_values(
        self,
        values: list[float],
        weights: list[float],
    ) -> None:
        """
        Set distribution sample positions and weights.

        Args:
            values: List of sample position values.
            weights: List of corresponding sample weights.
        """
        self.values = values
        self.weights = weights

    def empty_values(self) -> None:
        """
        Clear stored sample data (positions, weights, scalar outputs).
        """
        self.values = []
        self.weights = []
        self.scalar_output_dict = {}


@dataclass
class TimingMeasurements:
    """
    Per-variable timing measurements collected during the UxHw/MC/native
    pipeline phases. Owned by ``BenchmarkingVariable.timing_measurements``.
    """

    measurement_dict: dict[str, dict[str, float]] = field(default_factory=dict)

    def append(
        self,
        *,
        config: str,
        time: float,
        e2e_time: float,
        pin_dyn_inst_count: float,
        db_time: float = 0.0,
        db_dyn_inst_count: float = 0.0,
    ) -> None:
        """
        Record a timing measurement for a given configuration.

        Args:
            config: Configuration key (e.g. representation type string).
            time: In-application elapsed time in seconds.
            e2e_time: End-to-end elapsed time in seconds.
            pin_dyn_inst_count: Dynamic instruction count from PIN.
            db_time: Database-access time in seconds.
            db_dyn_inst_count: Dynamic instruction count for DB access.
        """
        dictionary: dict[str, float] = {}
        dictionary["In Application Time"] = time
        dictionary["Database Time"] = db_time
        dictionary["End-to-End Time"] = e2e_time
        dictionary["Database Dyn. Inst. Count"] = db_dyn_inst_count
        dictionary["PIN Dyn. Inst. Count"] = pin_dyn_inst_count

        self.measurement_dict[config] = dictionary


@dataclass
class EmccResults:
    """
    Per-variable EMCC (Equivalent Monte Carlo Count) analysis state. Owned by
    ``BenchmarkingVariable.emcc_results``.
    """

    equiv_mc_list: list[int] = field(default_factory=list)
    emcc_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UxhwDistanceRecord:
    """
    One UxHw configuration vs. ground-truth Wasserstein distance.

    ``uxhw_conf`` is a union: the producer pipeline
    (``analysis.compute_uxhw_distances``) stores a
    ``TaggedDistributionalValue``, while the CSV-round-tripped loader path
    (``measurement_loader.load_uxhw_distances``) stores its ``repr()`` string.
    Consumers handle both shapes.

    ``blow_up_reason`` marks a degraded (blown-up) representation: when not
    ``None`` the distances are set to ``inf`` and the config is reported as
    "blow-up / excluded" rather than dropped, so it stays in the full matrix
    (see ``BenchmarkingVariables.BLOW_UP_REASON``). In-memory only, not
    persisted to the CSVs.
    """

    uxhw_conf: "TaggedDistributionalValue | str"
    uxhw_distance: float
    uxhw_binned_distance: float | None = None
    blow_up_reason: str | None = None


@dataclass
class UxhwDistances:
    """
    Per-variable UxHw-distance records. Owned by
    ``BenchmarkingVariable.uxhw_distances``.
    """

    records: list[UxhwDistanceRecord] = field(default_factory=list)


@dataclass
class AsymptoticDistribution:
    """
    Per-variable asymptotic distribution data. Owned by
    ``BenchmarkingVariable.asymptotic_distribution``.

    Field shape varies by variable type:
    - Distribution-typed variables populate ``mean``,
      ``quantile_95``, ``quantile_99`` and ``samples``.
    - Scalar-typed variables populate ``mean``, ``quantile_95``,
      ``quantile_99``, ``is_normal`` and ``scale``.

    All fields default to ``None`` so a freshly-constructed instance
    can be filled in incrementally by the generator
    (``Benchmark.generate_asymptotic_distance_distributions``) or
    the loader (``measurement_loader.load_asymptotic_dist``).
    """

    mean: float | None = None
    quantile_95: float | None = None
    quantile_99: float | None = None
    # CDF value at ``mean``. Serialized to the asymptotic-distance
    # CSV and round-tripped back by ``load_asymptotic_dist``, but no
    # consumer reads this field at runtime.
    mean_quantile: float | None = None
    # Distribution-typed variables only.
    samples: np.ndarray | None = None
    # Scalar-typed variables only.
    is_normal: bool | None = None
    scale: float | None = None

    def value_for(self, reporting_method: str) -> float | None:
        """
        Look up the value associated with a ReportingMethods string.

        Supports the dynamic, reporting-method-keyed access pattern in
        ``compute_emcc_predictions``.

        Args:
            reporting_method: One of ``ReportingMethods.MEAN``,
                ``ReportingMethods.QUANTILE_95``, or
                ``ReportingMethods.QUANTILE_99``.

        Returns:
            The stored float for the requested method, or ``None`` if
            the field has not been populated yet.

        Raises:
            KeyError: If ``reporting_method`` is not a known method, so a
                typo does not silently return ``None``.
        """
        try:
            return {
                ReportingMethods.MEAN: self.mean,
                ReportingMethods.QUANTILE_95: self.quantile_95,
                ReportingMethods.QUANTILE_99: self.quantile_99,
            }[reporting_method]
        except KeyError as exc:
            raise KeyError(
                f"Unknown reporting method {reporting_method!r}; "
                f"expected one of {{Mean, Quantile-95, Quantile-99}}."
            ) from exc


class BenchmarkingVariable:
    """
    Holds all data for a single benchmarked variable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        value_id: str = "",
        program: str = "main",
        path: str = "",
        line_number: str = "",
        file_name: str = "main.c",
        type: str = VariableTypes.DISTRIBUTION,
        cla: str = "",
    ):
        """
        Initialise a BenchmarkingVariable.

        Args:
            name: The name of the variable as traced in the application.
            description: Human-readable description of the variable.
            value_id: Unique identifier for the traced value.
            program: Name of the sub-program owning the variable.
            path: Filesystem path context for the variable.
            line_number: Source line number at which the variable is traced.
            file_name: Source file name in which the variable is declared.
            type: Output type. One of VariableTypes constants.
            cla: Command-line arguments used to isolate this variable.
        """
        self.value_id = value_id
        self.name = name
        self.description = description
        self.program = program
        self.path = path
        self.file_name = file_name
        self.type = type
        self.line_number = line_number
        self.cla = cla
        self.command_line_arguments: str = ""
        self.timing_measurements: TimingMeasurements = TimingMeasurements()
        self.distribution_samples: DistributionSamples = DistributionSamples()
        self.emcc_results: EmccResults = EmccResults()
        self.asymptotic_distribution: AsymptoticDistribution = AsymptoticDistribution()
        self.formatted_description: str = re.sub(r"\s+", "-", self.description.lower())
        self.uxhw_distances: UxhwDistances = UxhwDistances()

    def __str__(self) -> str:
        """
        Return a string representation of the BenchmarkingVariable.
        """
        return (
            f"BenchmarkingVariable("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"type='{self.type}', "
            f"file='{self.file_name}:{self.line_number}', "
            f"program='{self.program}')"
        )
