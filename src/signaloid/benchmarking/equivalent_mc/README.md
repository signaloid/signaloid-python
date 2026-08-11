# Equivalent Monte Carlo

This is a Python library for computing the Equivalent Monte Carlo count numbers
for each UxHw configuration using a Ground Truth Monte Carlo simulation and a
series of Adversary Monte Carlo simulations.


### Definition: Equivalent Monte Carlo Count

An Equivalent Monte Carlo Count  (EMCC) is always assigned per variable per UxHw
instance. The Equivalent Monte Carlo Count is the number of Monte Carlo
re-executions required such that the result will of equal or better accuracy
compared to the corresponding UxHw instance for the chosen distance metric and
reporting method. For a given UxHw instance, if we wish to compute the
equivalent Monte Carlo count with the Wasserstein-1 distance metric and the mean
reporting method, we calculate the maximum number of Monte Carlo samples such
that across ALL adversary MC simulations, the Wasserstein distance between the
UxHw instance and the ground truth is greater than the mean Wasserstein distance
of Monte Carlo. I.e., we have 100% empirical confidence that the average Monte
Carlo of that size beats UxHw.

### Other terms

- **Ground Truth:** Samples which act as the authoritative golden reference of
  what the correct result of the application (under uncertainty) is. These can
  be provided via random samples such as a large Monte Carlo simulation or
  through weighted samples obtained through an analytic formula or empirical
  method.
- **Adversary:** Samples from a Monte Carlo simulation that we use to create
  samples of simulated Monte Carlo adversaries for UxHw.

## Usage

Import `load_data_and_compute_equivalent_mc` in your code.


### `load_data_and_compute_equivalent_mc`

`load_data_and_compute_equivalent_mc(args)` loads the ground-truth, UxHw, and
adversary distributions from their databases, computes the equivalent Monte
Carlo count for each benchmarking variable, prints the results, and (optionally)
writes plots and a CSV. It takes a single `LoadDataComputeEquivalentMCArgs`
`TypedDict`. A representative subset of its keys:

```python
from signaloid.benchmarking import (
    LoadDataComputeEquivalentMCArgs,
    load_data_and_compute_equivalent_mc,
)

args: LoadDataComputeEquivalentMCArgs = {
    "benchmarking_variables": benchmarking_variables,   # list[BenchmarkingVariable]
    "ground_truth_database_path": "GroundTruth.db",
    "ground_truth_table_name": "MonteCarlo",
    "ground_truth_type": "MonteCarlo",
    "adversary_database_path": "Adversary.db",
    "adversary_table_name": "MonteCarlo",
    "adversary_size_step": 1,
    "adversary_size_min": 1,
    "adversary_size_max": None,
    "n_adversaries": 100,
    "uxhw_database_path": "UxHw.db",
    "uxhw_table_names": ["Evaluation"],
    "uxhw_ur_types": ["Athens"],
    "uxhw_ur_sizes": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "correlations": ["Disabled"],
    "distance_type": "Wasserstein-1",
    "reporting_methods": ["Mean"],
    "n_processes": 1,
    "use_clt": False,
    "use_adaptive_steps": False,
    "use_binned_uxhw": False,
    "auto_prefix": False,
    "output_file": None,
    "plot_distributions": False,
    "plot_comparison_distributions": False,
    "plot_adversary_distances": False,
    "plots_dir": "",
}

load_data_and_compute_equivalent_mc(args)
```

See the `LoadDataComputeEquivalentMCArgs` definition in
[`equivalent_mc_main.py`](./equivalent_mc_main.py) for the full, authoritative
set of keys and their types.

## Outputs
Information printed includes:
- Statistics on ground truth and adversarial samples
- Statistics on UxHw distributions
- Table of equivalent MC information including MC counts, distances and analytic predictions.

Files generated include:
- For each traced variable:
  - `*_adversary_distances.npy`: the NumPy array of distances between each adversary and the ground truth.
  - `*_uxhw_distances.csv`: a csv file containing the distances between UxHw and the ground truth.
- `*-equivalent_mc.csv`: a csv file containing the Equivalent MC counts for each UxHw configuration
- When comparison-distribution plotting is enabled, for each UxHw configuration's equivalent MC count there will be a plot produced called `*_ground_truth-*_mc_count-*-adversaries.png`.

## Auditing the quality of Monte Carlo data
A question when comparing against Monte Carlo is how many samples are enough for both the ground truth and adversary arrays. There is no universal answer to this. We use use the following guiding principle:
1. Enable adversary-distance plotting. If the data is of a good enough quality then the measured adversary distances should not deviate significantly from the predicted line. Significant deviations are usually due to a poor quality ground truth or an adversary array that is too small.
2. Use analytic formulas. If an analytic formula for the target distribution is known, then a ground truth formed from 1 million weighted samples from the exact PDF will almost certainly be of better quality than 10-100 million random samples (this is due to the relatively slow inverse square root convergence of MC).
3. Focus on smaller representation sizes. Despite a 1 million MC ground truth sample count being unsuitable for Athens256 and Athens512 in many cases, it is more than good enough for representation sizes below this and you can rely on the analytic estimates to extrapolate up with.
