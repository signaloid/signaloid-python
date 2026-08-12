# Signaloid Python Library and SDK

The Signaloid Python Library and SDK provides tools for interacting with 
applications that utilize Signaloid's UxHw® technology for distributional 
arithmetic. Use the library to analyze Ux Data values from the application. 

Also, run benchmarking of applications to compare the performance with 
equivalent Monte Carlo methods.

![signaloid-python diagram](images/signaloid-external-illustration-signaloid-python-diagram-flat-light-withCR.png#gh-light-mode-only)
![signaloid-python diagram](images/signaloid-external-illustration-signaloid-python-diagram-flat-dark-withCR.png#gh-dark-mode-only)

## Requirements

The Signaloid Python Library and SDK requires Python 3.10 or later. See 
`pyproject.toml` for the full list of dependencies.

## Installation
Install `signaloid-python` package via pip (recommended):
```bash
python -m pip install signaloid-python
```


Install the latest version from the GitHub repository:
```bash
python -m pip install git+https://github.com/signaloid/signaloid-python
```

Alternatively, clone this repository and install from source with:
```bash
python -m pip install .
```

## Usage

### Benchmarking UxHw applications

Use the `signaloid-benchmarking` command-line tool to benchmark an application
running with UxHw against a Monte Carlo baseline. The following example
benchmarks for UxHw Core microarchitectures Athens and Jupiter for precisions 8,
16, and 32, for both types of correlation tracking.

```bash
python -m signaloid.benchmarking.automation \
    --path-to-application ./my-uxhw-app \
    --path-to-uxhw-sdk ~/project-uxhw-sdk \
    --path-to-pin ~/pin-external-4.2 \
    -u Athens Jupiter \
    -s 8 16 32 \
    -c Disabled Autocorrelation \
    -r Mean
```

The tool needs access to the Signaloid UxHw SDK to build the applications for
UxHw, and access to the Intel Pin tool for accurate benchmarking. Arguments
`-u/--representation-types`, `-s/--representation-sizes`,
`-c/--uncertainty-correlation_types`, `-r/--reporting-methods` can also be
supplied using a YAML file with `--config <file>`.

For details, see the package [README.md](src/signaloid/benchmarking/automation/README.md).

### Parsing Ux Data
Construct `DistributionalValue` Python objects by parsing 
[Ux Data](https://docs.signaloid.io/docs/uxhw-api/ux-data-format/) in 
Ux String or Ux Binary format.

```python
from signaloid.distributional.distributional import DistributionalValue

# Intermediate code which writes to ux_string and ux_binary_buffer
# ...

# Parse a Ux String
dist_value = DistributionalValue.parse(ux_string)

# Parse a Ux Binary buffer
dist_value = DistributionalValue.parse(ux_binary_buffer)
```

### Create Distribution Plots
Create plots to visualize distributional information by using the 
[`plot` function](./src/signaloid/distributional_information_plotting/plot_wrapper.py) 
with a `DistributionalValue` object containing Ux Data. The `plot` function is a 
wrapper function for the `PlotHistogramDiracDeltas` class for plotting a 
distributional value as a histogram with variable bin widths.

```python
from signaloid.distributional_information_plotting.plot_wrapper import plot

# Intermediate code which writes to ux_string
# ...

# Create distributional value object from Ux String
dist_value = DistributionalValue.parse(ux_string)
plot(dist_value)
```
