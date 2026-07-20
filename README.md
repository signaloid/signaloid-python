# Signaloid Python Library and SDK

The Signaloid Python Library and SDK provides tools for interacting with 
applications that utilize Signaloid's UxHw® technology for distributional 
arithmetic. Use the library to analyze Ux Data values from the application. 


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
