# signaloid-python
This repository contains a set of tools for parsing and plotting Signaloid distributional data in python. The code has been tested on MacOS-14 and Ubuntu-20.04 using python 3.11.

## Installation:
You can install the latest version of `signaloid-python` package via pip:
```bash
pip install git+https://github.com/signaloid/signaloid-python
```

## Parse `Ux` data
You can construct `DistributionalValue` objects by parsing `Ux` string or `Ux` bytes. You can find more details about the Signaloid `Ux` format [here](https://docs.signaloid.io/docs/hardware-api/ux-data-format/). Following is an example of parsing `Ux` strings and `Ux` bytes.

```python
from signaloid.distributional.distributional import DistributionalValue

...

# Parse a Ux string
distValue = DistributionalValue.parse(ux_string)

# Parse a Ux bytes buffer
distValue = DistributionalValue.parse(ux_bytes_buffer)
```

## Plot `DistributionalValue` objects
You can use the `PlotHistogramDiracDeltas` class for plotting a distributional value as a histogram with variable bin width. We also provide a wrapper function to assist plotting. You can use the `plot` function, which you can find [here](./src/signaloid/distributional_information_plotting/plot_wrapper.py), to easily plot a distributional value like in the following example:

```python
from signaloid.distributional_information_plotting.plot_wrapper import plot

...

# Create distributional value object from string
distValue = DistributionalValue.parse(ux_string)
plot(distValue)
```
