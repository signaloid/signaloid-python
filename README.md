# Signaloid-Tools-Python
This repository contains a set of tools for parsing and plotting Signaloid distributional data. The code has been tested on MacOS-11 and Ubuntu-20.04 using python 3.11.

### `DistributionalValue` class:
This is the main class used to represent distributional values. You can parse distributional data from from hex strings (`Ux` strings) or byte buffers. You can find more details regarding Signaloid distributional values [here](https://docs.signaloid.io/docs/hardware-api/ux-data-format/).

### `PlotHistogramDiracDeltas` class:
Histogram plotter class. Exposes `plot_histogram_dirac_deltas()`, which takes a `DistributionalValue` list and plots each DistributionalValue as a histogram.

## Installation:
You can download the latest wheel package from [here](https://github.com/signaloid/Signaloid-DistributionalPlotting-Tools/releases) and install it via pip:
```bash
pip install signaloid-0.2.0-py3-none-any.whl
```
