#   Copyright (c) 2024, Signaloid.
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

from typing import Any, Optional, Dict, Tuple
import matplotlib
import matplotlib.pyplot as plt
import math
from signaloid.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotHistogramDiracDeltas,
)


def printv(
        verbose: bool,
        formatString: str,
        *args) -> None:
    """
    A wrapper for `print()` that allows for controlled verbosity.

    Args:
        verbose: Flag that controls verbosity.
        formatString: Formatted string to print.
        args: Tuple holding values to be used in the formatted string.
    """
    if verbose:
        print(formatString % args)

    return


def plot(
    distribution: DistributionalValue,
    path: str = "./plot.png",
    plotting_resolution: int = 128,
    plot_expected_value_line: bool = True,
    no_special_y: bool = False,
    save: bool = False,
    verbose: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    x_label: Optional[str] = None,
    x_tick_label_rotation: Optional[float] = None,
    font_size: int = 20,
    matplotlib_rc_params_override: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Args:
        distribution: `DistributionalValue` to plot.
        path: Path to save the output if saving is enabled.
        plotting_resolution: Resolution of the plotted binning.
        plot_expected_value_line: Flag toggling whether the plot should have a vertical
            line at the expected value of the input distribution.
        no_special_y: Flag toggling the plotting of special values, e.g., `NaN`, `INF`, and `-INF`.
        save: Flag toggling if the plot should be saved to a file or just shown.
        verbose: Flag controlling printing verbosity.
        x_lim: Input x-axis limits for the plot.
        y_lim: Input y-axis limits for the plot.
        x_label: x-axis label.
        x_tick_label_rotation: Rotation of x-axis tick labels.
        font_size: Font size to use for the plot labels.
        matplotlib_rc_params_override: Dictionary to specify custom plotting parameters.
    Returns:
        `True` if successful, `False` else.
    """
    if (
        distribution is None or
        distribution.mean is None or
        distribution.UR_order is None
    ):
        printv(verbose, "Failed to load data")
        return False

    distribution.drop_zero_mass_positions()

    ph = PlotHistogramDiracDeltas()

    matplotlib_rcParams_update_defaults = {
        "font.size": font_size,
        "figure.facecolor": "FFFFFF30",
        "axes.facecolor": "FFFFFF30",
        "xtick.top": True,
        "ytick.right": True,
    }

    if matplotlib_rc_params_override is not None:
        matplotlib_rcParams_update_defaults.update(matplotlib_rc_params_override)

    matplotlib.rcParams.update(matplotlib_rcParams_update_defaults)

    axes_needed = 1
    any_special_value = False
    gridspec = None

    any_minusinf = any([math.isinf(x) and x < 0 for x in distribution.positions])
    any_inf = any([math.isinf(x) and x > 0 for x in distribution.positions])
    any_nan = any([math.isnan(x) for x in distribution.positions])
    printv(verbose, f"any_minusinf={any_minusinf}")
    printv(verbose, f"any_inf={any_inf}")
    printv(verbose, f"any_nan={any_nan}")

    if any_minusinf or any_inf or any_nan:
        axes_needed += 1
        any_special_value = True
        gridspec = {"width_ratios": [4.2, 1]}

    fig, axes = plt.subplots(
        nrows=1,
        ncols=axes_needed,
        sharey=False,
        figsize=(10 + (3 if any_special_value else 0), 6),
        gridspec_kw=gridspec,
    )

    if not any_special_value:
        # Force axes to be a list (Thanks pyplot for the wonderful semantics)
        axes = [axes]

    fig.sca(axes[0])
    printv(verbose, "Plotting...")
    max_value = 0.0
    any_finite_values = any([math.isfinite(x) for x in distribution.positions])
    if any_finite_values:
        # Set plot resolution to (N*2) where N is machine representation
        machine_representation = 2 ** math.floor(
            math.log2(distribution.UR_order)
        )  # type: ignore[arg-type]
        pr = min((machine_representation * 2), plotting_resolution)
        assert isinstance(ph, PlotHistogramDiracDeltas)
        (min_range, max_range, max_value) = ph.plot_histogram_dirac_deltas(
            [distribution],
            plotting_resolution=pr,
            colors=["#33A333", "#A569BD", "#F1C40F"],
        )

    # Default kwargs for plt.annotate
    annotation_default_args: Dict[str, Any] = {
        "xycoords": ("data", "axes fraction"),
        "textcoords": "offset points",
        "xytext": (3, 0),
        "fontsize": 16,
        "rotation": 90,
        "verticalalignment": "top",
        "parse_math": True,
    }

    # Plot mean value (if finite) and standard deviation (if available)
    if (
        plot_expected_value_line is True
        and distribution.mean is not None
        and math.isfinite(distribution.mean)
    ):

        axes[0].axvline(distribution.mean, lw=2, color="#29782d")
        plt.annotate(
            "$E(X)$",
            (distribution.mean, 0.9),
            color="#206024",
            **annotation_default_args,
        )

    if any_special_value:
        fig.sca(axes[1])
        ph.plot_special_values_barplot(distribution)

    printv(verbose, "Done plotting.")

    printv(verbose, "Adjusting plot...")
    for i, ax in enumerate(axes):
        if i == 0:
            if any_finite_values:
                # This prevents the plots failing if mean value is
                # incorrect and way off the range.
                if (
                    math.isnan(distribution.mean) or
                    math.isinf(distribution.mean)
                ):  # type: ignore[arg-type]
                    min_x = min_range
                    max_x = max_range
                else:
                    min_x = min(distribution.mean, min_range)
                    max_x = max(distribution.mean, max_range)

                range_spacing = 0.05 * (max_x - min_x)

                guessed_x_lim = (min_x - range_spacing, max_x + range_spacing)
                printv(
                    verbose, f"Would set x_lim to {guessed_x_lim} (override is {x_lim})"
                )

                if x_lim is not None:
                    ax.set_xlim(*x_lim)
                else:
                    ax.set_xlim(*guessed_x_lim)

                guessed_y_lim = (0, 1.1 * max_value)
                printv(
                    verbose, f"Would set y_lim to {guessed_y_lim} (override is {y_lim})"
                )

                if y_lim is not None:
                    ax.set_ylim(*y_lim)
                else:
                    ax.set_ylim(*guessed_y_lim)

            if x_label:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel("Distribution Support")

            ax.set_ylabel("Probability Density")
        else:
            ax.set_ylim(0, 1)
            if not no_special_y:
                ax.set_ylabel("Probability Amplitude")
            else:
                ax.set_yticklabels([])

        ax.grid(visible=True, which="major", color="#999999aa", linestyle="--")

        plt.subplots_adjust(
            left=0.16, bottom=0.14, right=1.0, top=0.9, hspace=0.0, wspace=0.05
        )

        ax.spines["right"].set_alpha(0.5)
        ax.spines["left"].set_alpha(0.5)
        ax.spines["top"].set_alpha(1)
        ax.spines["top"].set_linewidth(3)
        ax.spines["bottom"].set_linewidth(3)
        ax.tick_params(axis="x", which="major", width=2, length=10, direction="in")
        ax.tick_params(axis="y", which="major", width=1, length=4, direction="in")

        if x_tick_label_rotation is not None:
            plt.setp(
                ax.get_xticklabels(),
                rotation=x_tick_label_rotation,
                horizontalalignment="right",
            )

        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", width=1, length=5, direction="in")
        ax.tick_params(axis="y", which="minor", left=False, right=False)

    fig.tight_layout()

    if save:
        printv(verbose, f"Saving figure as `{path}`...")
        fig.savefig(path, format="png", bbox_inches="tight", pad_inches=0.2)
        printv(verbose, "Closing figure...")
        plt.close(fig=fig)
    else:
        plt.show()

    return True
