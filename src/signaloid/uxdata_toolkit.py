#!/usr/bin/env python3

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

"""
Signaloid UxData Toolkit

A set of tools for working with Signaloid Ux data.

Requirements:
- pip install git+https://github.com/signaloid/signaloid-python
"""

import argparse
import sys
import traceback
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from signaloid.distributional.distributional import DistributionalValue
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import \
    PlotData
from signaloid.distributional_information_plotting.plot_wrapper import plot


def parse_arguments() -> argparse.Namespace:
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Set of tools for working with Signaloid Ux data.'
    )

    command_subparsers = parser.add_subparsers(
        dest="command",
        title="Command",
    )

    # Register all command parsers
    command_plot_parser(command_subparsers)

    # Parse the arguments
    args = parser.parse_args()
    return args


def command_plot_parser(command_subparsers: Any) -> None:
    plot_parser = command_subparsers.add_parser(
        "plot",
        help="Plot Signaloid Ux distributional data",
        description="Plot Signaloid Ux distributional data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s 0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
  %(prog)s 09168733bf9ad93f000100000000000000c7c72324c19ad93f01000000c7c72324c19ad93f0000000000000080
  %(prog)s -o output.png 0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
        """
    )

    plot_parser.add_argument(
        '-o', '--output',
        type=str,
        default='plot.png',
        help='Output PNG filename (default: plot.png)'
    )

    plot_parser.add_argument(
        'ux_data',
        type=str,
        help='Ux-string or Ux-bytes to plot (e.g., "90.6Ux04000000...")',
    )


def command_plot(args: argparse.Namespace) -> None:
    """
    Ux-Data Plotter

    This script reads a Signaloid Ux-string or Ux-bytes from command-line arguments and plots
    the distributional data using the signaloid-python library.

    Ux-string format specification:
        - Particle value (double in string format)
        - "Ux"                                              (   2 chars)
        - Representation type (uint8_t)                     (   2 chars)
        - Number of samples (uint64_t)                      (  16 chars) (unused)
        - Mean value of distribution (double)               (  16 chars)
        - Number of non-zero mass Dirac deltas (uint32_t)   (   8 chars)
        - Pairs of:
            - Support position (float/double)               (8/16 chars)
            - Probability mass (uint64_t)                   (  16 chars)

    Ux-bytes specification:
        - Particle value (double)                           (  8 bytes)
        - Representation type (uint8_t)                     (  1 byte )
        - Number of samples (uint64_t)                      (  8 bytes) (unused)
        - Mean value of distribution (double)               (  8 bytes)
        - Number of non-zero mass Dirac deltas (uint32_t)   (  4 bytes)
        - Pairs of:
            - Support position (float/double)               (4/8 bytes)
            - Probability mass (uint64_t)                   (  8 bytes)

    Example usage:
        signaloid-uxdata-toolkit plot 0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
        signaloid-uxdata-toolkit plot 09168733bf9ad93f000100000000000000c7c72324c19ad93f01000000c7c72324c19ad93f0000000000000080
        signaloid-uxdata-toolkit plot -o output.png 0.40007Ux0000000000000000013FD99AC12423C7C7000000013FD99AC12423C7C78000000000000000
    """
    ux_data: str = args.ux_data.strip()

    print("Signaloid Ux-data Plotter")
    print("=" * 50)

    # Validate input
    if not ux_data:
        print("Error: Empty Ux-data provided.")
        sys.exit(1)

    try:
        # Parse the Ux string into a DistributionalValue object
        print("\nParsing Ux-data...")

        dist_value = DistributionalValue.parse(ux_data)
        if dist_value is None:
            raise ValueError("Failed to parse Ux-data into DistributionalValue.")

        # Display basic information about the distribution
        print("Successfully parsed Ux-data!")
        print(f"Particle value: {dist_value.particle_value}")
        print(f"Mean: {dist_value.mean}")
        print(f"Variance: {dist_value.variance}")
        print(f"Number of Dirac deltas: {dist_value.UR_order}")
        print(f"Double Precision: { dist_value.double_precision }")

        # Create PlotData object from the distributional value
        print("\nPreparing plot data...")
        plot_data = PlotData(dist_value)

        # Set matplotlib to non-interactive backend for saving
        matplotlib.use('Agg')

        # Plot the distribution using the plot wrapper function
        print("Generating plot...")
        plot(plot_data)

        # Save the figure to file
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {args.output}")
    except Exception as e:
        print(f"\nError parsing or plotting Ux-data: {e}")
        traceback.print_exc()
        print("\nPlease ensure you have a valid Ux-data format.")
        sys.exit(1)


def main() -> None:
    """
    Main function to parse and run the selected command.
    """
    # Parse arguments
    args: argparse.Namespace = parse_arguments()

    if args.command == "plot":
        command_plot(args)
    else:
        print("Unknown command. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
