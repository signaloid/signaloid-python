#   Copyright (c) 2025, Signaloid.
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


# type: ignore


import math

import board
import displayio
import terminalio

from adafruit_display_text import label
from circuitpython_uplot.cartesian import Cartesian
from circuitpython_uplot.plot import Plot
from circuitpython_uplot.scatter import Pointer, Scatter

from signaloid.circuitpython.extended_ulab_numpy import np
from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import \
    PlotData


# The background color of the plot
BG_COLOR = 0xFFFFFF
BG_COLOR_STR = f"{BG_COLOR:06x}"

# The text color of the plot
TEXT_COLOR = 0x000000

EDGE_COLOR = 0x33A333

FACE_COLOR = 0x134289

# The number of decimal places to show in the ticks
TICK_DECIMAL_POINTS = 2

# The size of the ticks (font characters) in pixels
TICK_SIZE = 8

# The size of the Signaloid logo in pixels
SIGNALOID_LOGO_SIZE = 24
SIGNALOID_LOGO_MARGIN = 2


# The Dirac Delta arrow characteristics
DELTA_WIDTH = 0.012
DELTA_COLOR = 0xFF00FF
DELTA_ARROW_HEAD_SHIFT = -0.025
DELTA_X_RANGE = 0.5
DELTA_Y_RANGE = 1.5


X_SHIFT = 4 * TICK_SIZE
Y_SHIFT = 0


def apply_array_multiplier(arr, multiplier):
    arr = np.divide(arr, 10 ** multiplier)

    return arr


def transform_array_with_multiplier(arr):
    max_value = max(np.abs(arr))
    multiplier = int(math.log(max_value, 10)) + (-1 if max_value < 1 else 0)
    arr = np.divide(arr, 10 ** multiplier)

    return arr, multiplier


def gen_xy_origin(position, width, height, align="center"):
    """
    Generates the x and y coordinates to plot a rectangle at the given
    position.

    :param position: The position of the rectangle.
    :param width: The width of the rectangle.
    :param height: The height of the rectangle.
    :param align: The alignment of the rectangle. Can be "center" or "edge".
                    If "center", the rectangle will be centered at the
                    given position.
                    If "edge", the rectangle will be aligned to the left
                    edge of the given position.

    :return: A tuple containing the x and y coordinates.
    """

    if align == "center":
        start = position - width / 2
        end = position + width / 2
    elif align == "edge":
        start = position
        end = position + width

    x = [start, start, end, end]
    y = [0, height, height, 0]

    return x, y


def render_bg_plot(g):
    # Create the background and append it to the group.
    # We have to create a background plot separately because the main plot is
    # smaller than the screen, and leaves empty/black space, so the background
    # one acts as a filler.
    bg_plot = Plot(
        x=0,
        y=0,
        width=board.DISPLAY.width,
        height=board.DISPLAY.height,
        show_box=False,
        background_color=BG_COLOR,
        box_color=TEXT_COLOR,
        scale=1,
    )
    g.append(bg_plot)


def create_main_plot(g):
    # Create the main plot and append it to the group
    plot = Plot(
        x=X_SHIFT,
        y=0,
        width=board.DISPLAY.width - X_SHIFT + 2 * TICK_SIZE,
        height=board.DISPLAY.height - 2 * TICK_SIZE,
        show_box=True,
        background_color=BG_COLOR,
        box_color=TEXT_COLOR,
        scale=1,
    )
    g.append(plot)

    # Set the axes type to 'line' and set the tick parameters
    plot.axs_params(axstype="line")
    plot.tick_params(
        show_ticks=True,
        tickx_height=TICK_SIZE,
        ticky_height=TICK_SIZE,
        tickcolor=TEXT_COLOR,
        tickgrid=True,
        showtext=True,
        decimal_points=TICK_DECIMAL_POINTS,
    )
    return plot


def render_histogram(plot, boundary_positions, bin_heights, width, align, edgecolor):
    # Generate the x and y coordinates for the rectangles
    x = []
    y = []
    for p, w, h in zip(boundary_positions, width, bin_heights):
        x_tmp, y_tmp = gen_xy_origin(p, w, h, align=align)
        x += x_tmp
        y += y_tmp

    # Set the plot range
    rangex = [min(x), max(x)]
    rangey = [0, max(y)]

    # Draw the histogram bars using a Cartesian plot and the generated x and y
    # coordinates
    c = Cartesian(
        plot=plot,
        x=x,
        y=y,
        rangex=rangex,
        rangey=rangey,
        line_color=edgecolor,
        fill=True,
        nudge=False,
        logging=False,
    )

    return c, rangex, rangey


def render_x_multiplier(g, c, multiplier):
    anchored_x = c.points[1][0] + X_SHIFT - .5 * TICK_SIZE
    anchored_y = c.points[1][1] - Y_SHIFT + .5 * TICK_SIZE
    text_area = label.Label(
        font=terminalio.FONT,
        text=f"e{multiplier:+d}",
        color=TEXT_COLOR,
        line_spacing=0.8,
        anchor_point=(1.0, 0.0),
        anchored_position=(anchored_x, anchored_y),
        label_direction="LTR",
    )
    g.append(text_area)


def render_y_multiplier(g, c, multiplier):
    minh = min(c.points[:][1])
    anchored_x = c.points[1][0] + X_SHIFT - .5 * TICK_SIZE
    anchored_y = minh - 2 * TICK_SIZE
    text_area = label.Label(
        font=terminalio.FONT,
        text=f"e{multiplier:+d}",
        color=TEXT_COLOR,
        line_spacing=0.8,
        anchor_point=(1.0, 0.0),
        anchored_position=(anchored_x, anchored_y),
        label_direction="LTR",
    )
    g.append(text_area)


def render_non_finite_values(g, nan, neg_inf, pos_inf):
    text_area = label.Label(
        font=terminalio.FONT,
        text=f"NaN: {nan:0.2f} | -Inf: {neg_inf:0.2f} | +Inf: {pos_inf:0.2f}",
        color=TEXT_COLOR,
        line_spacing=0.8,
        anchor_point=(0.5, 1.0),
        anchored_position=(board.DISPLAY.width // 2, board.DISPLAY.height - 1.5 * TICK_SIZE),
        label_direction="LTR",
    )
    g.append(text_area)


def render_dirac_delta(plot, position, height):
    # Generate the x and y coordinates for the rectangles
    x, y = gen_xy_origin(position, DELTA_WIDTH, height, align="center")

    # Add dummy points to the x and y coordinates to make the plot scaling
    # work correctly
    x = (
        [position - DELTA_X_RANGE]
        + x
        + [position + DELTA_X_RANGE, position + DELTA_X_RANGE]
    )
    y = [0] + y + [0, height * DELTA_Y_RANGE]

    # Set the plot range
    rangex = [min(x), max(x)]
    rangey = [0, max(y)]

    # Draw the Dirac Delta using a Cartesian plot and the generated x and y
    # coordinates
    c = Cartesian(
        plot=plot,
        x=x,
        y=y,
        rangex=rangex,
        rangey=rangey,
        line_color=DELTA_COLOR,
        fill=True,
        nudge=False,
        logging=False,
    )

    # Draw the Dirac Delta arrow head with a triangle
    Scatter(
        plot,
        [position + DELTA_ARROW_HEAD_SHIFT],
        [height],
        rangex=rangex,
        rangey=rangey,
        pointer=Pointer.TRIANGLE,
        pointer_color=DELTA_COLOR,
        nudge=False,
    )

    return c


def render_title(g, title):
    # Show the title text
    text_area = label.Label(
        font=terminalio.FONT,
        text=title,
        color=TEXT_COLOR,
        background_color=BG_COLOR,
        line_spacing=0.8,
        anchor_point=(0.5, 0.0),
        anchored_position=(board.DISPLAY.width // 2, 0),
        label_direction="LTR",
    )
    g.append(text_area)


def render_x_axis_label(g):
    # Show the x-axis label
    text_area = label.Label(
        font=terminalio.FONT,
        text="Distribution Support",
        color=TEXT_COLOR,
        background_color=BG_COLOR,
        line_spacing=0.8,
        anchor_point=(0.5, 1.0),
        anchored_position=(board.DISPLAY.width // 2, board.DISPLAY.height),
        label_direction="LTR",
    )
    g.append(text_area)


def render_y_axis_label(g):
    # Show the y-axis label
    text_area = label.Label(
        font=terminalio.FONT,
        text="Probability Density",
        color=TEXT_COLOR,
        background_color=BG_COLOR,
        line_spacing=0.8,
        anchor_point=(0.0, 0.5),
        anchored_position=(0, board.DISPLAY.height // 2),
        label_direction="UPR",
    )
    g.append(text_area)


def render_particle_value_value_label(g, particle_value):
    # Show the title text
    text_area = label.Label(
        font=terminalio.FONT,
        text=f"E(x) = {particle_value}",
        color=TEXT_COLOR,
        background_color=BG_COLOR,
        line_spacing=0.8,
        anchor_point=(0.5, 0.0),
        anchored_position=(board.DISPLAY.width // 2, 1.5 * TICK_SIZE),
        label_direction="LTR",
    )
    g.append(text_area)


def render_particle_value_line(plot, particle_value, bin_heights, width, rangex, rangey, facecolor):
    # To draw the particle value, we need to create a rectangle, the same
    # way we create the histogram bars

    # Calculate the width of the particle value rectangle so that it is
    # easily visible compared to the other histogram bars
    particle_width = np.average(width) / 4

    # Use the maximum height of the histogram bars to span over the entire
    # height of the plot
    height = max(bin_heights)

    # Generate the x and y coordinates for the particle value rectangle
    x, y = gen_xy_origin(particle_value, particle_width, height, align="center")

    # Draw the particle value rectangle using a Cartesian plot
    c = Cartesian(
        plot=plot,
        x=x,
        y=y,
        rangex=rangex,
        rangey=rangey,
        line_color=facecolor,
        fill=True,
        nudge=False,
        logging=False,
    )

    return c


def render_particle_value_line_label(g, c):
    # Add the "E(X)" label to the plot to indicate the particle value
    text_area = label.Label(
        font=terminalio.FONT,
        text="E(X)",
        color=TEXT_COLOR,
        line_spacing=0.8,
        anchor_point=(0.0, 0.0),
        anchored_position=(c.points[3][0] + X_SHIFT, c.points[3][1]),
        label_direction="UPR",
    )
    g.append(text_area)


def render_logo(g):
    # Add the Signaloid logo to the top right corner of the screen
    bitmap = displayio.OnDiskBitmap(
        open("assets/signaloid_logo_24x24.bmp", "rb")
    )
    sprite = displayio.TileGrid(
        bitmap,
        pixel_shader=bitmap.pixel_shader,
        width=1,
        height=1,
    )
    sprite.x = (
        board.DISPLAY.width - SIGNALOID_LOGO_SIZE - SIGNALOID_LOGO_MARGIN
    )
    sprite.y = SIGNALOID_LOGO_MARGIN
    g.append(sprite)


def show_plot(g):
    # Show the whole group to the screen
    board.DISPLAY.root_group = g


def bar(
    plot_data: PlotData,
    title="",
    align="edge",
    edgecolor=EDGE_COLOR,
    facecolor=FACE_COLOR,
):
    """
    This function plots a Signaloid histogram style bar chart.

    :param boundary_positions: The positions of the bin boundaries.
    :param bin_heights: The heights of the bins.
    :param width: The width of each bin.
    :param align: The alignment of the bars. Possible values are 'edge' and
                  'center'.
    :param edgecolor: The color of the bars.
    :param facecolor: The color of particle value.
    :param hatch: The hatch pattern of the bars. This is not used, but it is
                  included for compatibility with the existing code.
    :param title: The title of the chart.
    :param particle_value: The value of the particle.
    """
    boundary_positions = plot_data.positions
    bin_heights = plot_data.masses
    width = plot_data.widths
    particle_value = plot_data.dist.particle_value

    # Create a displayio Group to hold the plot
    g = displayio.Group()
    render_bg_plot(g)
    plot = create_main_plot(g)

    boundary_positions, boundary_multiplier = transform_array_with_multiplier(boundary_positions)
    bin_heights, bin_heights_multiplier = transform_array_with_multiplier(bin_heights)
    width = apply_array_multiplier(width, boundary_multiplier)

    c, rangex, rangey = render_histogram(
        plot,
        boundary_positions,
        bin_heights,
        width,
        align,
        edgecolor
    )

    render_x_multiplier(g, c, boundary_multiplier)
    render_y_multiplier(g, c, bin_heights_multiplier)

    if plot_data.dist.has_special_values:
        render_non_finite_values(
            g,
            plot_data.dist.nan_dirac_delta.mass,
            plot_data.dist.neg_inf_dirac_delta.mass,
            plot_data.dist.pos_inf_dirac_delta.mass
        )

    render_title(g, title)
    render_particle_value_value_label(g, particle_value)
    render_x_axis_label(g)
    render_y_axis_label(g)

    # Show the particle value
    if (
        particle_value is not None
        and np.isfinite(particle_value)
    ):
        particle_value_normalized = particle_value / (10 ** boundary_multiplier)
        if (
            particle_value_normalized >= rangex[0]
            and particle_value_normalized <= rangex[1]
        ):
            c = render_particle_value_line(
                plot,
                particle_value_normalized,
                bin_heights,
                width,
                rangex,
                rangey,
                facecolor
            )
            render_particle_value_line_label(g, c)

    render_logo(g)
    show_plot(g)


def annotate(
    title,
    xy=(0, 0),
):
    """
    This function plots an annotation with a Dirac Delta arrow head.

    :param title: The title of the annotation.
    :param xy: The position of the Dirac Delta arrow head.
    """
    # Create a displayio Group to hold the plot
    g = displayio.Group()
    render_bg_plot(g)
    plot = create_main_plot(g)

    boundary_positions, boundary_multiplier = transform_array_with_multiplier([xy[0]])
    bin_heights, bin_heights_multiplier = transform_array_with_multiplier([xy[1]])

    xy = (boundary_positions[0], bin_heights[0])

    c = render_dirac_delta(plot, xy[0], xy[1])

    render_x_multiplier(g, c, boundary_multiplier)
    render_y_multiplier(g, c, bin_heights_multiplier)

    render_title(g, title)
    render_x_axis_label(g)
    render_y_axis_label(g)

    render_particle_value_line_label(g, c)

    render_logo(g)
    show_plot(g)


def plot(
    plot_data: PlotData,
    plot_name: str = "",
) -> bool:
    """
    Args:
        plot_data: `PlotData` to plot.
        plot_name: The title of the chart.
    Returns:
        `True` if successful, `False` else.
    """
    # If there is only one finite Dirac delta, then plot just an
    # arrow representing a Dirac delta.
    if len(plot_data.positions) == 1:
        annotate(
            title=plot_name,
            xy=(plot_data.positions[0], plot_data.masses[0]),
        )
        return True

    if len(plot_data.positions) > 1:
        bar(
            plot_data=plot_data,
            title=plot_name,
        )
        return True

    if plot_data.dist.has_special_values:
        print(f" NaN : {plot_data.dist.nan_dirac_delta.mass}")
        print(f"-Inf : {plot_data.dist.neg_inf_dirac_delta.mass}")
        print(f" Inf : {plot_data.dist.pos_inf_dirac_delta.mass}")
        return True

    return False
