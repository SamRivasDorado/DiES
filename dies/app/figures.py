"""
Figures of the DiES app that plotting.py does not have (Phase 4, Nov 2026), and the
rendering of any figure to image bytes. No Streamlit here.

- plot_model(scenario): sketch of the model before computing it: the grid, the layers
  with their properties, the crack and the surface.
- render(draw, fmt, dpi): draws a figure, saves it to bytes (png, pdf or svg) and closes it.
- add_overlays(figure, probes, profiles): marks the probes and the profile lines on a map,
  with their labels kept inside it.
- plot_profile(data, colour, title, **options): the line plot of a profile.

pyplot keeps a global state and is not safe when several threads draw at the same time
(Streamlit serves each session in its own thread), so every figure is drawn and saved
inside DRAW_LOCK.
"""

import io
import math
import threading

import matplotlib
matplotlib.use('Agg')       # figures are only rendered to images, never shown in a window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects

from Helper_functions import model_locators

# One figure at a time (see the module docstring)
DRAW_LOCK = threading.RLock()

# Colours of the layers in the sketch, from the top down (light, so that the crack and
# the labels can be read); repeated if there are more layers
LAYER_COLOURS = ['#E8E1D3', '#D6CBB5', '#C4B596', '#B3A07A', '#A08B62']


def render(draw, fmt='png', dpi=150):
    """
    Calls draw() (a function returning a matplotlib figure), saves the figure to bytes
    in fmt ('png', 'pdf' or 'svg') and closes it. Returns the bytes.
    """
    with DRAW_LOCK:
        figure = draw()
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format=fmt, dpi=dpi, bbox_inches='tight')
            return buffer.getvalue()
        finally:
            plt.close(figure)


def _layer_label(layer):
    # Properties of a layer in plotting units, e.g. 'host rock: E 30 GPa, ν 0.25, ...'
    return ('%s: E %g GPa, ν %g, ρ %g kg/m³, μ′ %g, T %g MPa (C %g MPa)'
            % (layer['name'], round(layer['E']/1e9, 3), layer['nu'], round(layer['density'], 1),
               layer['friction'], round(layer['T']/1e6, 3), round(2*abs(layer['T'])/1e6, 3)))


def plot_model(scenario):
    """
    Sketch of a checked scenario (dies.Scenario) before computing it: the grid limits
    and its cells, the layers with their properties, the crack (red) with its tips and
    the surface (y = 0). Returns the figure.
    """
    data = scenario.data
    (x_min, x_max), (y_base, y_top) = data['grid']['x'], data['grid']['y']
    cell_x, cell_y = data['grid']['cell']
    width, height = x_max - x_min, y_top - y_base

    # Same proportions as the model, within a readable size
    aspect = height/width if width > 0 else 1
    figure = plt.figure(figsize=(8, max(3.0, min(8.0, 8*aspect + 1.2))), tight_layout=True)
    ax = figure.add_subplot(111)

    # Layers: each one from its top to the top of the next one (the last one to the base)
    layers = data['layers']
    tops = [layer['top'] for layer in layers]
    bottoms = tops[1:] + [y_base]
    for i, (layer, top, bottom) in enumerate(zip(layers, tops, bottoms)):
        top_in, bottom_in = min(top, y_top), max(bottom, y_base)
        if top_in <= bottom_in:
            continue
        ax.add_patch(Rectangle((x_min, bottom_in), width, top_in - bottom_in,
                               fc=LAYER_COLOURS[i % len(LAYER_COLOURS)], ec='none', zorder=0))
        ax.text(x_min + 0.01*width, top_in - 0.015*height, _layer_label(layer), ha='left', va='top',
                fontsize=8, zorder=3, bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
        if i > 0:
            ax.axhline(top, c='0.35', lw=1, ls='--', zorder=1)

    # Model limits and the size of one cell, drawn at the lower right corner
    ax.add_patch(Rectangle((x_min, y_base), width, height, fc='none', ec='0.2', lw=1.2, zorder=2))
    ax.add_patch(Rectangle((x_max - cell_x, y_base), cell_x, cell_y, fc='0.2', ec='none', zorder=2))
    # Number of nodes, from the grid the core builds (cheap: no stresses are computed)
    ny, nx = scenario.build_model().grid()[0].shape
    ax.text(x_max - 0.01*width, y_base + 0.02*height + cell_y,
            'cell %g × %g m · %d × %d nodes' % (cell_x, cell_y, nx, ny),
            ha='right', va='bottom', fontsize=8, zorder=3, bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))

    # The crack and its tips
    c = data['cracks'][0]
    (ux, uy), (lx, ly) = c['upper_tip'], c['lower_tip']
    ax.plot((ux, lx), (uy, ly), c='red', lw=3, zorder=4, solid_capstyle='round')
    ax.scatter((ux, lx), (uy, ly), c='red', s=18, zorder=5)
    length = math.hypot(lx - ux, ly - uy)
    # Its name and length, next to the upper tip
    ax.annotate('%s (%g km)' % (c['name'], round(length/1e3, 3)), (ux, uy), xytext=(8, 4),
                textcoords='offset points', fontsize=9, color='darkred', va='bottom', zorder=5,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))

    # The surface, above the top of the grid
    ax.axhline(0, c='black', lw=1.5, zorder=2)
    margin = 0.04*height
    ax.set_xlim(x_min - 0.02*width, x_max + 0.02*width)
    ax.set_ylim(min(y_base, ly, uy) - margin, max(0.0, uy) + margin)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('%s · g = %g m/s² · lithostatic %s' % (data['name'], data['gravity'],
                                                         'on' if data['lithostatic'] else 'off'), fontsize=10)
    model_locators(ax, (x_min, x_max), (y_base, y_top))
    return figure


def plot_crack(scenario):
    """
    Profile of the crack walls (plotting.plot_crack_displacements) of a checked scenario,
    with its final driving stresses and E and v of the layer at its centre, as
    ScenarioResults.figure('crack_displacements'), but without computing the model.
    """
    from Modeling_functions import crack_layer_properties
    import plotting
    current_model = scenario.build_model()
    disc = scenario.build_crack(current_model)
    E, v = crack_layer_properties(current_model, disc.up_tip(), disc.low_tip())
    return plotting.plot_crack_displacements(disc, v, E)


# Colours of the probes and profiles, in the order they are added: on the maps (bright,
# with a black halo, readable on any colour map) and in the profile plots (darker, readable
# on white). The same index gives the same probe or profile in both.
MARK_COLOURS = ['white', '#00e5ff', '#ff4fd8', '#ffd400', '#7cfc00', '#ff9f1c']
LINE_COLOURS = ['#1f5fad', '#00838f', '#c2185b', '#b8860b', '#2e7d32', '#e65100']


def mark_colour(index):
    """Colour of probe or profile number index (from 0) on the maps."""
    return MARK_COLOURS[index % len(MARK_COLOURS)]


def line_colour(index):
    """Colour of profile number index (from 0) in its plot."""
    return LINE_COLOURS[index % len(LINE_COLOURS)]


def _inward_label(ax, point, text, colour, halo, beside=True):
    """
    Writes text next to point, on the side towards the centre of the axes, so that it
    always stays inside the map (Sam, 27 Nov 2026: a label at the right edge ran out of it).
    beside: diagonally beside the point (the ends of a profile); otherwise just above or
    below it (the probes, so that the labels of two probes side by side do not run together).
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    upper = point[1] > (y0 + y1)/2
    if beside:
        right = point[0] > (x0 + x1)/2
        dx, ha = (-5, 'right') if right else (5, 'left')
    else:
        # Centred on the point, unless it is near a side of the map
        third = (point[0] - x0)/(x1 - x0) if x1 != x0 else 0.5
        dx, ha = 0, ('left' if third < 0.1 else 'right' if third > 0.9 else 'center')
    dy = -7 if upper else 7
    ax.annotate(text, point, xytext=(dx, dy), textcoords='offset points', ha=ha, va='top' if upper else 'bottom',
                color=colour, fontsize=10, fontweight='bold', zorder=21, annotation_clip=True, **halo)


def add_overlays(figure, probes=(), profiles=()):
    """
    Marks the probes and the profiles on a map figure (its first axes), on top of everything:
    probes: (x, y, label) of each probe (the node used), a circle with its label;
    profiles: ((xa, ya), (xb, yb), label) of each profile, a dashed line with its ends
    labelled A<label> and B<label>. The n-th probe or profile gets the n-th colour of
    MARK_COLOURS. The labels are kept inside the axes. Returns the figure.
    """
    ax = figure.axes[0]
    limits = ax.get_xlim(), ax.get_ylim()
    halo = dict(path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
    for i, ((xa, ya), (xb, yb), label) in enumerate(profiles):
        colour = mark_colour(i)
        ax.plot((xa, xb), (ya, yb), c=colour, lw=1.8, ls='--', zorder=20, **halo)
        _inward_label(ax, (xa, ya), 'A' + label, colour, halo)
        _inward_label(ax, (xb, yb), 'B' + label, colour, halo)
    for i, (x, y, label) in enumerate(probes):
        colour = mark_colour(i)
        ax.scatter([x], [y], s=70, facecolors=colour, edgecolors='black', linewidths=1.5, zorder=22)
        _inward_label(ax, (x, y), label, colour, halo, beside=False)
    # The marks must not change the limits of the map
    ax.set_xlim(limits[0]) ; ax.set_ylim(limits[1])
    return figure


# Options of the profile plots (Sam, 27 Nov 2026), and their defaults
PROFILE_OPTIONS = {'major_grid': True, 'minor_grid': False, 'axis_labels': True, 'line_width': 1.8, 'markers': False}


def plot_profile(data, colour=LINE_COLOURS[0], title=None, distance_label='distance (m)', **options):
    """
    Line plot of a profile (dies.app.views.profile): one panel per field, against the
    distance from A, in colour. options (PROFILE_OPTIONS): major_grid, minor_grid,
    axis_labels (the axis titles; without them, the field is written inside its panel),
    line_width, markers (one per sampled point). Returns the figure.
    """
    style = dict(PROFILE_OPTIONS)
    style.update(options)
    fields = [label for label in data if label not in (distance_label, 'x (m)', 'y (m)')]
    figure, axes = plt.subplots(len(fields), 1, figsize=(5, 1.9*len(fields) + 0.9), sharex=True, squeeze=False,
                                tight_layout=True)
    for ax, label in zip(axes[:, 0], fields):
        ax.plot(data[distance_label], data[label], c=colour, lw=style['line_width'],
                marker='o' if style['markers'] else None, markersize=2.5)
        # Grids: major and minor lines, each one optional
        if style['minor_grid']:
            ax.minorticks_on()
            ax.grid(which='minor', alpha=0.15, lw=0.5)
        if style['major_grid']:
            ax.grid(which='major', alpha=0.35)
        ax.tick_params(labelsize=8)
        if style['axis_labels']:
            ax.set_ylabel(label, fontsize=9)
        else:
            # The name of the field inside the panel, at its upper left corner
            ax.text(0.01, 0.97, label, transform=ax.transAxes, ha='left', va='top', fontsize=8,
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    if style['axis_labels']:
        axes[-1, 0].set_xlabel('Distance from A (m)', fontsize=9)
    if title:
        axes[0, 0].set_title(title, fontsize=10, color=colour, fontweight='bold')
    return figure
