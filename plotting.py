"""
DiES figures, drawn from already computed results.

Created on 2026-11-24 (Phase 1 of the roadmap). The drawing code of the
model.plot_* methods was moved here unchanged, so the figures are the same;
the difference is that these functions receive a DiESResults object
(model.compute) and never recompute anything. The model.plot_* methods are
kept as wrappers: they compute once and call these functions.

The aesthetic improvements listed in the roadmap (Phase 2) are not applied
here yet.

Example
-------
    res = current_model.compute(disc, rxx, ryy, litho=True)
    fig = plotting.plot_stresses(res, 'sxy', 1)
    fig = plotting.plot_principal_stresses(res, 'principal', 5)
    fig = plotting.plot_stress_trajectories(res, planes=True)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from Helper_functions import G
from Helper_functions import boundaries_contours
from Helper_functions import auto_contour_step
from Helper_functions import model_locators
from Helper_functions import create_figure
from Helper_functions import true_min_max
from Helper_functions import downsample_2darray


class PlotError(Exception):
    """Invalid plotting request (e.g. an unknown option or a missing field)."""


# Former keys of model.plot_stresses -> field name and colour-bar label
STRESS_KEYS = {
    'sxa': ('sxx', r'$\sigma_{xx}$ (MPa)'),
    'sya': ('syy', r'$\sigma_{yy}$ (MPa)'),
    'sav': ('sav', r'$\sigma_{av}$ (MPa)'),
    'sxya': ('sxy', r'$\sigma_{xy}$ (MPa)'),
    'sxza': ('sxz', r'$\sigma_{xz}$ (MPa)'),
    'sza': ('szz', r'$\sigma_{zz}$ (MPa)'),
    'ta': ('tmax', r'$\tau$ (MPa)'),
}
# The field names are accepted too
STRESS_KEYS.update({field: (field, label) for field, label in list(STRESS_KEYS.values())})


def _require(res, *names):
    # Fails early, with a clear message, if the results lack a field a figure needs
    missing = [name for name in names if name not in res]
    if missing:
        raise PlotError('The results do not contain ' + ', '.join(missing)
                        + '; compute them with model.compute(..., displacements=True, stresses=True)')


def _crack_tips(res):
    # The crack tips, stored in the metadata by model.metadata(), to draw the crack
    crack = res.metadata.get('crack')
    if crack is None:
        raise PlotError('The results have no crack in their metadata')
    return crack['upper_tip'], crack['lower_tip']


def _axes(res, figsize, locators):
    """Figure and axes with the model limits, as in the original plot_* methods."""
    figure = plt.figure(figsize=figsize, tight_layout=True)
    ax = figure.add_subplot(111, xlim=res.metadata['x_limits'], ylim=res.metadata['y_limits'])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.tick_params(labelsize=12, axis='both')
    return figure, ax


def _set_locators(ax, res):
    # Major ticks at round intervals that do not overlap (at most 6 labels in x,
    # 13 in y) and automatic minor ticks. Until Nov 2026 the major ticks were every
    # 1000 m (x) and 500 m (y), and the x labels of a 10 km model overlapped.
    model_locators(ax, res.metadata['x_limits'], res.metadata['y_limits'])


def _cell_size(res):
    """Smallest spacing between nodes of the grid, in m (to scale segments and arrows)."""
    spacing = [np.min(np.abs(np.diff(v))) for v in (res.x_nodes(), res.y_nodes()) if len(v) > 1]
    return float(min(spacing)) if spacing else 100.0


def _label_contours(ax, contours):
    """
    Labels of the contours, with the digits the levels need (e.g. 20, 2.5, 0.05).
    Until Nov 2026 the number of decimals depended on the step given.
    """
    if len(contours.levels):
        ax.clabel(contours, fmt='%g')


def _scalar_field(res, soi, label, step=None, figsize=(6, 6), locators=False, grid_lines=False,
                  cmap='viridis', limits=None, contours=True):
    """
    pcolormesh + contours of a field already in plotting units.

    step: contour interval; None chooses a round one automatically
    (Helper_functions.auto_contour_step). limits: (vmin, vmax) of the colours,
    instead of the 1st and 99th percentiles. contours: False for no contours.
    """
    upper_tip, lower_tip = _crack_tips(res)
    x, y = res.x, res.y
    figure, ax = _axes(res, figsize, locators)
    # The crack, in red
    ax.plot((upper_tip[0], lower_tip[0]), (upper_tip[1], lower_tip[1]), c='red', lw=3)
    if locators:
        _set_locators(ax, res)
    # A field that is the same everywhere (e.g. sigma_xz without mode III loading)
    # has nothing to map: a note says so, instead of an empty colour scale
    finite = np.asarray(soi, dtype=float)[np.isfinite(soi)]
    if finite.size == 0 or np.ptp(finite) == 0:
        value = finite[0] if finite.size else float('nan')
        ax.text(0.5, 0.5, label.split(' (')[0] + ' = ' + ('%g' % value) + ' everywhere', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, bbox=dict(fc='white', ec='0.5'))
        return figure, ax
    # Colour limits (1st and 99th percentiles) and contour levels, at multiples of step
    plot_params = boundaries_contours(soi, step)
    vmin, vmax = limits if limits is not None else plot_params[:2]
    # Colour map of the field, with a colour bar
    plot = ax.pcolormesh(x, y, soi, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    if grid_lines:
        ax.grid(which='both', alpha=0.5)
    cbar = figure.colorbar(plot, extend='neither' if limits is not None else 'both')
    cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    # Dashed, labelled contours at the levels
    if contours:
        lines = ax.contour(x, y, soi, levels=plot_params[-1], colors='black', linewidths=0.75, linestyles='--')
        _label_contours(ax, lines)
    return figure, ax


# --------------------------------------------------------------------- figures

def plot_displacement(res, displacement, option, step=None):
    """
    Displacement figure (formerly model.plot_displacement).

    res: DiESResults with ux and uy,
    displacement: 'ux', 'uy' or 'uxy' (in-plane vector; colour = its magnitude),
    option: 'vectors' or 'contours',
    step: contour step, in m (also used when option is 'vectors'); None (the
    default since Nov 2026) chooses a round one from the values.
    """
    _require(res, 'ux', 'uy')
    upper_tip, lower_tip = _crack_tips(res)
    crack = res.metadata['crack']
    dx, dy = res.ux, res.uy
    x, y = res.x, res.y

    figure, ax = _axes(res, (6, 6), False)
    ax.plot((upper_tip[0], lower_tip[0]), (upper_tip[1], lower_tip[1]), c='red', lw=3, label='Crack')

    # dp, ds: x and y components of the arrows; dt: the scalar used for the colours
    # and contours; plot_params: colour limits and contour levels
    if displacement == 'ux':
        dp = dx ; ds = 0 ; dt = dx ; plot_params = boundaries_contours(dp, step) ; label = r'u$_x$ (m)'
    elif displacement == 'uy':
        dp = 0 ; ds = dy ; dt = dy ; plot_params = boundaries_contours(ds, step) ; label = r'u$_y$ (m)'
    elif displacement == 'uxy':
        # Magnitude of the in-plane displacement vector
        dp = dx ; ds = dy ; dt = np.hypot(dx, dy) ; plot_params = boundaries_contours(dt, step) ; label = r'|u$_{xy}$| (m)'
    else:
        raise PlotError('Invalid displacement. Input "ux", "uy", or "uxy" instead.')

    if option == 'vectors':
        # The arrows are scaled with the largest magnitude of the plotted vector
        # (|ux|, |uy| or |u|, depending on the selected displacement)
        plot = ax.quiver(x, y, dp, ds, dt, width=0.0050, scale=np.nanmax(np.hypot(dp, ds))/0.1, cmap='viridis')
        cbar = figure.colorbar(plot, extend='both')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
    elif option == 'contours':
        # Colour map and labelled contours of dt
        plot = ax.pcolormesh(x, y, dt, cmap='viridis', vmin=plot_params[0], vmax=plot_params[1], shading='nearest')
        cbar = figure.colorbar(plot, extend='both')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        contours = ax.contour(x, y, dt, levels=plot_params[-1], colors='black', linewidths=0.75, linestyles='--')
        _label_contours(ax, contours)
    else:
        raise PlotError('Invalid option. Input "vectors" or "contours" instead.')

    # Box with the driving stresses of the crack
    ax.text(0.97, 0.10, r'$\sigma_I$ = ' + str(round(crack['dpx']/1E6, 2))
            + ' MPa\n' + r'$\sigma_{II}$ = ' + str(round(crack['dpy']/1E6, 2)) + ' MPa',
            bbox=dict(fc='white', ec='black'),
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes, fontsize=11, zorder=5)
    return figure


def plot_stresses(res, stress, step=None):
    """
    Stress component figure, in MPa (formerly model.plot_stresses).

    stress: 'sxa', 'sya', 'sav', 'sxya', 'sxza', 'sza', 'ta' (the former keys)
    or the field names 'sxx', 'syy', 'sav', 'sxy', 'sxz', 'szz', 'tmax',
    step: contour step, in MPa; None (the default since Nov 2026) chooses a
    round one from the values.
    """
    if stress not in STRESS_KEYS:
        raise PlotError('Invalid stress "' + str(stress) + '". Input one of ' + ', '.join(sorted(STRESS_KEYS)))
    field, label = STRESS_KEYS[stress]
    _require(res, field)
    # Drawn in MPa
    figure, ax = _scalar_field(res, res[field]/1e6, label, step, grid_lines=True)
    return figure


def plot_principal_stresses(res, stress, step=None):
    """
    Principal stress figure (formerly model.plot_principal_stresses).

    stress: 'principal' (sigma1, MPa), 'secondary' (sigma3, MPa) or 'angle'
    (orientation of sigma1, degrees counterclockwise from +x),
    step: contour step, in MPa; None (the default since Nov 2026) chooses a
    round one from the values. Not used for 'angle'.

    The angle is drawn with a cyclic colour scale ('twilight', -90 to 90
    degrees) and without contours, since -90 and +90 degrees are the same
    orientation (Nov 2026; until then a linear scale made vertical sigma1
    look both dark and bright, with a false jump between them).
    """
    _require(res, 's1', 's3', 's1a')
    # The field to draw (soi, 'stress of interest') and its colour-bar label
    if stress == 'principal':
        soi = res.s1/1e6 ; label = r'$\sigma_1$ (MPa)'
    elif stress == 'secondary':
        soi = res.s3/1e6 ; label = r'$\sigma_3$ (MPa)'
    elif stress == 'angle':
        soi = res.s1a ; label = r'$\sigma_1$ angle from +x (°)'
        figure, ax = _scalar_field(res, soi, label, locators=True, cmap='twilight', limits=(-90, 90),
                                   contours=False)
        return figure
    else:
        raise PlotError('Invalid stress. Input "principal", "secondary" or "angle" instead.')
    figure, ax = _scalar_field(res, soi, label, step, locators=True)
    return figure


def plot_stress_trajectories(res, planes):
    """
    Orientations of sigma1 and sigma3 and, if planes is True, of the two
    conjugate optimal shear planes (formerly model.plot_stress_trajectories).
    Orientations are drawn as (cos a, sin a), counterclockwise from +x.
    """
    if not isinstance(planes, bool):
        raise PlotError('planes must be a bool (True or False), not ' + repr(planes))
    _require(res, 's1', 's3', 's1a', 's3a', 'coula')
    upper_tip, lower_tip = _crack_tips(res)
    x, y = res.x, res.y
    # Magnitudes (MPa, for the colours) and orientations (radians) of sigma1 and sigma3
    p = res.s1/1e6 ; s = res.s3/1e6
    p_angles = np.deg2rad(res.s1a) ; s_angles = np.deg2rad(res.s3a)
    coula = res.coula

    figure, ax = _axes(res, (6, 6), True)
    ax.plot((upper_tip[0], lower_tip[0]), (upper_tip[1], lower_tip[1]), c='red', lw=3)
    _set_locators(ax, res)

    # Headless segments centred on each node (trajectories), coloured by the magnitude:
    # sigma1 in copper, sigma3 in cool colours. Their length is 0.9 times the spacing
    # of the grid and their width 0.08 times (Nov 2026; until then a fixed 111 m and
    # 10 m, too short on coarse grids and overlapping on fine ones)
    cell = _cell_size(res)
    seg = dict(headlength=0, headaxislength=0, pivot='middle', scale=1/(0.9*cell), units='xy',
               width=0.08*cell, headwidth=1)
    plot1 = ax.quiver(x, y, np.cos(p_angles), np.sin(p_angles), p, cmap='copper_r', **seg)
    plot2 = ax.quiver(x, y, np.cos(s_angles), np.sin(s_angles), s, cmap='cool', **seg)

    # Two horizontal colour bars under the map, at fixed places below the x label
    # (Nov 2026; side by side, their labels overlapped)
    for plot, label, bottom in ((plot1, r'$\sigma_1$ (MPa)', -0.23), (plot2, r'$\sigma_3$ (MPa)', -0.43)):
        cbar = figure.colorbar(plot, cax=ax.inset_axes([0.05, bottom, 0.9, 0.025]), extend='both',
                               orientation='horizontal')
        cbar.set_label(label, fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    if planes:
        # The two conjugate optimal shear planes lie at +/- coula from sigma1;
        # beta and beta_p are their orientations, counterclockwise from +x
        beta = p_angles - coula
        beta_p = 2*p_angles - beta
        ax.quiver(x, y, np.cos(beta), np.sin(beta), color='#4DAD4B', alpha=0.8, **seg)
        ax.quiver(x, y, np.cos(beta_p), np.sin(beta_p), color='#FA422B', alpha=0.8, **seg)
    return figure


def plot_plane_stresses(res, stress, angle, step=None):
    """
    Normal ('normal') or shear ('shear') stress on the optimal shear planes,
    in MPa (formerly model.plot_plane_stresses). angle is only written in the
    label box; step is the contour step in MPa, or None (the default since
    Nov 2026) for a round one chosen from the values.
    """
    _require(res, 'sn', 'ss')
    meta = res.metadata ; crack = meta['crack']
    # The field to draw, in MPa, and its label
    if stress == 'normal':
        soi = res.sn/1e6 ; label = r'$\sigma_n$ (MPa)'
    elif stress == 'shear':
        soi = res.ss/1e6 ; label = r'$\sigma_s$ (MPa)'
    else:
        raise PlotError('Invalid stress. Input "normal" or "shear" instead.')
    figure, ax = _scalar_field(res, soi, label, step, locators=True)

    # Box with the driving stresses and the material: single values for one layer,
    # the list of values of every layer otherwise
    dpx, dpy = crack['dpx'], crack['dpy']
    if meta['n_layers'] == 1:
        ax.text(0.97, 0.14, r'$\sigma_I$ = ' + str(round(dpx/1E6, 2)) + ' MPa ; ' + r'$\sigma_{II}$ = '
                + str(round(dpy/1E6, 2)) + ' MPa\n'
                + r'$\alpha$ = ' + str(angle) + '° ; ' + r'$\rho$ = ' + str(np.float64(meta['layers_d'][0])) + ' kg m$^3$'
                + '\nC = ' + str(np.float64(meta['layers_c'][0])/1e6) + ' MPa ; ' + r"$\mu'$ = "
                + str(np.float64(meta['layers_cfi'][0])) + '; '
                + r'$\theta_{opt}$ = ' + str(round(90 - np.degrees(np.float64(meta['layers_coula'][0])), 2)) + '°',
                bbox=dict(fc='white', ec='black'),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontsize=11, zorder=5)
    else:
        ax.text(0.98, 0.14, r'$\sigma_I$ = ' + str(round(dpx/1E6, 2)) + ' MPa\n' + r'$\sigma_{II}$ = '
                + str(round(dpy/1E6, 2)) + ' MPa\n'
                + r'$\alpha$ = ' + str(angle) + '°'
                + '\nC = ' + str(np.asarray(meta['layers_c'], dtype=float)/1e6) + ' MPa ; ' + r"$\mu'$ = "
                + str(np.asarray(meta['layers_cfi'], dtype=float)) + '; '
                + r'$\theta_{opt}$ = ' + str(np.round(90 - np.degrees(np.asarray(meta['layers_coula'])), 2)) + '°',
                bbox=dict(fc='white', ec='black'),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontsize=11, zorder=5)
    return figure


def plot_crack_opening(crack, v, E):
    """
    Displacements of the crack walls (formerly crack.plot_mode_I_crack_displacements):
    the component normal to the crack (opening, lines) and along it (slip, points)
    of each wall, and the out-of-plane one (mode III) if there is any.
    crack: a crack object, v: Poisson's ratio, E: Young's modulus in Pa.

    Since Nov 2026 the displacements come from crack.wall_displacements (the core
    solution, valid for modes I, II and III); until then only the mode I opening
    was drawn (Eq. 7.21 of Pollard and Martel, 2020), so a fault came out flat.
    """
    s, walls = crack.wall_displacements(v, E)
    pos, neg = walls['positive'], walls['negative']
    half = crack.extent()/2

    figure = plt.figure(figsize=(7.5, 4.2))
    ax = figure.add_subplot(111, xlabel='distance along the crack from its centre (m)', ylabel=r'u$_i$ (m)')
    # Normal to the crack (opening): lines; along the crack (slip): points
    ax.plot(s, pos['normal'], c='purple', ls='--', lw=2, label=r'u$_n$, + side')
    ax.plot(s, neg['normal'], c='purple', ls=':', lw=2, label=r'u$_n$, − side')
    ax.scatter(s, pos['tangential'], facecolor='white', edgecolor='peru', marker='o', s=36, lw=1.2,
               label=r'u$_t$, + side')
    ax.scatter(s, neg['tangential'], c='brown', marker='o', s=6, label=r'u$_t$, − side')
    # Out of the X-Y plane (mode III), only if there is any
    if np.nanmax(np.abs(pos['z'])) > 0:
        ax.plot(s, pos['z'], c='teal', ls='-', lw=1.5, label=r'u$_z$, + side')
        ax.plot(s, neg['z'], c='teal', ls='-.', lw=1.5, label=r'u$_z$, − side')
    ax.axhline(0, c='0.7', lw=0.6, zorder=0)

    # The inputs, rounded; only the driving stresses that are not zero
    stresses = [(r'$\sigma_I$', crack.dpx), (r'$\sigma_{II}$', crack.dpy), (r'$\sigma_{III}$', crack.dpz)]
    title = '\n'.join(name + ' = ' + ('%g' % round(value/1e6, 2)) + ' MPa' for name, value in stresses if value != 0)
    title = (title + '\n' if title else '') + r'$\nu$ = %g' % v + '\nE = %g GPa' % round(E/1e9, 2) \
        + '\nhalf-length = %g km' % round(half/1e3, 3)
    ax.legend(title=title, bbox_to_anchor=(1, 1), fontsize=9, title_fontsize=9)
    ax.text(0.01, 0.01, '+ side: normal n = (−t$_y$, t$_x$), t from the upper to the lower tip',
            transform=ax.transAxes, fontsize=7, color='0.4')
    return figure


def dynamic_color_range(dyn):
    """
    Colour range of sigma_n for every step of a dynamic model: the range of the
    final step, which holds every failure of the run (Sam's choice, Nov 2026),
    so that the steps can be compared. Pass it to plot_dynamic_step.
    """
    return tuple(true_min_max(dyn[-1]['failure_sn']))


def plot_dynamic_step(step, downscaling, color_range=None):
    """
    Figure of one step of the dynamic model (the drawing code formerly inside
    the loop of dynamic_model, moved unchanged in Nov 2026).
    
    step: one DiESResults of a DynamicResults (compute_dynamic_model),
    downscaling: int, keep one plane trace every downscaling nodes,
    color_range: (vmin, vmax) of sigma_n in MPa, the same for every step of a
    run (dynamic_color_range); None uses the range of this step, as until
    Nov 2026.
    
    It maps the normal stress on the failure planes (MPa), marks in red the
    nodes above failure and draws the traces of the failure planes: mode II
    under compression (red), mode II tension-compression (purple), mixed-mode
    I-II (dodgerblue) and mode I (blue), as (cos b, sin b).
    """
    _require(step, 'failure_sn', 'above_failure', 'plane_mode_I')
    # Fields and metadata of the step
    meta = step.metadata
    upper_tip, lower_tip = _crack_tips(step)
    x, y = step.x, step.y
    current_sn = step['failure_sn'] ; current_above = step['above_failure']
    percentage = meta['percentage'] ; length = meta['length'] ; length_label = meta['length_label']
    step_dx = meta['dx'] ; step_dy = meta['dy']
    
    figure = create_figure(meta['x_limits'], meta['y_limits'])
    ax = figure.axes[0]
    ax.plot((upper_tip[0], lower_tip[0]), (upper_tip[1], lower_tip[1]), c='red', lw=3, label='Crack')
    
    # Plotting stresses, with the colour range of the run (or of this step)
    vmin, vmax = color_range if color_range is not None else true_min_max(current_sn)
    plot = ax.pcolormesh(x, y, current_sn, cmap='viridis', vmin=vmin, vmax=vmax, shading='nearest')
    cbar = figure.colorbar(plot, extend='both', format='%.2f')
    cbar.set_label(r'$\sigma_n$ (MPa)', rotation=270, labelpad=20, fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    ax.contour(x, y, current_sn, levels=[0], colors='black', linewidths=0.75, linestyles='--')
    
    # Plotting the gridcells which are found above failure in each timestep
    ax.pcolormesh(x, y, current_above, cmap='Reds_r', shading='nearest')
    
    # Plotting downsampled versions of the failure plane orientations, one every
    # downscaling nodes in each direction
    trace_spacing = _cell_size(step)*downscaling
    ds = {name: downsample_2darray(step[name], downscaling) for name in
          ('plane_mixed_right', 'plane_tc_right', 'plane_comp_right',
           'plane_mixed_left', 'plane_tc_left', 'plane_comp_left', 'plane_mode_I')}
    
    # Plane traces at orientation b (counterclockwise from +x) are drawn as (cos b, sin b)
    for name, colour in (('plane_comp_right', 'red'), ('plane_comp_left', 'red'),          # mode II, compression
                         ('plane_tc_right', 'purple'), ('plane_tc_left', 'purple'),      # mode II, tension-compression
                         ('plane_mixed_right', 'dodgerblue'), ('plane_mixed_left', 'dodgerblue'),  # mixed mode
                         ('plane_mode_I', 'blue')):                                     # mode I
        # Traces 0.9 times the spacing between the traces drawn (Nov 2026; until then a fixed 250 m)
        ax.quiver(x, y, np.cos(ds[name]), np.sin(ds[name]), headlength=0, headaxislength=0, pivot='middle',
                  scale=1/(0.9*trace_spacing), units='xy', width=0.08*trace_spacing, headwidth=1, color=colour)
    
    # Box with the aperture or slip and the driving stresses reached at this step,
    # anchored at its bottom inside the axes (until Nov 2026 it ran over the x labels)
    ax.text(0.97, 0.03, length_label + '·' + str(round(percentage, 4)) + ' = ' + str(round(length, 2)) + ' m\n' +
            r'$\sigma_I$·' + str(round(percentage, 4)) + ' = ' + str(round(step_dx/1e6, 2)) + ' MPa\n' +
            r'$\sigma_{II}$·' + str(round(percentage, 4)) + ' = ' + str(round(step_dy/1e6, 2)) + ' MPa',
            bbox=dict(fc='white', ec='black'),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=11, zorder=5)
    return figure


def plot_mohr_circle(p, s, cfi, T):
    """
    Mohr circle of a stress state with the modified Griffith (parabola) and
    Coulomb (line) criteria, and the point of failure if there is one
    (the drawing code formerly inside evaluate_stress_state, moved unchanged
    in Nov 2026). The values come from evaluate_stress_state(..., 'values').
    
    p, s: maximum and minimum compressive principal stresses, in Pa (p > s),
    cfi: coefficient of internal friction, T: tensile strength in Pa (< 0).
    The diagram is drawn in MPa, compression positive.
    """
    # Imported here: Failure_Determination imports DiES_core, which imports this module
    from Failure_Determination import evaluate_stress_state
    from Helper_functions import circle_eq_y
    
    sn, ss, angle, mode = evaluate_stress_state(p, s, cfi, T, 'values')
    if mode == 'Singular' or p == s:
        raise PlotError('Cannot draw a Mohr circle for this state (mode ' + mode + ', p = ' + str(p) + ', s = ' + str(s) + ')')
    
    p = p/1e6 ; s = s/1e6 ; T = T/1e6
    r = abs(p - s)/2 ; x_center = p - (p - s)/2
    circle_points = np.linspace(s, p, int(r)*50)
    
    # Variables for plotting
    x_boundaries = np.asarray((x_center - r*2, x_center + r*2))
    y_boundaries = np.asarray((r*1, -r*2))
    x_values_G = np.linspace(T, 0, 200)
    x_values_MC = np.linspace(0, x_boundaries[1], 200)
    
    # Axis elements
    figure = plt.figure(figsize=(5, 4))
    ax = figure.add_subplot(111, xlim=x_boundaries, ylim=-y_boundaries)
    ax.set_ylabel(r'$\sigma_s$ (MPa)')
    ax.set_xlabel(r'$\sigma_n$ (MPa)')
    ax.hlines(0, x_boundaries[0], x_boundaries[1], color='black', lw=0.5)
    ax.vlines(0, y_boundaries[0]*2, y_boundaries[1], color='black', lw=0.5)
    
    # The Mohr circle and its defining parameters, coloured by failure mode
    color = {'Below': 'black', 'Above': 'black', 'Mode II - compression': 'red',
             'Mode II - tension-compression': 'purple', 'Mixed-mode I-II': 'dodgerblue',
             'Mode I': 'blue'}[mode]
    ax.plot(circle_points, circle_eq_y(x_center, r, circle_points)[0], lw=3, c=color, zorder=3)
    ax.scatter(x_center, 0, c='black', s=30)
    ax.scatter(p, 0, lw=0.5, edgecolor='black', c=color, s=70, zorder=3)
    ax.scatter(s, 0, lw=0.5, edgecolor='black', c=color, s=70, zorder=3)
    
    # The modified Griffith criterion and the Coulomb line
    y_values = (abs(4*T*(x_values_G - T)))**0.5
    ax.plot(x_values_G, y_values, ls='--', lw=3, c='darkred')
    y_values = -(abs(4*T*(x_values_G - T)))**0.5
    ax.plot(x_values_G, y_values, ls='--', lw=3, c='darkred')
    y_values = -2*T + cfi*x_values_MC
    ax.plot(x_values_MC, y_values, ls='--', lw=3, c='tomato')
    y_values = -2*T + cfi*x_values_MC
    ax.plot(x_values_MC, -y_values, ls='--', lw=3, c='tomato')
    
    # The contact point, only if there is any kind of failure
    if mode not in ('Below', 'Above'):
        ax.scatter(sn, ss, marker='*', edgecolor='black', c='orange', s=200, zorder=3)
        ax.plot((x_center, sn), (0, ss), ls='--', lw=1.5, c='black')
        ax.text(sn + r, ss + r*1, r'2$\theta$ = ' + str(round(angle, 2)), bbox=dict(fc='white', ec='black'), fontsize=12)
    
    ax.text(p - r/6, 0 - r/1, r'$\sigma_1$ = ' + str(round(p, 2)) + ' MPa', bbox=dict(fc='white', ec='black'), fontsize=12)
    ax.text(s - r/0.5, 0 - r/1, r'$\sigma_3$ = ' + str(round(s, 2)) + ' MPa', bbox=dict(fc='white', ec='black'), fontsize=12)
    return figure
