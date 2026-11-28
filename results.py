"""
DiESResults: the results of evaluating a DiES model (grid + crack) once.

Created on 2026-11-24 (Phase 1 of the roadmap: separate computation, plotting
and input/output). A DiESResults object holds every field computed on the grid,
so that figures, exports and the future application read them instead of
recomputing them.

Example
-------
    res = current_model.compute(disc, rxx, ryy, litho=True)
    res.s1                      # 2D array, Pa
    res['sxy']                  # the same access by name
    res.units['s1a']            # 'deg'
    res.probe(500, -1200)       # every field at the nearest node
    res.profile((-3000, -1500), (3000, -1500))   # every field along A -> B
    res.save('model.npz') ; DiESResults.load('model.npz')
    res.to_csv('model.csv')
    res.to_xarray()             # optional, only if xarray is installed

Conventions (see docs/crack_coordinates_and_rotation.md)
-------------------------------------------------------
- Coordinates and displacements in m, stresses in Pa, angles in degrees.
- Stress components (sxx, syy, sxy, sxz, szz, sav, tmax) are tension positive and
  include the remote and, if litho=True, the lithostatic stresses, as returned
  by model.model_stresses.
- Principal stresses (s1, s3) and the stresses on the optimal planes (sn, ss) are
  compression positive. s1a and s3a are counterclockwise from +x, in (-90, 90].
- All 2D arrays have the shape of the grid: (number of y nodes, number of x nodes),
  with x varying along the columns and y along the rows.
"""

import csv
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Name -> (units, description) of every field DiESResults may hold
FIELD_INFO = {
    # Material at each node (from model.grid)
    'E': ('Pa', "Young's modulus"),
    'nu': ('', "Poisson's ratio"),
    'density': ('kg m-3', 'density'),
    'cohesion': ('Pa', 'cohesion, implicit as 2|T|'),
    'friction': ('', 'coefficient of internal friction'),
    'coula': ('rad', 'angle between sigma1 and the optimal shear planes, 0.5*arctan(1/friction)'),
    'T': ('Pa', 'tensile strength (negative)'),
    # Displacements
    'ux': ('m', 'displacement along x'),
    'uy': ('m', 'displacement along y'),
    'uz': ('m', 'displacement along z'),
    # Stresses, tension positive
    'sxx': ('Pa', 'normal stress on x planes, tension positive'),
    'syy': ('Pa', 'normal stress on y planes, tension positive'),
    'sav': ('Pa', 'average in-plane normal stress (sxx + syy)/2, tension positive'),
    'sxy': ('Pa', 'shear stress on x planes along y, tension-positive convention'),
    'sxz': ('Pa', 'shear stress on x planes along z'),
    'szz': ('Pa', 'normal stress on z planes (plane strain), tension positive'),
    'tmax': ('Pa', 'maximum in-plane shear stress'),
    # Principal stresses, compression positive
    's1': ('Pa', 'maximum compressive principal stress, compression positive'),
    's3': ('Pa', 'minimum compressive principal stress, compression positive'),
    's1a': ('deg', 'orientation of sigma1, counterclockwise from +x, in (-90, 90]'),
    's3a': ('deg', 'orientation of sigma3, counterclockwise from +x, in (-90, 90]'),
    # Stresses on the optimal shear planes, compression positive
    'sn': ('Pa', 'normal stress on the optimal shear planes, compression positive'),
    'ss': ('Pa', 'shear stress on the optimal shear planes'),
    # Failure state of each node in dynamic_model (one DiESResults per step)
    'failure_sn': ('MPa', 'normal stress on the failure plane at failure (first failure is kept), compression positive'),
    'failure_ss': ('MPa', 'shear stress on the failure plane at failure (first failure is kept)'),
    'failure_theta': ('deg', 'angle between sigma1 and the normal to the failure plane'),
    'failure_mode': ('', "failure mode: 'Below', 'Above', 'Mode I', 'Mixed-mode I-II', "
                         "'Mode II - tension-compression', 'Mode II - compression' or 'Singular'"),
    'above_failure': ('', '0 where the node is above failure, NaN elsewhere (as drawn in red)'),
    # Failure state of a single model (Modeling_functions.compute_failure_state, Phase 4)
    'failure_ratio': ('', 'radius of the Mohr circle over the critical radius r/r_c (1: touching the envelope)'),
    'failure_contact_mode': ('', 'mode of the point where the Mohr circle would touch the envelope'),
    'failure_jumped': ('', '1 where the node jumped from below to above failure in one step and its failure '
                           'values are those of the contact with the envelope; 0 elsewhere'),
    'plane_mixed_left': ('rad', 'orientation of mixed-mode I-II planes, left of the crack, counterclockwise from +x'),
    'plane_tc_left': ('rad', 'orientation of mode II tension-compression planes, left of the crack'),
    'plane_comp_left': ('rad', 'orientation of mode II compression planes, left of the crack'),
    'plane_mixed_right': ('rad', 'orientation of mixed-mode I-II planes, right of the crack'),
    'plane_tc_right': ('rad', 'orientation of mode II tension-compression planes, right of the crack'),
    'plane_comp_right': ('rad', 'orientation of mode II compression planes, right of the crack'),
    'plane_mode_I': ('rad', 'orientation of mode I planes (they contain sigma1)'),
}

# Failure modes that mean that a node has failed (kept in later steps)
FAILED_MODES = ('Mode I', 'Mixed-mode I-II', 'Mode II - tension-compression', 'Mode II - compression', 'Above')


# Names changed since results were first saved: files written before the change are
# read with the new names (Nov 2026: cfi -> friction, the coefficient of internal friction)
RENAMED_FIELDS = {'cfi': 'friction'}
RENAMED_METADATA = {'layers_cfi': 'layers_friction'}


def _renamed(mapping, names):
    """A copy of a dict with its old keys replaced by the new ones (RENAMED_*)."""
    return {mapping.get(key, key): value for key, value in names.items()}


class DiESResults(object):
    """
    Container for the fields computed on a DiES grid.

    Parameters
    ----------
    x, y : 2D arrays with the coordinates of the grid nodes, in m.
    fields : dict, name -> 2D array with the shape of x and y.
    metadata : dict with the description of the model and the crack (limits,
        layers, crack tips, driving stresses, remote stresses, litho, g...).
        It must be serialisable to JSON (numbers, strings, lists, dicts).
    units : optional dict, name -> units. Defaults to FIELD_INFO.
    """

    def __init__(self, x, y, fields, metadata=None, units=None):
        # Node coordinates of the grid, as float arrays
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        # Every field must be a 2D array with the shape of the grid
        self.fields = {}
        for name, values in fields.items():
            values = np.asarray(values)
            if values.shape != self.x.shape:
                raise ValueError('Field ' + name + ' has shape ' + str(values.shape)
                                 + ', but the grid has shape ' + str(self.x.shape))
            self.fields[name] = values
        # A copy of the metadata, so that later changes to the input dict do not leak in
        self.metadata = dict(metadata or {})
        # Units from FIELD_INFO ('' for unknown fields), overridden by those given
        self.units = {name: FIELD_INFO.get(name, ('', ''))[0] for name in self.fields}
        if units:
            self.units.update(units)

    # ------------------------------------------------------------------ access

    def __getattr__(self, name):
        # Called only when normal attribute lookup fails: res.s1 -> res.fields['s1']
        fields = self.__dict__.get('fields', {})
        if name in fields:
            return fields[name]
        raise AttributeError("DiESResults has no field or attribute '" + name + "'")

    def __getitem__(self, name):
        # res['s1']: a field by name
        return self.fields[name]

    def __contains__(self, name):
        # 's1' in res: whether the field exists
        return name in self.fields

    def names(self):
        """Names of the fields held, in insertion order."""
        return list(self.fields)

    @property
    def shape(self):
        # Shape of the grid: (number of y nodes, number of x nodes)
        return self.x.shape

    def x_nodes(self):
        """1D array with the x coordinate of each column of the grid."""
        return self.x[0, :]

    def y_nodes(self):
        """1D array with the y coordinate of each row of the grid."""
        return self.y[:, 0]

    def __repr__(self):
        ny, nx = self.shape
        return ('DiESResults(' + str(ny) + ' x ' + str(nx) + ' nodes; x from '
                + str(self.x.min()) + ' to ' + str(self.x.max()) + ' m, y from '
                + str(self.y.min()) + ' to ' + str(self.y.max()) + ' m; fields: '
                + ', '.join(self.names()) + ')')

    # ------------------------------------------------------- DiES operations

    def probe(self, x, y, names=None):
        """
        Values of the fields at the grid node nearest to (x, y).

        Returns a dict with 'x' and 'y' (the coordinates of that node) and one
        entry per field (all fields, or only those in names).
        """
        names = self.names() if names is None else list(names)
        # Row and column of the node closest to (x, y)
        distance = np.hypot(self.x - x, self.y - y)
        i, j = np.unravel_index(np.nanargmin(distance), distance.shape)
        values = {'x': float(self.x[i, j]), 'y': float(self.y[i, j])}
        for name in names:
            values[name] = self.fields[name][i, j].item()
        return values

    def profile(self, A, B, n=200, names=None):
        """
        Values of the fields along the straight line from A = (xa, ya) to
        B = (xb, yb), sampled at n equally spaced points and interpolated
        bilinearly between the grid nodes.

        Returns a dict with 'distance' (from A, in m), 'x', 'y' and one 1D
        array per field. Points outside the grid get NaN. Angles (s1a, s3a)
        are interpolated as numbers, so they are not meaningful where the
        orientation wraps around +/-90; use probe() or the nodes for those.
        """
        names = self.names() if names is None else list(names)
        # n points along A -> B, as fractions t of the segment
        t = np.linspace(0, 1, int(n))
        xs = A[0] + t*(B[0] - A[0]) ; ys = A[1] + t*(B[1] - A[1])
        # The interpolator takes (y, x) pairs, in the order of the array axes (rows, columns)
        points = np.column_stack([ys, xs])
        out = {'distance': t*np.hypot(B[0] - A[0], B[1] - A[1]), 'x': xs, 'y': ys}
        for name in names:
            values = np.asarray(self.fields[name], dtype=float)
            interpolator = RegularGridInterpolator((self.y_nodes(), self.x_nodes()), values,
                                                   method='linear', bounds_error=False, fill_value=np.nan)
            out[name] = interpolator(points)
        return out

    def mask(self, condition, name):
        """
        Copy of a field with NaN where condition is False, e.g.
        res.mask(res.s3 < 0, 's1') keeps sigma1 only where sigma3 is tensile.
        """
        return np.where(condition, self.fields[name], np.nan)

    # -------------------------------------------------------- input / output

    def save(self, path):
        """
        Saves the results to a compressed .npz file: the coordinates, every
        field, and the units and metadata as JSON text. Reopen with load().
        """
        # Fields are stored with the prefix 'field_'; units and metadata as JSON text,
        # so that the file can be read without pickle
        arrays = {'x': self.x, 'y': self.y}
        for name, values in self.fields.items():
            arrays['field_' + name] = values
        arrays['units_json'] = np.array(json.dumps(self.units))
        arrays['metadata_json'] = np.array(json.dumps(self.metadata, default=_to_json))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        """Opens results saved with save()."""
        with np.load(path, allow_pickle=False) as data:
            # Every array saved as 'field_<name>' becomes the field <name>
            fields = {key[len('field_'):]: data[key] for key in data.files if key.startswith('field_')}
            units = json.loads(str(data['units_json']))
            metadata = json.loads(str(data['metadata_json']))
            x, y = data['x'], data['y']
        # Files saved before a field or metadata key was renamed (e.g. cfi -> friction)
        return cls(x, y, _renamed(RENAMED_FIELDS, fields), _renamed(RENAMED_METADATA, metadata),
                   _renamed(RENAMED_FIELDS, units))

    def to_csv(self, path, names=None):
        """
        Writes one row per grid node with x, y and the fields (all, or those in
        names). The header gives the units in brackets, e.g. 's1 [Pa]'.
        """
        names = self.names() if names is None else list(names)
        # One column per quantity, flattened so that each row is one node
        columns = [self.x.ravel(), self.y.ravel()] + [self.fields[name].ravel() for name in names]
        header = ['x [m]', 'y [m]'] + [name + ' [' + self.units.get(name, '') + ']' for name in names]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*columns))

    def to_xarray(self):
        """
        Optional conversion to an xarray.Dataset (requires xarray, which DiES
        does not need otherwise), e.g. to write NetCDF files.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError('to_xarray() needs the xarray package: pip install xarray')
        data_vars = {name: (('y', 'x'), values, {'units': self.units.get(name, ''),
                                                 'description': FIELD_INFO.get(name, ('', ''))[1]})
                     for name, values in self.fields.items()}
        attrs = {'metadata_json': json.dumps(self.metadata, default=_to_json)}
        return xr.Dataset(data_vars, coords={'x': self.x_nodes(), 'y': self.y_nodes()}, attrs=attrs)


class DynamicResults(object):
    """
    Results of model_functions.compute_dynamic_model: one DiESResults per step
    (Phase 1, 2026-11-24).
    
    Each step holds failure_sn, failure_ss, failure_theta, failure_mode,
    above_failure and the seven plane_* orientations (see FIELD_INFO), and its
    metadata describes the model, the crack of that step and the step itself:
    'step', 'n_steps', 'percentage', 'dx', 'dy', 'dz' (driving stresses of the
    step, Pa), 'length' and 'length_label' (aperture 'a' or slip 'u', m).
    
    Example
    -------
        dyn = compute_dynamic_model(...)
        len(dyn) ; dyn[-1]                 # number of steps ; last step
        dyn.final_mode()                   # failure mode at the last step
        dyn.failure_step()                 # step at which each node first failed
        dyn.save('dynamic.npz') ; DynamicResults.load('dynamic.npz')
    """
    
    def __init__(self, steps, metadata=None):
        self.steps = list(steps)
        self.metadata = dict(metadata or {})
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, index):
        return self.steps[index]
    
    def __iter__(self):
        return iter(self.steps)
    
    def __repr__(self):
        shape = self.steps[0].shape if self.steps else None
        return 'DynamicResults(' + str(len(self)) + ' steps on a grid of ' + str(shape) + ' nodes)'
    
    def final_mode(self):
        """2D array with the failure mode of each node at the last step."""
        return self.steps[-1]['failure_mode']
    
    def failure_step(self):
        """
        2D array with the number of the step (1, 2, ...) at which each node first
        reached a failed state (any mode in FAILED_MODES, including 'Above'), and
        0 where it never did.
        """
        first = np.zeros(self.steps[0].shape, dtype=int)
        for k, step in enumerate(self.steps, start=1):
            # Nodes failed at this step; only those not yet marked get the number k
            failed = np.isin(step['failure_mode'].astype(str), FAILED_MODES)
            first = np.where((first == 0) & failed, k, first)
        return first
    
    def step_loads(self):
        """
        The load of each step (Phase 4, Nov 2026): the largest of its three driving
        stresses |dx|, |dy|, |dz|, in Pa, as a 1D array (one value per step). It is the
        driving stress the step and steps settings of a scenario refer to.
        """
        return np.array([max(abs(step.metadata.get(name, 0.0)) for name in ('dx', 'dy', 'dz'))
                         for step in self.steps])

    def failure_load(self, upto=None):
        """
        2D array with the load (step_loads, Pa) at which each node first failed, and NaN
        where it never did (Phase 4, Nov 2026). upto: only the failures of the first upto
        steps (1, 2, ...); None for the whole run.
        """
        first = self.failure_step()
        if upto is not None:
            first = np.where(first <= upto, first, 0)
        loads = np.concatenate(([np.nan], self.step_loads()))
        return loads[first]

    def save(self, path):
        """Saves every step to one compressed .npz file; reopen with load()."""
        if not self.steps:
            raise ValueError('There are no steps to save')
        # All steps share the grid; each field is stored as 'step<k>_<name>'
        arrays = {'x': self.steps[0].x, 'y': self.steps[0].y}
        for k, step in enumerate(self.steps):
            for name, values in step.fields.items():
                arrays['step' + str(k) + '_' + name] = values
        arrays['n_steps'] = np.array(len(self.steps))
        arrays['units_json'] = np.array(json.dumps([step.units for step in self.steps]))
        arrays['step_metadata_json'] = np.array(json.dumps([step.metadata for step in self.steps], default=_to_json))
        arrays['metadata_json'] = np.array(json.dumps(self.metadata, default=_to_json))
        np.savez_compressed(path, **arrays)
    
    @classmethod
    def load(cls, path):
        """Opens dynamic results saved with save()."""
        with np.load(path, allow_pickle=False) as data:
            n = int(data['n_steps'])
            units = json.loads(str(data['units_json']))
            step_meta = json.loads(str(data['step_metadata_json']))
            steps = []
            # Rebuild one DiESResults per step from its 'step<k>_' arrays
            for k in range(n):
                prefix = 'step' + str(k) + '_'
                fields = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
                # Files saved before a field or metadata key was renamed (e.g. cfi -> friction)
                steps.append(DiESResults(data['x'], data['y'], _renamed(RENAMED_FIELDS, fields),
                                         _renamed(RENAMED_METADATA, step_meta[k]), _renamed(RENAMED_FIELDS, units[k])))
            return cls(steps, json.loads(str(data['metadata_json'])))


def _to_json(value):
    """Converts numpy values for json.dumps."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Not serialisable to JSON: ' + repr(value))
