"""
The scenario the app opens with: the vertical dike of examples/dike_example.yaml (the
control case of the tests), copied here because examples/ is not installed with the
package. tests/test_app_state.py checks that both stay the same.
"""

DEFAULT_SCENARIO = {
    'name': 'Vertical dike, opening only',
    'description': 'Control case of the tests (Rivas-Dorado et al., 2023, example model), Earth gravity.',
    'grid': {'x': ['-5 km', '5 km'], 'y': ['-6 km', '-10 m'], 'cell': ['100 m', '100 m']},
    'gravity': '9.81 m/s2',
    'layers': [{'name': 'host rock', 'top': '0 m', 'E': '30 GPa', 'nu': 0.25, 'density': '2900 kg/m3',
                'friction': 0.57, 'T': '-1.5 MPa'}],
    'cracks': [{'name': 'dike', 'upper_tip': ['0 m', '-1 km'], 'lower_tip': ['0 m', '-5 km'],
                'aperture': '2 m'}],
    'remote_stresses': {'xx': '0 MPa', 'yy': '0 MPa'},
    'lithostatic': True,
    'analysis': {'type': 'dynamic', 'step': '0.25 MPa', 'plane_angle': '60 deg'},
    'output': {'results': ['npz', 'csv'],
               'figures': {'static': ['uxy', 'sxy', 's1', 'angle', 'trajectories', 'plane_normal',
                                      'crack_displacements'],
                           'dynamic': {'show_every': 4, 'downscaling': 2}},
               'format': 'png', 'dpi': 200},
}

# Name shown in the sidebar for the default scenario
DEFAULT_SOURCE = 'Example: vertical dike (examples/dike_example.yaml)'
