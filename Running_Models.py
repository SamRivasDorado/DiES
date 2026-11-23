import sys
from Modeling_functions import single_model
from Modeling_functions import dynamic_model

################### Single model ################### 

# Quick instructions

# To run a static model of the displacements or stresses induced by a crack 
# in an elastic medium. The crack may be an opening/closing fracture or a fault. 

# Fill the following variables and see the single_model function docstring 
# for extra details if needed. Then run the code only up to the sys.exit() line.
    
# The result is a single figure plotting the desired parameter.


crack_inputs=['crack total aperture','x coordinate of the upper crack tip','y coordinate of the upper crack tip'
              ,'x coordinate of the lower crack tip','y coordinate of the lower crack tip']

# If the model is of a fault, aperture should be specified to 0.
# The total height of the crack is calculated from the coordinates of its tips.

model_inputs=['Youngs modulus','poissons ratio','density','coefficient of internal friction','tensile strength']
# The cohesion is not an input: it is implicit in the modified Griffith criterion as C = 2|T|

dx='driving stress orthogonal to the crack, only needed if crack aperture is 0'

dy='driving stress parallel to the crack' ; dz='driving stress into the X-Y plane'

rx='constant remote stress along x' ; ry='constant remote stress along y'

g='gravitational acceleration' ; litho='True or False: a boolean, not a str'
# These do not have any effect in the displacements

angle='An angle from 0 to 90'

xlims=['min x','max x'] ; ylims=['min y must be negative and smaller than max y','max y, must be negative']

resolution=['resolution in x','resolution in y']

parameter='The parameter to plot, can be ux, uy, uxy, sx, sy, sxy, sav, t, s1, s3, st, sn, ss'

model=single_model(crack_inputs, model_inputs, dx, dy, dz, rx,ry,g,litho,angle,xlims,ylims,resolution,parameter)


sys.exit()

################### Dynamic model ################### 

# Quick instructions

# To run a dynamic model of the discontinuities formed as a consequence of the
# opening or the slip of a crack, when its driving stresses grow from a small
# fraction to a larger fraction of their final values. This applies to opening
# fractures such as dikes, but also to faults, dilatant faults, and compaction bands.

# Fill each of these variables with the desired parameters and check the dynamic_model
# function docstring for more details. Then run the code only from this block,
# i.e., with the single model block above commented out.

# The result is a series of figures saved in the desired
# folder (if plotting is True) which show the evolution of the discontinuities formed.

crack_inputs=['crack total aperture','x coordinate of the upper crack tip','y coordinate of the upper crack tip'
              ,'x coordinate of the lower crack tip','y coordinate of the lower crack tip']

# If the model is of a fault, aperture should be specified to 0.
# The total height of the crack is calculated from the coordinates of its tips.

model_inputs=['Youngs modulus','poissons ratio','density','coefficient of internal friction','tensile strength']
# The cohesion is not an input: it is implicit in the modified Griffith criterion as C = 2|T|

dx='final driving stress orthogonal to the crack, only needed if crack aperture is 0. >0 for a dilatant fault, <0 for a compaction band, 0 for pure shearing'

dy='final driving stress parallel to the crack' ; dz='final driving stress into the X-Y plane'
# dx, dy, and dz all grow together with each step of the model

rx='constant remote stress along x' ; ry='constant remote stress along y'

g='gravitational acceleration' ; litho='True or False: a boolean, not a str'

investigation_range=['fraction of the final driving stresses in step 1, between 0 and 1','fraction in step 2','...']

xlims=['min x','max x'] ; ylims=['min y must be negative and smaller than max y','max y, must be negative']

resolution=['resolution in x','resolution in y'] ; downscaling='An integer value here'

plotting='True or False: a boolean, not a str'

folder='Type here the desired folder'

current_mode=dynamic_model(crack_inputs,model_inputs,dx,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,downscaling,plotting,folder)
