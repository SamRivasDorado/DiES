import sys
from Modeling_functions import single_model
from Modeling_functions import dynamic_model

################### Single model ################### 

# To run a static model of what would be the displacements or stresses induced
# induced by a crack in an elastic medium. The crack may be an opening fracture
# or a fault, and may be vertical or dipping. Fill the following variables according
# to the single_model function docstring and run. The result is a single figure
# plotting the desired parameter.


crack_inputs=['crack aperture','crack height','x coordinate of the upper crack tip','y coordinate of the upper crack tip'
              ,'x coordinate of the lower crack tip','y coordinate of the lower crack tip']
# If the model is of a dike, height and the latter two parameters do not have any effect

model_inputs=['Youngs modulus','poissons ratio','density','cohesion','coefficient of internal friction','tensile strength']

dx='driving stress orthogonal to the crack, only needed if crack aperture is 0'

dy='driving stress parallel to the crack' ; dz='driving stress into the X-Y plane'

rx='constant remote stress along x' ; ry='constant remote stress along y'

g='gravitational acceleration' ; litho='A boolean here' 
# These do not have any effect in the displacements

angle='An angle from 0 to 90'

xlims=['min x','max x'] ; ylims=['min y must be negative and smaller than max y','max y']

resolution=['resolution in x','resolution in x']

parameter='The parameter to plot'

model=single_model(crack_inputs, model_inputs, dx, dy, dz, rx,ry,g,litho,angle,xlims,ylims,resolution,parameter)

sys.exit()

################### Dynamic model ################### 

# To run a dynamic model of the discontinuities formed as a consequence of dike
# opening, when dike opens from X to Y % of its final aperture. Fill each of these 
# variables with the desired parameters and according to the dynamic_model 
# function docstring. The result is a series of figures saved in the desired
# folder which show the evolution of the discontinuities formed.

dike_inputs=['dike aperture','dike height','x coordinate of the upper crack tip','y coordinate of the upper crack tip'
             ,'x coordinate of the lower crack tip','y coordinate of the lower crack tip']

model_inputs=['Youngs modulus','poissons ratio','density','cohesion','coefficient of internal friction','tensile strength']

py='driving stress parallel to the dike' ; pz='driving stress into the X-Y plane'

rx='constant remote stress along x' ; ry='constant remote stress along y'

g='gravitational acceleration' ; litho='A boolean here'

investigation_range=['min %','max %']

xlims=['min x','max x'] ; ylims=['min x','max x']

resolution=['resolution in x','resolution in x'] ; downscaling='An integer value here'

folder='Type here the desired folder'

dynamic_model(dike_inputs,model_inputs,dy,dz,rx,ry,g,litho,investigation_range,xlims,ylims,resolution,downscaling,folder)