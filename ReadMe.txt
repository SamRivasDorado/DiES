DiES: Discontinuities in Elastic Space, is a piece of phython code which allows to model the displacements and stresses induced by
dilatant or closing cracks or faults in an elastic space. This includes dikes, compaction bands, or faults, which can be placed
in single or multi-layer models, for which the both elastic and frictional properties are specified. 

The files contained in this repository are:
· DiES_core.py: contains what is needed to construct the key objects (the cracks and the model classes), and calcualte displacements and stresses
· Modeling_functions.py: contains the functions which run the two key types of models (described below) which can be run with the code in the current version
· Helper_functions.py: contains multiple functions to perform miscelaneous operations used in other parts of the program
· Running_Models.py: contains the simplified instructions to run the two types of models. THIS IS THE USER FILE.

As stated above, the current code is designed so that it is useful to perform two types of models:
  1 - Model the displacements and stresses induced by a crack, which may be opening, closing, and/or shearing, due to tractions acting orthogonal and/or parallel to it,
  in a single-layer model with specific properties (mainly, E, v, and density). This allows to calculate and plot:
      -The crack-realted displacements along the x axis
      -The crack-realted displacements along the y axis
      -The xy crack-realted displacements
      -The crack-induced stresses along the x axis
      -The crack-induced stresses along the x axis
      -The xy crack-induced stresses
      -The average crack-induced stresses
      -The shear crack-induced stresses
      -The maximum principal stress, sigma1
      -The minimum principal stress, sigma3
      -The normal stresses if resolved onto planes of a specified angle
      -The shear stresses if resolved onto planes of a specified angle
     Please read the intructions in Running_Models.py and the docstrings in Modeling_functions.py for all the details, but some things to note here. The code allows to 
     switch on and off the calculation of lithostatic stresses. These only take part in the stress but not in the displacement calculations, so having them 'on' or 'off' 
     does not the latter. There are additional parameters which can be manipulated within the DiES_core.py file for a better visualization, which may not always come 
     up with the default parameters. This include contour spacing in certain plots, vectors' scaling, visualizing the optimal failure planes about the stress trajectories, 
     etc. Feel free to play with this in the core file, but do so at your own risk. The code works best for vertical cracks and faults. Dipping 
     faults will show the right displacements but the stress calculations
     are at the moment innacurate.
     Time-permitting, I will be improving the wrong and/or incomplete aspects of the code, as well as its accesibility. 
 
 2 - Model the discontinuities which form as a consequence of the opening of a dike which final aperture is specified, in a single-layer model in which this time
     frictional properties are also key (coefficient of internal friction, and tensile strength). In this case, there is only one outcome: the result of this model 
     will be a series of figures saved in your folder of choice, which show how different discontinuities are formed as a dike opens in the subsurface, ideally, 
     between a very small % and another % of your choice. These are % of the total final opening that you define for the dike. The color codes of the discontinuities 
     can be found in the associated paper to this code. As above, follow the simplified instructions in Running_Models.py and the docstrings in  Modeling_functions.py. 
     
More details on the methodology behind these models can be found in the paper 'Modeling of dike-induced graben nucleation in the Elysium region, Mars: the role of 
planetary gravity'. The first type of models are adressed by the first section of the methodology. The displacement equations are not found there but the user may be
referred to Pollard and Segall, 1987 for the full description. The second types of models are fully described in the second section of the methodology. 
