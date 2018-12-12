Stephanie Allen
AMSC663: Inverse Optimization Methods
Updated: 12/12/2018

(A) DESCRIPTION
This project implements state-of-the-art inverse optimization
(IO) methods found in the literature.  Currently, we are focusing
on Chan et al.'s 2018 paper on generalized inverse optimization,
whose full citation is as follows:

Chan, Timothy CY, Taewoo Lee, and Daria Terekhov. 
"Inverse Optimization: Closed-Form Solutions, Geometry, 
and Goodness of Fit." Management Science (2018).

Note: We reference a "chapter" at times in the files.
This refers to the extensive documentation we have written
to go along-side the code.  We are not publicly releasing the 
chapter at this moment, but it is something about which interested
parties can contact the researcher.

(B) MAIN CODE FILES
The gio.py file contains the GIO class which allows users
to create a GIO object that will hold data and has a 
series of methods to perform IO functionality found in the paper.  

(C) VALIDATION FILES
There are a series of files that all start with "testing".
They are as follows:

-- testing_gio.py includes unit tests for the GIO model methods
based on Example 1 from Chan et al.'s 2018 paper
-- testing_gio_pyomo_params unit tests are the same as 
testing_gio.py just with pyomo parameter data instead of
numpy parameter data
-- testing_rho_p.py produces heatmaps for the calculate_rho_p
and calculate_rho_p_approx methods for the purposes of 
validation rho_p and rho_p_approx
-- testing_rho_a_and_rho_r.py produces heatmaps for the
calculate_rho_a and calculate_rho_r methods
-- testing_gio_structural_epsilon.py has the unittests for the
structural constraints on epsilon workflow methods

(D) DEMO FILES
-- demo_gio_class.ipynb Jupyter notebook shows some of 
the functionality of the GIO class 

(E) DATA/VISUALS
-- inf_norm_rho_approx_mesh.npy was used to validate the 
calculate_rho_r method
-- rho_heat_maps folder contains heatmap output from the 
testing_rho_p and testing_rho_a_and_rho_r scripts
-- Steph_Test_Ex.jpg is an image for the Jupyter notebook 








