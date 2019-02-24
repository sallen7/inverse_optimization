# Inverse Optimization Methods for Pyomo #

## Description ##

This project implements state-of-the-art inverse optimization (IO) methods found in the literature. 

Note: We reference a "chapter" at times in the files. This refers to the extensive documentation we have written to go along-side the code.  We are not publicly releasing the chapter at this moment, but it is something about which interested parties can contact the researcher.

* In the `GIO_files` folder, we have the `GIO` class which implements methods from Chan et al.'s 2018 paper on generalized inverse optimization.  The class allows users to create a `GIO` object that will hold data and has a series of methods to perform IO functionality found in the Chan et al. paper.  The full citation for the paper can be found here:

	Chan, Timothy CY, Taewoo Lee, and Daria Terekhov. 
	"Inverse Optimization: Closed-Form Solutions, Geometry, 
	and Goodness of Fit." Management Science (2018).

	* The `gio.py` file within the `gio_source` folder contains the GIO class which allows users to create a GIO object that will hold data and has a series of methods to perform IO functionality found in the paper.
	* The `testing_gio_generating_epsilon_c` folder contains the validation scripts for the GIO model methods within the `GIO` class.
	* The `testing_gio_rho` folder contains the validation scripts for the coefficient of complementarity (rho) methods.




## Specifications for Set-up of Code ##

Information regarding relevant packages needed for the use of this code will be provided in the near future.


   

