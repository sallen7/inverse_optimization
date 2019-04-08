# Inverse Optimization Methods for Pyomo #

## Description ##

This project implements state-of-the-art inverse optimization (IO) methods found in the literature. 

Note: We reference a "chapter" at times in the files. This refers to the extensive documentation we have written to go along-side the code.  We are not publicly releasing the chapter at this moment, but it is something about which interested parties can contact the researcher.

* In the `GIO_files` folder, we have the `GIO` class which implements methods from Chan et al.'s 2018 paper on generalized inverse optimization.  The class allows users to create a `GIO` object that will hold data and has a series of methods to perform IO functionality found in the Chan et al. paper.  The full citation for the paper can be found here:

	Chan, Timothy CY, Taewoo Lee, and Daria Terekhov. 
	"Inverse Optimization: Closed-Form Solutions, Geometry, 
	and Goodness of Fit." Management Science (2018).

	* The `gio.py` file within the `gio_source` folder contains the GIO class which allows users to create a GIO object that will hold data and has a series of methods to perform IO functionality found in the Chan et al. paper.
	* The `jupyter_notebook_demo` folder contains the `demo_gio_class.ipynb` Jupyter notebook which demos the use of the `GIO` class.
	* The `testing_gio_generating_epsilon_c` folder contains the validation scripts for the GIO model methods within the `GIO` class.
	* The `testing_gio_rho` folder contains the validation scripts for the coefficient of complementarity (rho) methods.
	* **NOTE:** We recently introduced this file structure and, thus, still need to go back and add the import statements for the modulated `GIO` class into the testing scripts and the demo Jupyter notebook (see next section for more information).

* In the `online_IO_files` folder, we have **in progress** code for implementing state-of-the-art online inverse optimization methods.  This folder is changing quite a bit right-now and is **no-where near usage level yet**.  Not bleeding edge, nothing.  However, interested parties can be entertained by my various  (changing) comments in the files.  These are the references for the papers we are implementing:
	
	Dong, Chaosheng, Yiran Chen, and Bo Zeng. "Generalized Inverse Optimization through Online Learning." *Advances in Neural Information Processing Systems.* 2018.

	Bärmann, Andres, Alexander Martin, Sebastian Pokutta, Oskar Schneider. An Online-Learning Approach to Inverse Optimization. *arXiv preprint arXiv:1810.12997.* (2018).

A more preliminary version of this last paper can be cited as follows:

	Bärmann, Andreas, Sebastian Pokutta, and Oskar Schneider. "Emulating the expert: inverse optimization through online learning." *Proceedings of the 34th International Conference on Machine Learning-Volume 70.* JMLR. org, 2017.


## Specifications for Set-up of Code ##

1. Observers should note that they need to input the file path location of this repository on their machine (assuming they have downloaded it) in a `system.path.insert()` statement at the beginning of their documents (see files in `online_IO_files` for examples).  Then, users can import the `Online_IO` class via the statement `from online_IO_files.online_IO_source.online_IO import Online_IO`.  
	* A similar process must be undertaken for the `GIO` class.  Users will again need to place the location of this repository in a `system.path.insert()` statement at the beginning of their documents and, then, users can import the `GIO` class via the statement `from GIO_files.gio_source.gio import GIO`.  See `testing_gio.py` in the `testing_gio_generating_epsilon_c` folder for an example.  We need to add this statement to the other files within `GIO_files`.  

2. To set up an environment to accommodate the code, you can build a `conda` environment using the following series of commands:

```
conda create -n enviro_for_664 python numpy pytest pyomo matplotlib jupyter
```
   
which will prompt the user to also agree to download a whole series of other packages as well.
