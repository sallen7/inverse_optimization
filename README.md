# Inverse Optimization Methods for Pyomo #

## Description ##

This project implements state-of-the-art inverse optimization (IO) methods found in the literature. 

**Note:** We reference a "chapter" at times in the files. This refers to the extensive documentation we have written to go along-side the code.  We are not publicly releasing the chapter at this moment, but it is something about which interested parties can contact the researcher.  This chapter provides the citations for the project.

#### AMSC 663 ####

* In the `GIO_files` folder, we have the `GIO` class which implements methods from Chan et al.'s 2018 paper on generalized inverse optimization.  The class allows users to create a `GIO` object that will hold data and has a series of methods to perform IO functionality found in the Chan et al. paper.  The full citation for the paper can be found here:

	Chan, Timothy CY, Taewoo Lee, and Daria Terekhov. 
	"Inverse Optimization: Closed-Form Solutions, Geometry, 
	and Goodness of Fit." Management Science (2018).

* This is the file structure of the `GIO_files` folder:

	* The `gio.py` file within the `gio_source` folder contains the GIO class which allows users to create a GIO object that will hold data and has a series of methods to perform IO functionality found in the Chan et al. paper.
	* The `jupyter_notebook_demo` folder contains the `demo_gio_class.ipynb` Jupyter notebook which demos the use of the `GIO` class.
	* The `testing_gio_generating_epsilon_c` folder contains the validation scripts for the GIO model methods within the `GIO` class.
	* The `testing_gio_rho` folder contains the validation scripts for the coefficient of complementarity (rho) methods.
	* **NOTE:** We recently introduced this file structure and, thus, still need to go back and add the import statements for the modulated `GIO` class into the testing scripts and the demo Jupyter notebook (see next section for more information).

#### AMSC 664 ####

* In the `online_IO_files` folder, we have code for implementing state-of-the-art online inverse optimization methods.  We create a class to implement the methods entitled `Online_IO`.  These are the references for the papers we are implementing:
	
	Dong, Chaosheng, Yiran Chen, and Bo Zeng. "Generalized Inverse Optimization through Online Learning." *Advances in Neural Information Processing Systems.* 2018.

	Bärmann, Andres, Alexander Martin, Sebastian Pokutta, Oskar Schneider. An Online-Learning Approach to Inverse Optimization. *arXiv preprint arXiv:1810.12997.* (2018).

* There are several sub-folders within `online_IO_files`, so we will describe the contents of each:
	* `online_IO_files/experiments`: This folder has the .py files for the computational experiments drawn from the papers. There are some .pickle data structures and image files that are products of the experiments.
	* `online_IO_files/jupyter_notebook_demo`: This folder has 2 Jupyter notebooks that demonstrate the workflow of the `Online_IO` class; these demonstrating workflow notebooks contain the code from the ``running experiment'' files found in the previous folder.  There are .pickle data files associated with these notebooks in the folder as well.  There is a third notebook that has some `pyomo` templates to aid users in defining the necessary `pyomo` models for the `Online_IO` class.
	* `online_IO_files/online_IO_source`: This folder contains all of the code for the `Online_IO` class.
	* `online_IO_files/testing_toy_problem`: This folder has all of the unit test files for the `Online_IO` class.  This includes a MATLAB file that was useful in helping us generate data for the experiments.


**IMPORTANT:** For the Jupyter notebooks in this repository, we recommend copying and pasting their links into the following Jupyter notebook viewer: <https://nbviewer.jupyter.org/>  GitHub does not always display all of the elements in them, and we trust the nbviewer to provide a better experience for users.

## AMSC664 Pytest Results ##

These are the results of our unit tests using the `pytest` unit testing package:

>==================== test session starts =====================

>platform win32 -- Python 3.7.3, pytest-4.3.1, py-1.8.0, pluggy-0.9.0
>rootdir: C:\Users\StephanieAllen\Documents\1_AMSC663\Repository_for_Code\online_IO_files\testing_toy_problems, inifile:

>collected 22 items

>test_BMPS_code.py .........                             [ 40%]

>test_DCZ_code.py .......                                [ 72%]

>test_mechanics_methods.py ......                        [100%]

>================= 22 passed in 4.79 seconds ==================


## Specifications for Set-up of Code ##

1. Observers should note that they need to input the file path location of this repository on their machine (assuming a potential user has downloaded the repository) in a `system.path.insert()` statement at the beginning of their documents (see files in `online_IO_files` for examples).  Then, users can import the `Online_IO` class via the statement `from online_IO_files.online_IO_source.online_IO import Online_IO`.  
	* A similar process must be undertaken for the `GIO` class.  Users will again need to place the location of this repository in a `system.path.insert()` statement at the beginning of their documents and, then, users can import the `GIO` class via the statement `from GIO_files.gio_source.gio import GIO`.  See `testing_gio.py` in the `testing_gio_generating_epsilon_c` folder for an example.  We still need to add this insert statement to the other files within `GIO_files`.  

2. To set up an environment to accommodate the code, you can build a `conda` environment using the following series of commands:

	```
	conda create -n enviro_for_664 python numpy pytest pyomo matplotlib jupyter
	``` 
which will prompt the user to also agree to download a whole series of other packages as well.

3. As discussed in the Pyomo Documentation (and in the Pyomo book), optimization solvers are not installed automatically when a user installs `pyomo`.  For this code, a user will need to install `gurobi`, `ipopt`, and `glpk` and then, for Windows, specify these solvers in his/her/their Path environment variables.  We suggest running a quick google search if a user is unfamiliar with how to do this.  That's how we learned how to do this.

	Pyomo Documentation: <https://buildmedia.readthedocs.org/media/pdf/pyomo/stable/pyomo.pdf>

	Pyomo Book: Hart, William E., Carl D. Laird, Jean-Paul Watson, David L. Woodruff, Gabriel A. Hackebeil, Bethany L. Nicholson, and John D. Siirola. *Pyomo – Optimization Modeling in Python.* Second Edition.  Vol. 67. Springer, 2017.