#### decorator_for_online_IO.py #####
# 5/20/2019

# This file contains the decorator function that we are UTILIZING FROM:
# Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html
# Again, we take NO CREDIT for this function; it comes from this source.
# We simply changed the names of the parameters and functions a bit. 
# but were still inspired by the naming conventions that Qtrac used.

# We consulted several resources listed below to understand the code presented
# by Qtrac Ltd. better.  

# We talk about this function a bit in Section 1.5: Code Attribution

# Additional notes:

## Code based on code from Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html

## Helpful references for decorators (to help us understand the func below):
#https://realpython.com/primer-on-python-decorators/
#https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
#https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
#(^have a good example with decorators here)

# Thanks to python-3-patterns-idioms-test for noting to us that 
# the outter function takes the functions as arguments but the inner
# function takes the thing to be decorated; thus this website inspired
# some of the naming conventions below

import pdb #for debugging
import math

def func_that_adds_methods(funcs_to_add): #we are taking in functions as arguments (but we are DECORATING a class)
                                #we assume this is a sequence of functions
    def actual_decorator_for_class_steph(Class_decorating): #maybe because we define "funcs" before online_IO, the decorator
                        #perhaps realizes that Class is the thing we are hitting (yup the third
                        #link python-3-patterns-idioms-test above demoed this); we are actually decorating the class
        
        for method in funcs_to_add: #iterate through all of the functions
            setattr(Class_decorating,method.__name__,method) #since __name__ attribute gives the name of the func
                                                            #^We googled for this
        return Class_decorating #returning the class we just added functions to (realpython ref above)
    return actual_decorator_for_class_steph 




