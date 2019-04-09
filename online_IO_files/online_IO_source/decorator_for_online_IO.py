### Defining a Decorator Function to be "Pied" before the beginning of
## the online_IO class

## Code based on code from: http://www.qtrac.eu/pyclassmulti.html

## Helpful references for decorators:
#https://realpython.com/primer-on-python-decorators/
#https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
#https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
#(have a good example with decorators here)

#Need to see if we will need to add the **args and **kwargs
#Also might need to pull in stuff from multiple modules

import pdb #for debugging
import math

def func_adds_methods(funcs): #we are taking in functions as arguments (but we are DECORATING a class)
                                #we assume this is a sequence of functions
    def decorator(Class): #maybe because we define "funcs" before online_IO, the decorator
                        #perhaps realizes that Class is the thing we are hitting (yup the third
                        #link above demoed this)
        for func in funcs: #iterate through all of teh functions
            setattr(Class,func.__name__,func) #since __name__ attribute gives the name of the func
        return Class
    return decorator 
