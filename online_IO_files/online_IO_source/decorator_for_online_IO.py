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

#I predict that we will be able to experiment with these messages

import pdb #for debugging
import math

def func_adds_methods(funcs): #we are taking in functions as arguments (but we are DECORATING a class)
                                #we assume this is a sequence of functions
    def decorator_steph(Class_decorating): #maybe because we define "funcs" before online_IO, the decorator
                        #perhaps realizes that Class is the thing we are hitting (yup the third
                        #link above demoed this)
                        #We are actually decorating the class
        for func in funcs: #iterate through all of teh functions
            setattr(Class_decorating,func.__name__,func) #since __name__ attribute gives the name of the func
        #print("inside the decorator func, before return the class with added attributes")
        return Class_decorating #decorator_steph returning the class we just added functions to (realpython ref above)
    #print("outside the decorator func, before return the decorator func (which replaces Online_IO")
    #return Class_decorating
    return decorator_steph #from realpython looks like 
                #Online_IO will "point" to the "decorator"
            #From python-3-patterns and realpython
            #Actually, func_adds_methods is returning the 
            #function decorator_steph
            #I think it actually "replaces" the Online_IO class
            #with the decorator_steph
            #Which then, when Online_IO gets invoked, decorator_steph
            #gets called, and all the funcs get added and the 
            #class with the additional functions gets returned
