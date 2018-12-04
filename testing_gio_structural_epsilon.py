#### Testing the GIO_structural_epsilon_setup(self)
####           & GIO_structural_epsilon_solve(self,p)
#### Solution Method Set up

import pdb #for debugging
import numpy as np
import unittest
from gio import GIO #importing the GIO class for testing

A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

#####ALSO WORKS AS A GOOD CHECK OF RHO_P calculations
testmod = GIO(A,b,x0) #initialize the model
testmod.GIO_structural_epsilon_setup()
testmod.GIO_struc_ep.pprint()
#pdb.set_trace()
testmod.GIO_structural_epsilon_solve(2)
print("This is x0-ep*:",testmod.x0_epsilon_p) #WORKED

#Need to test against all the rest of the stuff (copy and paste
#the testing_gio.py stuff again) - also need to test rho calculations (make
#SURE these numbers match the p numbers - really good way to validate)

#ALSO do positive epsilon validation example - the istar/c_vector 
#validation test (need to show that can do something with this new method)




