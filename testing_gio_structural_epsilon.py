#### Testing the GIO_structural_epsilon_setup(self)
####           & GIO_structural_epsilon_solve(self,p)
#### Solution Method Set up

import pdb #for debugging
import numpy as np
import pyomo.environ as pyo
import unittest
from gio import GIO #importing the GIO class for testing

A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

#####ALSO WORKS AS A GOOD CHECK OF RHO_P calculations
#testmod = GIO(A,b,x0) #initialize the model
#testmod.GIO_structural_epsilon_setup()
#testmod.GIO_struc_ep.pprint()
#pdb.set_trace()
#testmod.GIO_structural_epsilon_solve(2)
#print("This is x0-ep*:",testmod.x0_epsilon_p) #WORKED

#Need to test against all the rest of the stuff (copy and paste
#the testing_gio.py stuff again) - also need to test rho calculations (make
#SURE these numbers match the p numbers - really good way to validate)
#REALLY NEED to check the 

#ALSO do positive epsilon validation example - the right istar/c_vector 
#validation test (need to show that can do something with this new method)

class TestGIO_Structural_Ep(unittest.TestCase):
    "Tests for GIO Structural Epsilon stuff"
    def setUp(self):
        """Generating one (or more) test instance(s) that can be shared among all of the 
        unit tests"""
        ##### Data from Example 1 of Chan et al. Paper #####
        A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
        b = np.array([[10],[-6],[4],[-10]])
        x0 = np.array([[2.5],[3]])
        self.example1Chan = GIO(A,b,x0)
        #self.ex1Chan_testingGIOallmethod = GIO(A,b,x0)
        
    def test_GIO_struc_ep_p_2(self):
        self.example1Chan.GIO_structural_epsilon_setup()
        self.example1Chan.GIO_structural_epsilon_solve(2) #carries out the same storing protocol that
                                #the more traditional stuff does
                                #epsilon* will be stored in self.example1Chan.epsilon_p
                                #x0-epsilon* will be stored in self.example1Chan.x0_epsilon_p
        chan_x0_ep_p_2 = np.array([[2.19],[3.46]]) #from the Chan et al. paper
        self.assertTrue(  np.all(chan_x0_ep_p_2 == np.around(self.example1Chan.x0_epsilon_p[0],decimals=2))  )
        #####ADDING THE CHECK OF c calculation#####
        chan_c_p_2 = np.array([[0.4],[-0.6]])
        self.assertTrue(  np.all(chan_c_p_2 == np.around(self.example1Chan.c_p[0],decimals=2))  )
    
    def test_GIO_struc_ep_p_1(self): #unittest function
        self.example1Chan.GIO_structural_epsilon_setup() #need to re-run this because 
                                                        #we need to reset the model 
        self.example1Chan.GIO_structural_epsilon_solve(1) 
        chan_x0_ep_p_1 = np.array([[2.5],[3.7]]) #rounded 3.666666 to 3.7 since that will be what the np.round function does
        self.assertTrue(  np.all(chan_x0_ep_p_1 == np.around(self.example1Chan.x0_epsilon_p[0],decimals=1))  )
        #####ADDING THE CHECK OF c calculation#####
        chan_c_p_1 = np.array([[0.4],[-0.6]])
        self.assertTrue(  np.all(chan_c_p_1 == np.around(self.example1Chan.c_p[0],decimals=2))  )        
        
    def test_GIO_struc_ep_p_inf(self):
        self.example1Chan.GIO_structural_epsilon_setup() #need to re-run this because 
                                                        #we need to reset the model 
        self.example1Chan.GIO_structural_epsilon_solve('inf') 
        chan_x0_ep_p_inf = np.array([[2.1],[3.4]])
        self.assertTrue(  np.all(chan_x0_ep_p_inf == np.around(self.example1Chan.x0_epsilon_p[0],decimals=1))  )
        #####ADDING THE CHECK OF c calculation#####
        chan_c_p_inf = np.array([[0.4],[-0.6]])
        self.assertTrue(  np.all(chan_c_p_inf == np.around(self.example1Chan.c_p[0],decimals=2))  )
    
    def test_force_ep_pos_p_2(self):
        print("this test will check that we get the c that I'm expecting when I force epsilon to be positive")
        #expecting projection to the one that the relative duality gap model projects to 
        #WOULD NEED FEASIBILITY (WHICH THE METHOD FORCES)
        self.example1Chan.GIO_structural_epsilon_setup()
        def ep_constraint(model):  #should provide the details of the index sets and the numvar parameters
            return model.ep[1] <= model.ep[2] #specifically did not ID the epsilon as nonnegative in gio.py
        self.example1Chan.GIO_struc_ep.constraint_ep = pyo.Constraint(rule=ep_constraint)
        def non_neg_ep(model,i):
            return model.ep[i] <= 0
        self.example1Chan.GIO_struc_ep.non_neg_ep = pyo.Constraint(\
                                                    self.example1Chan.GIO_struc_ep.varindex,rule=non_neg_ep)
        
        self.example1Chan.GIO_structural_epsilon_solve(2) #solve under the 2 norm
        chan_theory_c = np.array([[-0.67],[-0.33]])
        pdb.set_trace()
        self.assertTrue(  np.all(chan_theory_c == np.around(self.example1Chan.c_p[0],decimals=2))  )
        
        ###WILL HAVE TO THINK ABOUT IF NEED THE SAME SET OF CONSTRAINTS FOR THE OTHERS
                
        
unittest.main() #to run the unittest stuff in the file




