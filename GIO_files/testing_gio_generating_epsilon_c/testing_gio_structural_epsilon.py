#### Testing the GIO_structural_epsilon_setup(self)
####           & GIO_structural_epsilon_solve(self,p)
#We are testing the structural constraints upon epsilon model
#workflow.

#We use the example 1 tests from the testing_gio.py file
#and then we institute constraints upon epsilon for each of the norm
#cases to force the model to choose a specific hyperplane for the 
#c^* calculations

#See the Appendix of the chapter for more details

import pdb #for debugging
import numpy as np
import pyomo.environ as pyo
import unittest
from gio import GIO #importing the GIO class for testing


class TestGIO_Structural_Ep(unittest.TestCase):
    def setUp(self):
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
        self.example1Chan.GIO_structural_epsilon_setup()
        def ep_constraint_p_2(model):  #should provide the details of the index sets and the numvar parameters
            return model.ep[1] <= model.ep[2] #specifically did not ID the epsilon as nonnegative in gio.py
        self.example1Chan.GIO_struc_ep.constraint_ep = pyo.Constraint(rule=ep_constraint_p_2)
        def neg_ep_p_2(model,i):
            return model.ep[i] <= 0
        self.example1Chan.GIO_struc_ep.neg_ep = pyo.Constraint(\
                                                    self.example1Chan.GIO_struc_ep.varindex,rule=neg_ep_p_2)
        
        self.example1Chan.GIO_structural_epsilon_solve(2) #solve under the 2 norm
        chan_theory_c = np.array([[-0.67],[-0.33]])
        #pdb.set_trace()
        self.assertTrue(  np.all(chan_theory_c == np.around(self.example1Chan.c_p[0],decimals=2))  )
        
    def test_force_ep_p_1(self): 
        self.example1Chan.GIO_structural_epsilon_setup()
        
        def ep_constraint_equal_p_1(model):  
            return model.ep[2] == 0 
        self.example1Chan.GIO_struc_ep.constraint_ep = pyo.Constraint(rule=ep_constraint_equal_p_1)
        
        def neg_ep_p_1(model):
            return model.ep[1] <= 0
        self.example1Chan.GIO_struc_ep.neg_ep = pyo.Constraint(rule=neg_ep_p_1)
        
        self.example1Chan.GIO_structural_epsilon_solve(1) #solve under the 2 norm
        chan_theory_c = np.array([[-0.67],[-0.33]])
        #pdb.set_trace()
        self.assertTrue(  np.all(chan_theory_c == np.around(self.example1Chan.c_p[0],decimals=2))  )

    def test_force_ep_p_inf(self): 
        self.example1Chan.GIO_structural_epsilon_setup()
        
        def ep_constraint_equal_p_inf(model):  
            return model.ep[1] == 0 
        self.example1Chan.GIO_struc_ep.constraint_ep = pyo.Constraint(rule=ep_constraint_equal_p_inf)
        
        def non_neg_constraint_p_inf(model):
            return model.ep[2] >= 0
        self.example1Chan.GIO_struc_ep.non_neg_constraint_p_inf = pyo.Constraint(rule=non_neg_constraint_p_inf)
        
        self.example1Chan.GIO_structural_epsilon_solve('inf') #solve under the 2 norm
        chan_theory_c = np.array([[0.29],[0.71]])
        #pdb.set_trace()
        self.assertTrue(  np.all(chan_theory_c == np.around(self.example1Chan.c_p[0],decimals=2))  )

                
        
unittest.main() #to run the unittest stuff in the file




