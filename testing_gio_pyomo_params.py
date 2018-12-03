#### Testing GIO class ####
#This script tests the GIO_p method from the GIO class against Example 1
#from the Chan et al. paper.  The example provided optimal x0-epsilon* values
#under the 1,2, and infinity norms, and we use the unit tests to ensure that 
#we achieve these values

import numpy as np
import unittest
from gio import GIO #importing the GIO class for testing
import pyomo.environ as pyo

#########   Unit Test References   ######################

#####   The Unit Test   #######
class TestGIO_(unittest.TestCase):
    "Tests for GIO class when pass Pyomo Parameters"
    def setUp(self):
        """Generating one (or more) test instance(s) that can be shared among all of the 
        unit tests"""
        ##### Creating a Pyomo Model from the data in Chan et al. Example 1 #####
        A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
        #b = np.array([[10],[-6],[4],[-10]])
        x0 = np.array([[2.5],[3]])
        
        test_model = pyo.ConcreteModel()
        test_model.varindex = pyo.RangeSet(2)
        test_model.numvars = pyo.Param(initialize=2)
        test_model.eqindex = pyo.RangeSet(4)
        test_model.numeqs = pyo.Param(initialize=4)
        test_model.x = pyo.Var(test_model.varindex)
        
        ##### Importing the A and b as param objects #####
        def A_mat_func(model,i,j):
            return A[i-1,j-1]

        test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
        test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:10,2:-6,3:4,4:-10})    
        
        ##### Creating the GIO Objects #####
        self.example1Chan = GIO(test_model.Amat,test_model.bvec,x0,\
                                4,2,'T')
        self.ex1Chan_testingGIOallmethod = GIO(test_model.Amat,test_model.bvec,x0,\
                                               4,2,'T')
    
    def test_GIO_p_2(self):
        self.example1Chan.GIO_p(2,'F')  #the methods are attached to the instances
                                #epsilon* will be stored in self.example1Chan.epsilon_p
                                #x0-epsilon* will be stored in self.example1Chan.x0_epsilon_p
        chan_x0_ep_p_2 = np.array([[2.19],[3.46]]) #from the Chan et al. paper
        self.assertTrue(  np.all(chan_x0_ep_p_2 == np.around(self.example1Chan.x0_epsilon_p[0],decimals=2))  ) 
    
    def test_GIO_p_1(self): #unittest function
        self.example1Chan.GIO_p(1,'F')
        chan_x0_ep_p_1 = np.array([[2.5],[3.7]]) #rounded 3.666666 to 3.7 since that will be what the np.round function does
        self.assertTrue(  np.all(chan_x0_ep_p_1 == np.around(self.example1Chan.x0_epsilon_p[0],decimals=1))  )
        
    def test_GIO_p_inf(self):
        self.example1Chan.GIO_p('inf','F')
        chan_x0_ep_p_inf = np.array([[2.1],[3.4]])
        self.assertTrue(  np.all(chan_x0_ep_p_inf == np.around(self.example1Chan.x0_epsilon_p[0],decimals=1))  )
        
    def test_GIO_abs_duality(self):
        self.example1Chan.GIO_abs_duality()
        chan_x0_ep_a = np.array([[2.1],[3.4]])
        self.assertTrue(  np.all(chan_x0_ep_a == np.around(self.example1Chan.x0_epsilon_a[0],decimals=1))  )
        
    def test_GIO_relative_duality(self):
        self.example1Chan.GIO_relative_duality()
        chan_x0_ep_r = np.array([[3.17],[3.67]])
        self.assertTrue(  np.all(chan_x0_ep_r == np.around(self.example1Chan.x0_epsilon_r[0],decimals=2))  )
    
    def test_GIO_all_measures(self):
        self.ex1Chan_testingGIOallmethod.GIO_all_measures() #initiating all of the measures
        
        chan_x0_ep_p_1 = np.array([[2.5],[3.7]]) #rounded 3.666666 to 3.7 since that will be what the np.round function does
        self.assertTrue(  np.all(chan_x0_ep_p_1 == np.around(self.ex1Chan_testingGIOallmethod.x0_epsilon_p[0],decimals=1))  )
        
        chan_x0_ep_p_2 = np.array([[2.19],[3.46]]) #from the Chan et al. paper
        self.assertTrue(  np.all(chan_x0_ep_p_2 == np.around(self.ex1Chan_testingGIOallmethod.x0_epsilon_p[1],decimals=2))  )
        
        chan_x0_ep_p_inf = np.array([[2.1],[3.4]])
        self.assertTrue(  np.all(chan_x0_ep_p_inf == np.around(self.ex1Chan_testingGIOallmethod.x0_epsilon_p[2],decimals=1))  )
        
        chan_x0_ep_a = np.array([[2.1],[3.4]])
        self.assertTrue(  np.all(chan_x0_ep_a == np.around(self.ex1Chan_testingGIOallmethod.x0_epsilon_a[0],decimals=1))  )
        
        chan_x0_ep_r = np.array([[3.17],[3.67]])
        self.assertTrue(  np.all(chan_x0_ep_r == np.around(self.ex1Chan_testingGIOallmethod.x0_epsilon_r[0],decimals=2))  )
        

unittest.main() #to run the unittest stuff in the file

