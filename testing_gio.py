#### Testing GIO class ####
#This script tests the GIO_p method from the GIO class against Example 1
#from the Chan et al. paper.  The example provided optimal x0-epsilon* values
#under the 1,2, and infinity norms, and we use the unit tests to ensure that 
#we achieve these values

import numpy as np
import unittest
from gio import GIO #importing the GIO class for testing

#####Going to Run the test less sophisticated out here too
#A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
#b = np.array([[10],[-6],[4],[-10]])
#x0 = np.array([[2.5],[3]])
#
#gio_testing = GIO(A,b,x0)
#
#gio_testing.GIO_p(1,'T')
#
#print(gio_testing.x0_epsilon_p)

#gio_testing.GIO_relative_duality()
#
#print("GIO relative duality x0-ep_r:",gio_testing.x0_epsilon_r)

#gio_testing.GIO_p('inf') #IS SOMETHING GETTING STORED WEIRD?
#
#gio_testing.GIO_abs_duality() #doesn't take any arguments
#
#print("******************************")
#print("x0-ep for abs dual:",gio_testing.x0_epsilon_a) #not outputting correctly
#print("epsilon a:",gio_testing.epsilon_a) #not outputting correctly
#print("x0-ep for p=inf:",gio_testing.x0_epsilon_p)
#print("epsilon p:",gio_testing.epsilon_p)

#print(gio_testing.A)
#gio_testing.GIO_p(1)
#print(gio_testing.x0_epsilon_p)
#gio_testing.GIO_p('inf')
#print(gio_testing.x0_epsilon_p)
#gio_testing.GIO_p(2)
#print(gio_testing.x0_epsilon_p)

#########   Unit Test References   ######################
#Inspired by: https://stackoverflow.com/questions/3302949/whats-the-best-way-to-assert-for-numpy-array-equality
        #and https://stackoverflow.com/questions/10062954/valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous
        
        #For numpy in the future, might use this: https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.testing.html
        #https://docs.scipy.org/doc/numpy-1.15.1/reference/testing.html (numpy has its own assert methods, def useful for
        #almost equal cases)


#####   The Unit Test   #######
class TestGIO(unittest.TestCase):
    "Tests for GIO class"
    def setUp(self):
        """Generating one (or more) test instance(s) that can be shared among all of the 
        unit tests"""
        ##### Data from Example 1 of Chan et al. Paper #####
        A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
        b = np.array([[10],[-6],[4],[-10]])
        x0 = np.array([[2.5],[3]])
        self.example1Chan = GIO(A,b,x0)
        self.ex1Chan_testingGIOallmethod = GIO(A,b,x0)
    
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
        
        
        
        
        
        
        