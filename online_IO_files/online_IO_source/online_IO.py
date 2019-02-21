#### Class for Online Inverse Optimization Methods ####
##These methods come from Dong et al. 2018 and B\"armann et al. 2018



import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

class online_IO(): #SHOULD WE RENAME THE FILE SLIGHTLY
    
    def __init__(self,initial_model):
        
        self.initial_model = initial_model #load initial pyomo model into attribute
        self.obj_vals = {} #going to initialize as a dictionary to make easier to handle
                            #different types of objective functions
    
    def receive_data(self,p_t,x_t,alg_specification):
        
        print("working on")
        
        
        
