#### Class for Online Inverse Optimization Methods ####
##These methods come from Dong et al. 2018 and B\"armann et al. 2018



import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity

class online_IO(): #SHOULD WE RENAME THE FILE SLIGHTLY
    
    def __init__(self,initial_model):
        
        self.initial_model = initial_model #load initial pyomo model into attribute
        self.obj_vals = {} #going to initialize as a dictionary to make easier to handle
                            #different types of objective functions
    
    def receive_data(self,p_t,x_t,alg_specification):
        
        print("working on")
        
        
    def compute_KKT_conditions(self,Q,c,A,b,D,f,dimQ=(1,1),dimc=(1,1),dimA=(1,1),dimD=(1,1),use_numpy=0):
        ##We are going to assume that the Pyomo model is of the following form:
        ## min f(x) = (1/2) x^t Q x + c^T x
        ## st Ax <= b
        ## Dx = f
        
        # We also assume that each of the necessary parameters for the KKT conditions
        # are in pyomo parameter objects (and we assume that A and D have not been transposed
        # already)
        
        #We also need users to provide if they are using numpy matrices or Pyomo parameter
        #values.  If use_numpy=0, then this means that we are using pyomo parameter
        #objects (you can indeed specifically pass in parameter objects into a variable!). 
        #With use_numpy=0, this also means that the user will need to provide the 
        #dimensions of the parameter values (as two dimensional tuples)
        
        #If use_numpy=1, then we know that Q,c,A,D are numpy arrays.
        KKT_model = pyo.ConcreteModel()
        
        if use_numpy == 0: #means we are using parameter objects
            ##### Step 1: Write out the Stationarity Conditions #####
            
            ## Setting up the Variables ##
            (m,n) = dimA
            (p,n) = dimD #some notation use from Dr. Gabriel's ENME741 notes
            
            KKT_model.xindex = pyo.RangeSet(1,n)
            KKT_model.x = pyo.Var(KKT_model.xindex)
            KKT_model.uindex = pyo.RangeSet(1,m)
            KKT_model.u = pyo.Var(KKT_model.uindex)
            KKT_model.vindex = pyo.RangeSet(1,p)
            KKT_model.v = pyo.Var(KKT_model.vindex)
            
            ## Establishing the KKT Stationarity Stuff ##
            def KKT_stationarity_rule(model,i):
                return sum(Q[i,j]*model.x[j] for j in range(1,n+1)) + c[i]\
                    + sum(A[j,i]*model.u[j] for j in range(1,m+1)) \
                    + sum(D[j,i]*model.v[j] for j in range(1,p+1)) == 0
            
            KKT_model.stationary_conditions = pyo.Constraint(KKT_model.xindex,rule=KKT_stationarity_rule)
            
            
            ##### Step 2: Complementarity Constraints #####
            def complementarity_constraints(model,i):
                placeholder_1 = sum(A[i,j]*model.x[j] for j in range(1,))
                return pyompec.complements()
            ###NEED TO DO THE TWO TRANSFORMATIONS THAT WE LEARNED ABOUT IN THE
            ####725 PROJECT
            
            ##ALSO NOTE THAT A is not transposed in this context now!!
            
            
            ##### Step 3: Equality Constraints #####
            def equality_constraints(model,i):
                return sum(D[i,j]*model.x[j] for j in range(1,n+1)) == f[i]
            
            KKT_model.equality_conditions = pyo.Constraint(KKT_model.xindex,rule=equality_constraints)
            
            
            
            
            
                        
    
        elif use_numpy == 1: #means we are using numpy objects
            
            
        else:
            print("Error: This argument can only be 0 or 1")
            return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
