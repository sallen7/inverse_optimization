## _barmann_martin_pokutta_schneider_methods.py ##
## 5/20/2019

# This file contains the methods/functions for the 
# B\"armann, Martin, Pokutta, & Schneider (2018) online gradient descent 
# online inverse optimization
# algorithm; these are algorithm specific methods. Users/readers can refer to 
# Section 1.3.3: Methods to Implement Online Gradient Descent in Online_IO
# in the Chapter documentation to learn more about these methods.
# The mathematical background/implementation can be found in 
# Sections 1.3.1-1.3.2

# Additional Notes:
# Big thanks to Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html for explaining
# how to break up methods into multiple files for a class.

# BMPS = B\"armann, Martin, Pokutta, & Schneider 

import copy
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math


def compute_standardized_model(self,dimQ=(0,0),dimc=(0,0),dimA=(0,0),\
                           dimD=(0,0),non_negative=0):
    
    #METHOD DESCRIPTION: This method creates the standardized model for 
    #the BMPS algorithm.  The compute_KKT_conditions code was very helpful
    #in generating this code
    
    ######################################
    #### Step 0: Setting up the Variables and the Parameters ####

    standard_model = pyo.ConcreteModel()
    
    (n,ph) = dimc #assume c is column vector
    (m,ph) = dimA #m is the number of inequality constraints
    (p,ph) = dimD #p is the number of equality constraints
    (n2,ph) = dimQ 
    
    standard_model.xindex = pyo.RangeSet(1,n)
    standard_model.uindex = pyo.RangeSet(1,m) #number of inequality constraints
    standard_model.vindex = pyo.RangeSet(1,p) #number of equality constraints
    
    
    ## If a user did pass in var_bounds for the BMPS Method ##
    if self.var_bounds is not None: #if something got passed to it
        assert isinstance(self.var_bounds,tuple), "Error: If you specify var_bounds for the BMPS_online_GD method, must be a tuple"
        assert len(self.var_bounds) == 2, "Error: The tuple for var_bounds must be of length 2"
    
    if non_negative == 0:
        standard_model.x = pyo.Var(standard_model.xindex,bounds=self.var_bounds)
    elif non_negative == 1: 
        standard_model.x = pyo.Var(standard_model.xindex,domain=pyo.NonNegativeReals,bounds=self.var_bounds)
    else:
        print("Error: Can only have 0 or 1 for the non-negative parameter")
    
    ## Setting up Parameter Data (and make the things mutable that need to be) ##
    
    for (i,param_name) in self.num2param_key.items():
        if self.model_data_names[param_name] == 'None':
            pass #continue onward
        else:
            data = getattr(self.initial_model,self.model_data_names[param_name])
            data = data.extract_values()
            if param_name == 'Q': 
                setattr(standard_model,param_name,pyo.Param(standard_model.xindex,standard_model.xindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'c': #FOR THIS MODEL: c is NOT a variable
                setattr(standard_model,param_name,pyo.Param(standard_model.xindex,\
                            initialize=data,mutable=True)) #NEW 4/24/2019 - need to be able to update
                                    #the c - you CANNOT reconstruct c if it is NOT mutable
            elif param_name == 'A':
                setattr(standard_model,param_name,pyo.Param(standard_model.uindex,\
                standard_model.xindex,initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'b':
                setattr(standard_model,param_name,pyo.Param(standard_model.uindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'D':
                setattr(standard_model,param_name,pyo.Param(standard_model.vindex,standard_model.xindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'f':
                setattr(standard_model,param_name,pyo.Param(standard_model.vindex,\
                            initialize=data,mutable=self.if_mutable[i]))
    
    #### Step 1: Set up the Constraints ####
    # We just need to follow the standard model: We are just transcribing
    # what the user inputted into a standard form

    if m > 0: #then there are inequality constraints
        def inequality_constraints_rule(model,i):
            return sum(model.A[i,j]*model.x[j] for j in range(1,n+1)) <= model.b[i]
        
        standard_model.inequality_constraints = pyo.Constraint(standard_model.uindex,\
                                                    rule=inequality_constraints_rule)
        
    if p > 0:
        def equality_constraints_rule(model,i):
            return sum(model.D[i,j]*model.x[j] for j in range(1,n+1)) == model.f[i]
        
        standard_model.equality_constraints = pyo.Constraint(standard_model.vindex,\
                                                    rule=equality_constraints_rule)
    
    #### Step 2: Set up the Objective Function ####
    
    if n2 > 0: #if there is indeed a Q (we ALWAYS assume there is a c)
        def obj_func_with_Q(model):
            xt_Q_x_term = sum(sum(model.Q[i,j]*model.x[i]*model.x[j] for i in range(1,n2+1)) for j in range(1,n2+1))
            return (0.5)*xt_Q_x_term + sum(model.c[j]*model.x[j] for j in range(1,n+1))
        
        standard_model.obj_func = pyo.Objective(rule=obj_func_with_Q,sense=pyo.minimize)
        
    elif n2 == 0: #if there is NO Q
        def obj_func_without_Q(model):
            return sum(model.c[j]*model.x[j] for j in range(1,n+1))
        
        standard_model.obj_func = pyo.Objective(rule=obj_func_without_Q,sense=pyo.minimize)
    
    else:
        print("Incorrect value for dim of Q.  Somehow you put in a negative value...")
        return        
    
    self.BMPS_subproblem = standard_model #BMPS call this model with the changing c 
                                        #"subproblem (2)" so we have named it "BMPS_subproblem"
        

def project_to_F(self,dimc=(0,0),y_t=None):
    #METHOD DESCRIPTION: This method carries out the c_t update with
    #c_t = argmin{||c - z_{t-1} ||_2^2 : c \in C}
    #Remember F == C and y_t == z_t 
    #This is Step 2 of Online Gradient Descent.
    
    #Assume y_t is a column vector 
    
    ########################################################
    
    (n,ph) = dimc
    region_F = self.feasible_set_C.clone()
    
    
    ### Adding Objective Function to Region_F ###
    def obj_func_region_F(model):
        return sum((model.c[j] - y_t[j-1,0])**2 for j in range(1,n+1))
    
    region_F.obj_func = pyo.Objective(rule=obj_func_region_F)
    
    ### Solving ###
    solver = SolverFactory("gurobi") 
    
    results = solver.solve(region_F)
    print("This is the termination condition (project_to_F):",results.solver.termination_condition)
    
    ### Extracting Solution ###
    ct_vals = region_F.c.extract_values() #obtain a dictionary
    
    self.c_t_BMPS = ct_vals ##will need to update subproblem with this 
                            #(hence need to keep in dictionary format)
                            
    self.project_to_F_model = region_F.clone()
    
    
def solve_subproblem(self):
    #METHOD DESCRIPTION: This method solves the min c_t^T x st x \in X(p_t)
    #problem, also known as BMPS_subproblem for the current c_t and p_t
    #that the user has inputted into it
    
    ##########################################
    
    subproblem = self.BMPS_subproblem.clone()
    
    ### Solving ###
    solver = SolverFactory("gurobi") 
    
    results = solver.solve(subproblem)
    print("This is the termination condition (solve_subproblem):",results.solver.termination_condition)
    
    ### Extracting Solution ###
    #Thanks to: https://stackoverflow.com/questions/15579649/python-dict-to-numpy-structured-array
    xbar_t_vals = subproblem.x.extract_values() #obtain a dictionary
    xbar_t_vec = np.fromiter(xbar_t_vals.values(),dtype=float,count=len(xbar_t_vals))
    xbar_t_col_vec = np.reshape(xbar_t_vec,(len(xbar_t_vals),1))
    
    self.xbar_t_BMPS = xbar_t_col_vec
    

def gradient_step(self,eta_t,x_t):
    #METHOD DESCRIPTION: This method carries out Step 1 of Online Gradient Descent
    #by computing the intermediate value z_{t+1} = c_t - eta_t (x_t - xbar_t)
    #Remember: y_t == z_t
    
    ##################################
    
    ### Need to convert the c_t dictionary into a vector
    ct_vals = copy.deepcopy(self.c_t_BMPS)
    ct_vec = np.fromiter(ct_vals.values(),dtype=float,count=len(ct_vals))
    ct_col_vec = np.reshape(ct_vec,(len(ct_vals),1))
    self.c_t_BMPS = copy.deepcopy(ct_col_vec) #replacing with vector
    
    y_t = ct_col_vec - eta_t*(self.x_t_BMPS - self.xbar_t_BMPS) #NEW CODE: 4/24/2019 - flipped the x_t and xbar_t
    
    return y_t #going to leave it this way for now because I pass in y_t to
    #the next function
    
    
# From Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html
# As the website instructs, we put the methods in this tuple form to 
# be able to pass them into the decorator function. 
# We are just following what the website demonstrates.
    
barmann_martin_pokutta_schneider_funcs = (compute_standardized_model,\
                project_to_F,solve_subproblem,gradient_step)



