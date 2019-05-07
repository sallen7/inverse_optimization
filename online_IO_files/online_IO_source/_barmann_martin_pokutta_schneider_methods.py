### Methods for B\"armann, Martin, Pokutta, & Schneider 2018 ###

#NOTE TO SELF: DO NEED TO BAKE IN THE MUTABILITY OF C!! BECAUSE IM SCREWING
#UP THE UPDATE STEP BY MAKING IT THINK THAT IT NEEDS TO GET A CVEC FOR THE P_t PASS IN

import copy
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math

#NOTE: will need separate experiment files for the BMPS experiment!!

def compute_standardized_model(self,dimQ=(0,0),dimc=(0,0),dimA=(0,0),\
                           dimD=(0,0),non_negative=0):
    #Can use stuff from compute_KKT_conditions to implement this (including
    #the mutability stuff that is initialized in the beginning)
    #Would be part of the initialize_method
    
    #Want to store the model in like a "family of opt models" variable
    
    #Will be updated throughout the iterations via the receive_data method
    #using the dictionary correspondence stuff I've already established
    
    #To test this: we can make sure that its solution matches the input model
    #(like when we run a solver against it, will make sure it works)
    
    #And we can do the same update data stuff test as we did before with the portfolio
    #optimization stuff
    
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
    
    if non_negative == 0:
        standard_model.x = pyo.Var(standard_model.xindex) #bounds=self.var_bounds)
    elif non_negative == 1: 
        standard_model.x = pyo.Var(standard_model.xindex,domain=pyo.NonNegativeReals) #,bounds=self.var_bounds)
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
    #pdb.set_trace()
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
    ### We are assuming that F is of a form where we have variables c
    ## NEED TO FIGURE OUT IF y_t, c_t, etc are going to be dictionaries
    ## or numpy arrays or like what the data format is gonna be
    ## I don't know if you can add dictionaries?
    
    ##Gonna use np.fromiter(b.values(),dtype=float,count=len(b))
    #bvec_col = np.reshape(bvec,(len(b),1)) #then this to make a column vec
    
    #ASSUME y_t IS A COLUMN VECTOR - and REMEMBER that it is used for 
    #something else in this algorithm than in the last algorithm
    
    (n,ph) = dimc
    region_F = self.feasible_set_C.clone()
    
    
    ### Adding Objective Function to Region_F ###
    def obj_func_region_F(model):
        return sum((model.c[j] - y_t[j-1,0])**2 for j in range(1,n+1))
    
    region_F.obj_func = pyo.Objective(rule=obj_func_region_F)
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(region_F)
    print("This is the termination condition (project_to_F):",results.solver.termination_condition)
    
    ### Extracting Solution ###
    ct_vals = region_F.c.extract_values() #obtain a dictionary
    
    self.c_t_BMPS = ct_vals ##will need to update subproblem with this 
                            #(hence need to keep in dictionary format)
                            
    self.project_to_F_model = region_F.clone()
    
    
def solve_subproblem(self):
    #Once we implement the compute_standardized_model method, we can easily
    #solve the updated subproblem through this
    #ASSUME EVERYTHING HAS BEEN UPDATED - the objective function and the constraints
    
    subproblem = self.BMPS_subproblem.clone()
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(subproblem)
    print("This is the termination condition (solve_subproblem):",results.solver.termination_condition)
    
    ### Extracting Solution ###
    #Thanks to: https://stackoverflow.com/questions/15579649/python-dict-to-numpy-structured-array
    xbar_t_vals = subproblem.x.extract_values() #obtain a dictionary
    xbar_t_vec = np.fromiter(xbar_t_vals.values(),dtype=float,count=len(xbar_t_vals))
    xbar_t_col_vec = np.reshape(xbar_t_vec,(len(xbar_t_vals),1))
    
    self.xbar_t_BMPS = xbar_t_col_vec
    

def gradient_step(self,eta_t,x_t):
    ### Need to convert the c_t dictionary into a vector
    ct_vals = copy.deepcopy(self.c_t_BMPS)
    ct_vec = np.fromiter(ct_vals.values(),dtype=float,count=len(ct_vals))
    ct_col_vec = np.reshape(ct_vec,(len(ct_vals),1))
    self.c_t_BMPS = copy.deepcopy(ct_col_vec) #replacing with vector
    
    y_t = ct_col_vec - eta_t*(self.x_t_BMPS - self.xbar_t_BMPS) #NEW CODE: 4/24/2019 - flipped the x_t and xbar_t
    
    #pdb.set_trace()
    
    return y_t #going to leave it this way for now because I pass in y_t to
    #the next function
    #IT DOES SEEM SLIGHTLY RANDOM
    #It is kind of nice because its the final method in the "next_iteration" thing
    
    

barmann_martin_pokutta_schneider_funcs = (compute_standardized_model,\
                project_to_F,solve_subproblem,gradient_step)



