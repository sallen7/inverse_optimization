### _dong_chen_zeng_methods.py: Methods for Dong, Chen, & Zeng 2018 ###
# 5/20/2019

# This file contains the methods/functions for the
# Dong, Chen, & Zeng (2018) implicit update rule online inverse 
# optimization algorithm; these are algorithm specific methods
# Users/readers can refer to 
# Section 1.4.3 Methods to Implement Implicit Update Rule in Online_IO
# in the Chapter documentation for more details on these specific methods.
# The mathematical background/implementation details can be found in
# Sections 1.4.1-1.4.2

# Additional Notes:
# Big thanks to Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html for explaining
# how to break up methods into multiple files for a class.

# DCZ = Dong, Chen, Zeng

import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math

#Thanks to the "Managing Expressions" Page of Pyomo Documentation
#for informing me about the ability to turn pyomo
#expressions into strings
#https://pyomo.readthedocs.io/en/latest/developer_reference/expressions/managing.html
from pyomo.core.expr import current as EXPR 


def compute_KKT_conditions(self,dimQ=(0,0),dimc=(0,0),dimA=(0,0),\
                           dimD=(0,0),non_negative=0,bigM=10000):
    
    #METHOD DESCRIPTION: This method carries out forming the KKT_conditions_model
    #pyomo model which contains the KKT conditions for the inputted original model.
       
    ##We are going to assume that the Pyomo model is of the following form:
    ## min f(x) = (1/2) x^t Q x + c^T x
    ## st Ax <= b
    ## Dx = f
    #We assume c,b,f are vectors and Q,A,D are matrices
    
    # We also assume that each of the necessary parameters for the KKT conditions
    # are in pyomo parameter objects (and we assume that A and D have not been transposed
    # already)
    
    #############################################
    KKT_model = pyo.ConcreteModel()
    
    ###### Step 0: Setting Up the Book-Keeping #####
    
    ## Setting up the Variables ##
    #ph is for placeholder (wont end up being used)

    (n,ph) = dimc #assume c is column vector
    (m,ph) = dimA 
    (p,ph) = dimD #some notation use from Dr. Gabriel's ENME741 notes
    (n2,ph) = dimQ #n2 for second parameter definition
    
    KKT_model.xindex = pyo.RangeSet(1,n)
    KKT_model.x = pyo.Var(KKT_model.xindex)
    KKT_model.uindex = pyo.RangeSet(1,m)
    KKT_model.u = pyo.Var(KKT_model.uindex,domain=pyo.NonNegativeReals) #just for u
    KKT_model.vindex = pyo.RangeSet(1,p)
    KKT_model.v = pyo.Var(KKT_model.vindex)
    
    ## Setting Data ##
    #USE SET ATTRIBUTE HERE TO ITERATE OVER EVERYTHING!!
    #https://github.com/Pyomo/pyomo/issues/525
    #Then go back in and fix the A[] to KKT_model.A[]
    for (i,param_name) in self.num2param_key.items():
    #I think you can index into things with generators - dictionaries are generators!
        if self.model_data_names[param_name] == 'None':
            pass #continue onward
        else:
            data = getattr(self.initial_model,self.model_data_names[param_name])
            data = data.extract_values()
            if param_name == 'Q': 
                setattr(KKT_model,param_name,pyo.Param(KKT_model.xindex,KKT_model.xindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'c': #THINK I NEED TO ACTUALLY MAKE THESE VARIABLES (need to figure out
                                    #if need to change the mutability thing around)
                setattr(KKT_model,param_name,pyo.Var(KKT_model.xindex))
                getattr(KKT_model,param_name).fix(0) #.fix method can take in a value and
                                #set all the variables to that value
                KKT_model.c.set_values(data) #NEW CHANGE: we are setting the values FOR NOW to the c_data we took in
                
            elif param_name == 'A':
                #pdb.set_trace()
                setattr(KKT_model,param_name,pyo.Param(KKT_model.uindex,\
                KKT_model.xindex,initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'b':
                setattr(KKT_model,param_name,pyo.Param(KKT_model.uindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'D':
                setattr(KKT_model,param_name,pyo.Param(KKT_model.vindex,KKT_model.xindex,\
                            initialize=data,mutable=self.if_mutable[i]))
            elif param_name == 'f':
                setattr(KKT_model,param_name,pyo.Param(KKT_model.vindex,\
                            initialize=data,mutable=self.if_mutable[i]))
                
    ##### Step 1: Write out the Stationarity Conditions #####
    
    ## Establishing the KKT Stationarity Stuff ##
    #Determining the Stationarity Rule to Use Based upon Existance of Data
    #We will assume that c vector is always present (for now) but all
    #others will be up for grabs
    
    ### Version A: x Decision Variables are Free ###
    if non_negative == 0:
        if n2==0 and m>0 and p>0: #A mat, D mat, no Q
            def KKT_stationarity_rule(model,i): #no Q
                return model.c[i]\
                + sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) \
                + sum(model.D[j,i]*model.v[j] for j in range(1,p+1)) == 0
                
        elif n2==0 and m>0 and p==0: #no Q, no D
            def KKT_stationarity_rule(model,i):
                return model.c[i] + sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) == 0
            
        elif n2>0 and m>0 and p>0: #have all the data
            def KKT_stationarity_rule(model,i):
                return sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) + model.c[i]\
                + sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) \
                + sum(model.D[j,i]*model.v[j] for j in range(1,p+1)) == 0
                
        elif n2==0 and m==0 and p>0: #no Q no A
            def KKT_stationarity_rule(model,i):
                return model.c[i] + sum(model.D[j,i]*model.v[j] for j in range(1,p+1)) == 0
            
        elif n2>0 and m==0 and p>0: #Q,D,c no A
            def KKT_stationarity_rule(model,i):
                return model.c[i] + sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) +\
                sum(model.D[j,i]*model.v[j] for j in range(1,p+1)) == 0
        
        elif n2>0 and m>0 and p==0: #Q,c,A no D
            def KKT_stationarity_rule(model,i):
                return model.c[i] + sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) +\
                sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) == 0
        
        KKT_model.stationary_conditions = pyo.Constraint(KKT_model.xindex,rule=KKT_stationarity_rule)
    
    ### Version B: x Decision Variables are Non-Negative ###
    elif non_negative == 1:
        ### Define the Expression for Stationarity Condition ###
        if n2==0 and m>0 and p>0:
            def KKT_stationarity_expr_rule(model,i): #no Q
                return model.c[i]\
                + sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) \
                + sum(model.D[j,i]*model.v[j] for j in range(1,p+1))
                
        elif n2==0 and m>0 and p==0: #no Q, no D
            def KKT_stationarity_expr_rule(model,i):
                return model.c[i] + sum(model.A[j,i]*model.u[j] for j in range(1,m+1))
            
        elif n2>0 and m>0 and p>0: #have all the data
            def KKT_stationarity_expr_rule(model,i):
                return sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) + model.c[i]\
                + sum(model.A[j,i]*model.u[j] for j in range(1,m+1)) \
                + sum(model.D[j,i]*model.v[j] for j in range(1,p+1))
                
        elif n2==0 and m==0 and p>0: #no Q no A
            def KKT_stationarity_expr_rule(model,i):
                return model.c[i] + sum(model.D[j,i]*model.v[j] for j in range(1,p+1))
        
        elif n2>0 and m==0 and p>0: #Q,D,c no A
            def KKT_stationarity_expr_rule(model,i):
                return model.c[i] + sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) +\
                sum(model.D[j,i]*model.v[j] for j in range(1,p+1))
        
        elif n2>0 and m>0 and p==0: #Q,c,A no D
            def KKT_stationarity_expr_rule(model,i):
                return model.c[i] + sum(model.Q[i,j]*model.x[j] for j in range(1,n+1)) +\
                sum(model.A[j,i]*model.u[j] for j in range(1,m+1))
        
            
        KKT_model.stationary_cond_expr = pyo.Expression(KKT_model.xindex,rule=KKT_stationarity_expr_rule)
        
        ##### Define the Disjunctive Constraints for Stationarity Condition #####
        ## Two sets: One set for the Stationary Condition and One for the x ##
        KKT_model.z_non_neg = pyo.Var(KKT_model.xindex,\
                            domain=pyo.Binary) #same number of z vars as u vars
        
        ## Stationary Conditions ##
        def stationary_disjunctive_1(model,i):
            return model.stationary_cond_expr[i] <= bigM*(1-model.z_non_neg[i])
        
        KKT_model.stationary_disjunc_1 = pyo.Constraint(KKT_model.xindex,\
                                                rule=stationary_disjunctive_1)
        def stationary_disjunctive_2(model,i):
            return 0 <= model.stationary_cond_expr[i]
        
        KKT_model.stationary_disjunc_2 = pyo.Constraint(KKT_model.xindex,\
                                                rule=stationary_disjunctive_2)
        
        ## x Var Conditions ## 
        def x_disjunctive_1(model,i):
            return 0 <= model.x[i]
        
        KKT_model.x_disjunctive_1 = pyo.Constraint(KKT_model.xindex,\
                                                rule=x_disjunctive_1)
        
        def x_disjunctive_2(model,i):
            return model.x[i] <= bigM*(model.z_non_neg[i])
        
        KKT_model.x_disjunctive_2 = pyo.Constraint(KKT_model.xindex,\
                                                rule=x_disjunctive_2)
    
    else:
        print("Error: Bad value for non_negative parameter. Must be 0 or 1.")
        return                

    ##### Step 2: Complementarity Constraints #####
    #We are going to do the disjunctive method ourselves
    if m>0:
        KKT_model.z = pyo.Var(KKT_model.uindex,domain=pyo.Binary) #same number of z vars as u vars
        
        ## Maintaining Feasibility of Ax <= b ##
        def Ax_b_feasibility_rule(model,i):
            return sum(model.A[i,j]*model.x[j] for j in range(1,n+1)) <= model.b[i]
        
        KKT_model.Ax_b_feas = pyo.Constraint(KKT_model.uindex,\
                                             rule=Ax_b_feasibility_rule)
        
        ## Disjunctive Constraints to Ensure Corresponding Rows of 
        ## u == 0 OR Ax == b
        def u_disjunctive(model,i):
            return model.u[i] <= bigM*model.z[i]
        
        KKT_model.u_disjunc = pyo.Constraint(KKT_model.uindex,rule=u_disjunctive)
        
        def Ax_b_disjunctive(model,i):
            return model.b[i] - sum(model.A[i,j]*model.x[j] for j in range(1,n+1))\
                            <= bigM*(1-model.z[i])
        
        KKT_model.Ax_b_disjunc = pyo.Constraint(KKT_model.uindex,rule=Ax_b_disjunctive)
    
    ##### Step 3: Equality Constraints #####
    if p>0:
        def KKT_equality_constraints(model,i):
            return sum(model.D[i,j]*model.x[j] for j in range(1,n+1)) == model.f[i]
        
        KKT_model.equality_conditions = pyo.Constraint(KKT_model.vindex,\
                                            rule=KKT_equality_constraints)            
       
    
    #### Putting Model into Instance Attribute ####
    self.KKT_conditions_model = KKT_model #putting model into attribute
    
def loss_function(self,y,theta=0):         
    #METHOD DESCRIPTION: This method constructs the l(y,u,theta) loss 
    #function defined in Dong, Chen, & Zeng
    
    #As a note, theta == c.  Dong, Chen, & Zeng just happen to use theta
    #in their paper and we use c in our background documentation
    
    #We assume that y has the same dimensions as x, and we also assume that
    #y is inputted as a numpy column vector.
    
    ##################################################################
    
    loss_model = self.KKT_conditions_model.clone() #copying over the KKT_conditions_model
                                            #When we run the receive_data method,
                                            #this should get updated
    
    ##### Step 1: Update the c in the model with theta #####
    ### We need to take the theta parameter (which is the c data)
    ## and put it in the c variables which are fixed
    
    #From the runs, looks like we don't need to unfix the
    #variables in order to reset their values (just have to pass
    #in a dictionary with the right key values)
    
    loss_model.c.set_values(theta) #can just use c directly since copied
                                    #over the KKT_conditions_model
    
    ##### Step 2: Add in the Objective Function #####
    (n,ph) = np.shape(y) #getting dimensions of y
    
    def loss_objective_rule(model):
        return sum((y[j-1,0] - model.x[j])**2 for j in range(1,n+1))
    
    loss_model.obj_func = pyo.Objective(rule=loss_objective_rule)
    
    
    ##### Step 3: Solve the loss_model #####
    solver = SolverFactory("gurobi") #right solver to use because of 
                                    #nonlinear objective function and the 
                                    #possible binary variables
    
    results = solver.solve(loss_model)
    print("This is the termination condition (loss func):",results.solver.termination_condition)
        
    ##Putting Model in Attribute and Put Noisy Decision in Dictionary##
    self.loss_model_dong = loss_model.clone()
    self.noisy_decision_dict[self.dong_iteration_num] = y
    
    return pyo.value(loss_model.obj_func)

def update_rule_optimization_model(self,y,theta,eta_t):
    
    #METHOD DESCRIPTION: This method will construct the update rule optimization
    #model for the Dong et al. algorithm. 
    
    #As a note, theta == c. Dong, Chen, & Zeng just happen to use theta
    #in their paper and we use c in our background documentation
    
    ############################################## 
    
    ### Step 1: Copy over the KKT_conditions_model to update_rule_model ###
    ### and unfix the c variables ###
    update_rule_model = self.KKT_conditions_model.clone() 
    
    update_rule_model.c.unfix() #unfix the c variables (since coming from the KKT_conditions_model
                                #we know the initial value is set at the original inputted c_t data)
                                
    ### Step 2: Set Bounds on c variables (if feasible_set_C tuple has been passed) ###
    ##NEW CODE 5/5/2019
    #Thanks to https://www.geeksforgeeks.org/type-isinstance-python/ for illustrating
    #the use of the isinstance function
    #Thanks to https://www.tutorialspoint.com/python/tuple_len.htm for helping with the length
    #of the tuple
    if self.feasible_set_C is not None: #if something got passed to it
        assert isinstance(self.feasible_set_C,tuple), "Error: If you specify feasible_set_C for the Dong_implicit_update method, must be a tuple"
        assert len(self.feasible_set_C) == 2, "Error: The tuple for feasible_set_C must be of length 2"
        
        (lb,ub) = self.feasible_set_C
        
        def bounds_c_lower(model,i):
            return lb <= model.c[i]
        
        def bounds_c_upper(model,i):
            return model.c[i] <= ub     
        
        update_rule_model.c_bound_lower = pyo.Constraint(update_rule_model.xindex,rule=bounds_c_lower)
        update_rule_model.c_bound_upper = pyo.Constraint(update_rule_model.xindex,rule=bounds_c_upper)
    
    
    ### Step 3: Create the Objective Function ###
    #Decided that I want these to be stand alone methods, so I will have
    #the mechanics methods pass into them
    update_rule_model.c_t = pyo.Param(update_rule_model.xindex,\
                                      initialize=theta) #MAYBE DIDNT actually need this
    
    def obj_rule_update(model):
        norm_c_ct = sum((model.c[i] - model.c_t[i])**2 for i in range(1,len(model.x)+1)) #LEN FUNC HERE
        norm_yt_x = sum((y[i-1,0] - model.x[i])**2 for i in range(1,len(model.x)+1))
        return 0.5*norm_c_ct + eta_t*norm_yt_x
    
    update_rule_model.obj_func = pyo.Objective(rule=obj_rule_update)
    
    
    ### Step 4: Solve the Model and Obtain the Next Guess at Theta ###
    solver = SolverFactory("gurobi") 
    
    results = solver.solve(update_rule_model)
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
    ## Putting Model in Attribute and Returning Guess at Theta ##
    self.update_model_dong = update_rule_model.clone()
    
    new_c_t = update_rule_model.c.extract_values() #gives back a dictionary
    
    return new_c_t    
    
    
        
    
# From Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html
# As the website instructs, we put the methods in this tuple form to 
# be able to pass them into the decorator function. 
# We are just following what the website demonstrates.
    
dong_chen_zeng_funcs = (compute_KKT_conditions,loss_function,update_rule_optimization_model)



