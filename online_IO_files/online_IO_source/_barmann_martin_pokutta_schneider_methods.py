### Methods for B\"armann, Martin, Pokutta, & Schneider 2018 ###

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
        standard_model.x = pyo.Var(standard_model.xindex)
    elif non_negative == 1: 
        standard_model.x = pyo.Var(standard_model.xindex,domain=pyo.NonNegativeReals)
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
                            initialize=data,mutable=self.if_mutable[i]))
                #FOR NOW: NOT GOING TO BAKE IN THE MUTABILITY
                #setattr(standard_model,param_name,pyo.Var(standard_model.xindex))
                #getattr(standard_model,param_name).fix(0) #.fix method can take in a value and
                                #set all the variables to that value
                #standard_model.c.set_values(data) #NEW CHANGE: we are setting the values FOR NOW to the c_data we took in
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
            return sum(model.D[i,j]*model.x[j] for j in range(1,n+1)) <= model.f[i]
        
        standard_model.equality_constraints = pyo.Constraint(standard_model.vindex,\
                                                    rule=equality_constraints_rule)
    
    #### Step 2: Set up the Objective Function ####
    
    if n2 > 0: #if there is indeed a Q (we ALWAYS assume there is a c)
        def obj_func_with_Q(model):
            xt_Q_x_term = sum(sum(model.Q[i,j]*model.x[i]*model.x[j] for i in range(1,n2+1)) for j in range(1,n2+1))
            return (0.5)*xt_Q_x_term + sum(model.c[j]*model.x[j] for j in range(1,n+1))
        
        standard_model.obj_func = pyo.Objective(rule=obj_func_with_Q)
        
    elif n2 == 0: #if there is NO Q
        def obj_func_without_Q(model):
            return sum(model.c[j]*model.x[j] for j in range(1,n+1))
        
        standard_model.obj_func = pyo.Objective(rule=obj_func_without_Q)
    
    else:
        print("Incorrect value for dim of Q.  Somehow you put in a negative value...")
        return        
    
    self.BMPS_subproblem = standard_model #BMPS call this model with the changing c 
                                        #"subproblem (2)" so we have named it "BMPS_subproblem"
    
    
    

def compute_diam_F(self,dimc=(0,0)):
    #So we can indeed use 1/sqrt(t) as our learning rate but
    #multiplying this by D/G does help things
    #So I think we will give it as an option to users.  And we will
    #Show in the computational experiment both options
    #Think will return D/G for each iteration 
    #SHOULD CALL BETWEEN STEPS 3 AND 4 of the algorithm
    
    #Put an objective function on F and solve (for D)
    #Going to Assume that self.feasible_set_C is a pyomo model
    #MIGHT NEED TO PROVIDE A TEMPLATE FOR how to input c - I'm gonna make
    #some assumptions about c - might need an example or two
    
    #c is assumed to be a column vector
    
    #if first time calling the compute_learning_rate function, need to 
    #calculate the diameter of F.  Otherwise, diameter of F doesn't change
    #so we don't need to keep changing it for each iteration
    
    
    ##### Step 1: Find the Diameter of the Set F (set C) #####
    #Going to assume the variable is named c in this case
    
    set_C = self.feasible_set_C.clone()
    (n,ph) = dimc #getting number of variables
    
    set_C_squared = pyo.ConcreteModel() #creating another Pyomo model with two copies of
                                        #the constraints from set_C bc we want to find maximum
                                        #distance between two points in the set
    
    ### Creating two Copies of the Constraints ###
    def set_C_block_func(b):
        b.cindex = pyo.RangeSet(n)
        b.c = pyo.Var(b.cindex)
        #For this to work, will need to require people to define
        #constraints with model.c
        for constr in set_C.component_objects(pyo.Constraint):
            setattr(b,constr.name,pyo.Constraint(constr._index,rule=constr.rule))
    
    set_C_squared.block1 = pyo.Block(rule=set_C_block_func)
    set_C_squared.block2 = pyo.Block(rule=set_C_block_func) #should create TWO sets of variables
    
    pdb.set_trace()
    
    ### Defining Equalities for y[i] = c1[i] - c2[i] ###
    set_C_squared.cvarindex = pyo.RangeSet(n)
    set_C_squared.y = pyo.Var(set_C_squared.cvarindex)
    
    def equalities_for_c_and_y(model,i):
        return model.block1.c[i]-model.block2.c[i] == set_C_squared.y[i]
    
    set_C_squared.equality_for_c_and_y = pyo.Constraint(set_C_squared.cvarindex,\
                                                rule=equalities_for_c_and_y)
    
    
    ### Defining Objective Function for set_C_squared ###
    def set_C_obj_func(model):
        return -1*sum((model.y[i])**2 for i in range(1,n+1)) #minimizing the objective
    
    set_C_squared.obj_func = pyo.Objective(rule=set_C_obj_func)
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(set_C_squared)
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
    print("This is the objective function value (multiplied by -1 and then square rooted):",math.sqrt(-1*pyo.value(set_C_squared.obj_func)))
    
    self.D = math.sqrt(-1*pyo.value(set_C_squared.obj_func))
    
    #might need a RETURN statement, depending upon how implement this stuff
    #in the online_IO shared methods
    
    
def compute_diam_X_pt(self,dimc=(0,0)):    
    ##### Step 1: Find the Diameter of X_t ######
    # Problem is that in theory we would want all of the p_t
    # But the entire point of the online methods is that we don't have all of
    # the p_t at the beginning (or all of the decisions)
    # Need an upper bound on the X(p_t) set diameters 
    
    # Going to assume that the compute_standardized_model method has been called
    # before this AND going to assume has been updated
    BMPS_subproblem = self.BMPS_subproblem.clone()
    (n,ph) = dimc #getting number of variables
    
    set_X_pt_squared = pyo.ConcreteModel()
    
    def set_X_pt_block_func(b):
        b.xindex = pyo.RangeSet(n)
        b.x = pyo.Var(b.xindex)
        #For this to work, will need to require people to define
        #constraints with model.x
        for constr in BMPS_subproblem.component_objects(pyo.Constraint):
            setattr(b,constr.name,pyo.Constraint(constr._index,rule=constr.rule)) #should use the 
            #rule functions from the previous model WITH the new variables
            #because we made the rules as involving model.x
    
    set_X_pt_squared.block1 = pyo.Block(rule=set_X_pt_block_func)
    set_X_pt_squared.block2 = pyo.Block(rule=set_X_pt_block_func) #should create TWO sets of variables
    
    ### Defining Equalities for Objective Function:  ###
    set_X_pt_squared.xvarindex = pyo.RangeSet(n)
    set_X_pt_squared.y = pyo.Var(set_X_pt_squared.xvarindex)
    
    def equalities_for_y_and_x(model,i):
        return set_X_pt_squared.y[i] == set_X_pt_squared.block1.x[i] - set_X_pt_squared.block2.x[i]
    
    set_X_pt_squared.equalities_constraints = pyo.Constraint(set_X_pt_squared.xvarindex,\
                                                    rule=equalities_for_y_and_x)
    
    ### Defining Objective Function for set_X_pt_squared ###
    def set_X_pt_obj_func(model):
        return -1*sum((model.y[i])**2 for i in range(1,n+1)) #minimizing the objective
    
    set_X_pt_squared.obj_func = pyo.Objective(rule=set_X_pt_obj_func)
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(set_X_pt_squared)
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
    print("This is the objective function value (multiplied by -1 and then square rooted):",\
          math.sqrt(-1*pyo.value(set_X_pt_squared.obj_func)))
    
    
    G = math.sqrt(-1*pyo.value(set_X_pt_squared.obj_func))
    
    ### Maximum Between this G and Previous Gs ###
    max_G = max(self.G_max,G)
    
    self.G_max = max_G #max_G is a local variable
    
    #MIGHT NEED A RETURN STATEMENT!
    

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
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
    ### Extracting Solution ###
    ct_vals = region_F.c.extract_values() #obtain a dictionary
    
    self.c_t_BMPS = ct_vals #ct_col_vec #will need to update subproblem with this
    
    
def solve_subproblem(self):
    #Once we implement the compute_standardized_model method, we can easily
    #solve the updated subproblem through this
    #ASSUME EVERYTHING HAS BEEN UPDATED - the objective function and the constraints
    
    subproblem = self.BMPS_subproblem.clone()
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(subproblem)
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
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
    
    y_t = ct_col_vec - eta_t*(self.xbar_t_BMPS - self.x_t_BMPS)
    
    return y_t #going to leave it this way for now because I pass in y_t to
    #the next function
    #IT DOES SEEM SLIGHTLY RANDOM
    #It is kind of nice because its the final method in the "next_iteration" thing
    
    

barmann_martin_pokutta_schneider_funcs = (compute_standardized_model,compute_diam_F,\
                compute_diam_X_pt,project_to_F,solve_subproblem,gradient_step)


