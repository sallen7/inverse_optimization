### Functions that Didn't Work for BMPS ###

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
    
    set_C = self.feasible_set_C.clone() #MAKE SURE ALL THE DATA STUFF HAPPENED CORRECTLY - THAT ME REMOVING CLONE FROM FEASIBLE SET WASNT A PROBLEM
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
    
    #pdb.set_trace()
    
    ### Defining Equalities for y[i] = c1[i] - c2[i] ###
    set_C_squared.cvarindex = pyo.RangeSet(n)
    set_C_squared.y = pyo.Var(set_C_squared.cvarindex)
    
    def equalities_for_c_and_y(model,i):
        return model.block1.c[i]-model.block2.c[i] == model.y[i]
    
    set_C_squared.equality_for_c_and_y = pyo.Constraint(set_C_squared.cvarindex,\
                                                rule=equalities_for_c_and_y)
    
    
    ### Defining Objective Function for set_C_squared ###
    def set_C_obj_func(model):
        return -1*sum((model.y[i])**2 for i in range(1,n+1)) #minimizing the objective
    
    set_C_squared.obj_func = pyo.Objective(rule=set_C_obj_func)
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(set_C_squared,tee=True)
    print("This is the termination condition (update rule):",results.solver.termination_condition)
    
    print("This is the objective function value (multiplied by -1 and then square rooted):",math.sqrt(-1*pyo.value(set_C_squared.obj_func)))
    
    self.D = math.sqrt(-1*pyo.value(set_C_squared.obj_func))
    
    pdb.set_trace()
    
    #pdb.set_trace()
    
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
    
    #NEED A LOWER BOUND!! - YOU NEED TO HAVE A BOUNDED FEASIBLE REGION!
    
    BMPS_subproblem = self.BMPS_subproblem.clone()
    (n,ph) = dimc #getting number of variables
    
    set_X_pt_squared = pyo.ConcreteModel()
    
    #pdb.set_trace() #I THINK WE NEED TO BRING IN THE PARAMETERS INTO THE BLOCKS
    
    def set_X_pt_block_func(b):
        b.xindex = pyo.RangeSet(n)
        if self.non_negative == 1:
            b.x = pyo.Var(b.xindex,domain=pyo.NonNegativeReals)
        else:
            b.x = pyo.Var(b.xindex)
        #For this to work, will need to require people to define
        #constraints with model.x
        for param in BMPS_subproblem.component_objects(pyo.Param):
            #pdb.set_trace()
            setattr(b,param.name,pyo.Param(param._index,initialize=param._data))
            #WILL NEED TO SEE IF THE _index DOES THE RIGHT INDICE STUFF FOR A?        
        for constr in BMPS_subproblem.component_objects(pyo.Constraint):
            #pdb.set_trace()
            setattr(b,constr.name,pyo.Constraint(constr._index,rule=constr.rule)) #should use the 
            #rule functions from the previous model WITH the new variables
            #because we made the rules as involving model.x
    
    set_X_pt_squared.block1 = pyo.Block(rule=set_X_pt_block_func)
    set_X_pt_squared.block2 = pyo.Block(rule=set_X_pt_block_func) #should create TWO sets of variables
    
    ### Defining Equalities for Objective Function:  ###
    set_X_pt_squared.xvarindex = pyo.RangeSet(n)
    set_X_pt_squared.y = pyo.Var(set_X_pt_squared.xvarindex)
    
    def equalities_for_y_and_x(model,i):
        return model.block1.x[i] - model.block2.x[i] == model.y[i]
    
    set_X_pt_squared.equalities_constraints = pyo.Constraint(set_X_pt_squared.xvarindex,\
                                                    rule=equalities_for_y_and_x)
    
    ### Defining Objective Function for set_X_pt_squared ###
    def set_X_pt_obj_func(model):
        return -1*sum((model.y[i])**2 for i in range(1,n+1)) #minimizing the objective
    
    set_X_pt_squared.obj_func = pyo.Objective(rule=set_X_pt_obj_func)
    
    #pdb.set_trace()
    
    ### Solving ###
    solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
    results = solver.solve(set_X_pt_squared)
    print("This is the termination condition (compute_diam_F):",results.solver.termination_condition)
    
    print("This is the objective function value (multiplied by -1 and then square rooted):",\
          math.sqrt(-1*pyo.value(set_X_pt_squared.obj_func)))
    
    
    G = math.sqrt(-1*pyo.value(set_X_pt_squared.obj_func))
    
    ### Maximum Between this G and Previous Gs ###
    max_G = max(self.G_max,G)
    
    self.G_max = max_G #max_G is a local variable
    
    #MIGHT NEED A RETURN STATEMENT!
