#### Class for Online Inverse Optimization Methods ####
##These methods come from Dong et al. 2018 and B\"armann et al. 2018


import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math

class Online_IO(): 
    
    def __init__(self,initial_model,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
                 dimQ=(0,0),dimc=(0,0),dimA=(0,0),dimD=(0,0),binary_mutable=[0,0,0,0,0,0],non_negative=0):
        
        #We are assuming that binary_mutable is in the order of Q,c,A,b,D,f
        
        self.initial_model = initial_model #load initial pyomo model into attribute
        self.obj_vals = {} #going to initialize as a dictionary to make easier to handle
                            #different types of objective functions
        self.alg_specification = None #not initialized until call "initialize_IO_method"
                                        #by setting as None, can use "is __ none" method
        self.model_data_names = {'Q':Qname,'c':cname,'A':Aname,'b':bname,'D':Dname,'f':fname}
        self.model_data_dimen = {'Q':dimQ,'c':dimc,'A':dimA,'D':dimD}
        self.num2param_key = {1:'Q',2:'c',3:'A',4:'b',5:'D',6:'f'} 
        self.non_negative = non_negative
        
        ## Dictionary for Mutable/Not Mutable ##
        #Assume the vector is in the order of Q,c,A,b,D,f
        mutability = {}
        mutable_params = {}
        #counter = 1
        for (i,param_name) in self.num2param_key.items():
            if binary_mutable[i-1] == 0:
                mutability[i] = 'False'    #decided to index from 1 since this will
                                            #be used in the pyomo model
            elif binary_mutable[i-1] == 1:
                mutability[i] = 'True'
                mutable_params[param_name] = self.model_data_names[param_name] #NEW: subset of model_data_names
                #mutable_params[counter] = self.num2param_key[i+1] #creating a list of the mutable params
                #counter = counter + 1
            else:
                print('Error: There should only be 0s and 1s in the binary_mutable parameter.')
                return
            
        self.if_mutable = mutability 
        self.model_data_names_mutable = mutable_params #NEW: (paramname,username) subset of model_data_names
        
        ##### Dong_implicit_update Attributes #####
        self.KKT_conditions_model = None
        self.loss_model_dong = None
        self.noisy_decision_dong = None #holds the next noisy decision 
                                        #(if passed in, assume numpy column vector)
        self.dong_iteration_num = 1
        self.c_t_dong = None
        self.c_t_dict_dong = {}
        self.update_model_dong = None
        self.losses_dong = [] #MIGHT CHANGE to numpy data at some point
    
    
    def initialize_IO_method(self,alg_specification):
        ### Each algorithm has its own set up procedure before the first iteration
        if alg_specification=="Dong_implicit_update":  
            ### Step 0: Set Algorithm Name
            self.alg_specification = alg_specification 
            
            #### Step 1: Obtain c values to put into c_t_guess ####
            #We assume that the initial guess for c_t is in the parameter
            #that corresponds with c
            data = getattr(self.initial_model,self.model_data_names['c'])
            data = data.extract_values() #produces a dictionary
            self.c_t_dong = data 
            #NEW CODE
            self.c_t_dict_dong[self.dong_iteration_num] = self.c_t_dong
                    
            #### Step 2: Construct the Initial KKT conditions Model ####        
            #We can use getattr() to pass in the parameter values into the 
            #compute_KKT_conditions function
            #Thank you to:
            #https://stackoverflow.com/questions/2612610/how-to-access-object-attribute-given-string-corresponding-to-name-of-that-attrib
            #https://docs.python.org/2/library/functions.html#getattr
            self.compute_KKT_conditions(dimQ=self.model_data_dimen['Q'],\
                dimc=self.model_data_dimen['c'],dimA=self.model_data_dimen['A'],\
                dimD=self.model_data_dimen['D'],non_negative=self.non_negative) 

        else:
            print("Error, we do not support inputted method. (It is possible",\
                    "that you spelled something wrong.)")
            return
        
            
    
    def receive_data(self,p_t=None,x_t=None):
        
        #METHOD DESCRIPTION: We are assuming that if a parameter is mutable then the 
        #entire parameter block is mutable.  Users cannot just pass in single values.
        
        if self.alg_specification is None:
            print("Error: Must initialize an online algorithm before using this method. ",\
                  "You will need to use .initialize_IO_method.")
            return 0
        elif self.alg_specification == "Dong_implicit_update":
            if p_t is not None:
                for (class_pname,user_pname) in self.model_data_names_mutable.items():
                    data = p_t[user_pname]
                    getattr(self.KKT_conditions_model,class_pname).clear() #clearing the attribute
                    getattr(self.KKT_conditions_model,class_pname).reconstruct(data) #reconstruct the attribute
                    ### Obtain All of the Constraint Components of the KKT_conditions
                    ## Model and Reconstructing Them
                    for expr in self.KKT_conditions_model.component_objects(pyo.Expression):
                        expr.reconstruct()
                    for constr in self.KKT_conditions_model.component_objects(pyo.Constraint):
                        constr.reconstruct()
                    
                    #setattr(self.KKT_conditions_model,class_pname,data)
            if x_t is not None:
                self.noisy_decision_dong = x_t #putting the noisy decision in an 
                                            #attribute that can be shared among the other methods
            
    def next_iteration(self,eta_factor=1):
        
        # METHOD DESCRIPTION: This method carries out another iteration of 
        # the algorithm with which the instance of the object has been working 
        
        #Maybe I can implement the passing in of the theta
        #by setting a c attribute that gets initially filled
        #when I initiate Dong et al, then the attribute gets updated with
        #new parameter update at the end of the algorithm
        
        if self.alg_specification is None:
            print("Error: Must initialize an online algorithm before using this method. ",\
                  "You will need to use .initialize_IO_method.")
            return 0
        elif self.alg_specification == "Dong_implicit_update":
            self.dong_iteration_num = self.dong_iteration_num + 1 #increasing the iteration count
            ### Step 1: Calculating Loss ###
            loss_this_iteration = self.loss_function(y=self.noisy_decision_dong,\
                                            theta=self.c_t_dong,if_solve=1)
            self.losses_dong.append(loss_this_iteration)
            
            #### Step 2: Update Rule ####
            eta = eta_factor*(1/(math.sqrt(self.dong_iteration_num)))
            
            self.c_t_dong = self.update_rule_optimization_model(y=self.noisy_decision_dong,\
                                                theta=self.c_t_dong,eta_t=eta)
            
            self.c_t_dict_dong[self.dong_iteration_num] = self.c_t_dong
        
    def compute_KKT_conditions(self,dimQ=(0,0),dimc=(0,0),dimA=(0,0),\
                               dimD=(0,0),non_negative=0,bigM=10000):
                
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
        
    def loss_function(self,y,theta=0,if_solve=1):        
        #This method constructs the l(y,u,theta) loss function defined in 
        #Dong et al.
        #In order to run this method, would need to run the 
        #initialize method first. 
        #WE DO NOT have a parameter for u because u would have been taken
        #care of in the ``receive_data'' method
        
        #We assume that y has the same dimensions as x, and we also assume that
        #y is inputted as a numpy column vector.
        
        loss_model = self.KKT_conditions_model.clone() #copying over the KKT_conditions_model
                                                #When we run the receive_data method,
                                                #this should get updated
        
        ##### Step 1: Update the c in the model with theta #####
        ### We need to take the theta parameter (which is the c data)
        ## and put it in the c variables which are fixed
        
        #From the runs of things, looks like we don't need to unfix the
        #variables in order to reset their values (just have to pass
        #in a dictionary with the right key values)
        
        #pdb.set_trace()
        
        loss_model.c.set_values(theta) #can just use c directly since copied
                                        #over the KKT_conditions_model
        #pdb.set_trace()
        
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
        
#        if results.solver.termination_condition == "optimal":
#            pass
#        else:
#            pdb.set_trace()
        
        ##Putting Model in Attribute##
        self.loss_model_dong = loss_model.clone()
        
        return pyo.value(loss_model.obj_func)
    
    def update_rule_optimization_model(self,y,theta,eta_t):
        
        #METHOD DESCRIPTION: This method will construct the update rule optimization
        #model for the Dong et al. algorithm. 
        
        #Inputs: y - (as the ``noisy decision'' from Dong et al lingo) assumed
        #to be a numpy column vector
        #theta - as the last guess for the parameterization of c, assumed to
        #be a dictionary
        #eta - the ``learning rate'' 
        
        ### Step 1: Copy over the KKT_conditions_model to update_rule_model ###
        ### and unfix the c variables ###
        update_rule_model = self.KKT_conditions_model.clone() 
        
        #pdb.set_trace()
        
        update_rule_model.c.unfix() #unfix the c variables (since coming from the KKT_conditions_model
                                    #we know the initial value is set at 0)
        
        ### Step 2: Create the Objective Function ###
        #Decided that I want these to be stand alone methods, so I will have
        #the mechanics methods pass into them
        update_rule_model.c_t = pyo.Param(update_rule_model.xindex,\
                                          initialize=theta) #MAYBE DIDNT actually need this
        
        def obj_rule_update(model):
            norm_c_ct = sum((model.c[i] - model.c_t[i])**2 for i in range(1,len(model.x)+1)) #LEN FUNC HERE
            norm_yt_x = sum((y[i-1,0] - model.x[i])**2 for i in range(1,len(model.x)+1))
            return 0.5*norm_c_ct + eta_t*norm_yt_x
        
        update_rule_model.obj_func = pyo.Objective(rule=obj_rule_update)
        
        #pdb.set_trace()
        
        ### Step 3: Solve the Model and Obtain the Next Guess at Theta ###
        solver = SolverFactory("gurobi") #right solver to use because of 
                                        #nonlinear objective function and the 
                                        #possible binary variables
        
        results = solver.solve(update_rule_model)
        print("This is the termination condition (update rule):",results.solver.termination_condition)
        
        ## Putting Model in Attribute and Returning Guess at Theta ##
        self.update_model_dong = update_rule_model.clone()
        
        new_c_t = update_rule_model.c.extract_values() #gives back a dictionary
        
        return new_c_t
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
