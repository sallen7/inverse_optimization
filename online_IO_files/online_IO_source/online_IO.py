#### Class for Online Inverse Optimization Methods ####
## These methods come from Dong, Chen, & Zeng 2018 
## and B\"armann, Martin, Pokutta, & Schneider 2018

#Big thanks to: http://www.qtrac.eu/pyclassmulti.html for explaining
#how to break up methods to multiple files

#Helpful for telling me about stale variables:
#https://stackoverflow.com/questions/55711079/what-does-it-mean-when-a-variable-returns-stale-true

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math

from pyomo.core.expr import current as EXPR #trying something

from online_IO_files.online_IO_source.decorator_for_online_IO import func_adds_methods
from online_IO_files.online_IO_source._dong_chen_zeng_methods import dong_chen_zeng_funcs #importing the Dong, Chen, Zeng methods
from online_IO_files.online_IO_source._barmann_martin_pokutta_schneider_methods import barmann_martin_pokutta_schneider_funcs

#Can concatanate tuples using + operator
#https://www.digitalocean.com/community/tutorials/understanding-tuples-in-python-3

@func_adds_methods(dong_chen_zeng_funcs+barmann_martin_pokutta_schneider_funcs)
class Online_IO(): 
    
    def __init__(self,initial_model=None,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
                 dimQ=(0,0),dimc=(0,0),dimA=(0,0),dimD=(0,0),binary_mutable=[0,0,0,0,0,0],non_negative=0,\
                 feasible_set_C=None): 
        
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
        
        self.feasible_set_C = feasible_set_C #NEW CODE 5/5/2019 feasible_set_C.clone()
                            #NEED TO CHECK THAT NOT USING .CLONE ISN'T GONNA BE AN ISSUE!
        
        #pdb.set_trace()
        
        ##### Dong_implicit_update Attributes #####
        self.KKT_conditions_model = None
        self.batch_model = pyo.ConcreteModel() #initiating the batch model 
        self.loss_model_dong = None
        self.noisy_decision_dong = None #holds the next noisy decision 
                                        #(if passed in, assume numpy column vector)
        self.noisy_decision_dict = {}
        
        self.dong_iteration_num = 0
        self.c_t_dong = None
        self.c_t_dict_dong = {}
        self.update_model_dong = None
        self.losses_dong = [] #MIGHT CHANGE to numpy data at some point
        self.opt_batch_sol = []
        
        ###### BMPS_online_GD Attributes #####
        #self.var_bounds = var_bounds #if there are variable bounds, 
                                    #needs to be passed in as a tuple
                                    #FOR NOW, only enabling for BMPS
        self.BMPS_subproblem = None
        self.project_to_F_model = None
        self.D = None
        self.G_max = 0 #initializing max of diam(X_pt)
        self.c_t_BMPS = None
        self.xbar_t_BMPS = None
        self.x_t_BMPS = None
        self.y_t_BMPS = None
        self.BMPS_iteration_number = 0
        self.if_use_diam = 1
    
    
    def initialize_IO_method(self,alg_specification,alg_specific_params=None):
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
        
        elif alg_specification == "BMPS_online_GD":
            ### Step 0: Check that have a feasible_set_C Param and Set Algorithm Name ###
            ##NEW CODE 5/5/2019
            assert self.feasible_set_C is not None, "Error: you need to specify a feasible_set_C for BMPS_online_GD"
            
            self.alg_specification = alg_specification
            
            #pdb.set_trace()
            
            ### Step 1: Call Method to Create BMPS_subproblem ###
            self.compute_standardized_model(self.model_data_dimen['Q'],\
                dimc=self.model_data_dimen['c'],dimA=self.model_data_dimen['A'],\
                dimD=self.model_data_dimen['D'],non_negative=self.non_negative)
            
            ### Step 2: Moving y_t guess (which is c_t) into self.y_t_BMPS ###
            data = getattr(self.initial_model,self.model_data_names['c'])
            data = data.extract_values() #produces a dictionary
            data_vec = np.fromiter(data.values(),dtype=float,count=len(data))
            data_vec_col = np.reshape(data_vec,(len(data),1))
            
            self.y_t_BMPS = data_vec_col
            
            ### Step 3: Checking for Parameters ###
            if alg_specific_params is not None:
                self.if_use_diam = alg_specific_params['diam_flag']
        

        else:
            print("Error, we do not support inputted method. (It is possible",\
                    "that you spelled something wrong.)")
            return
        
            
    
    def receive_data(self,p_t=None,x_t=None):
        
        #METHOD DESCRIPTION: We are assuming that if a parameter is mutable then the 
        #entire parameter block is mutable.  Users cannot just pass in single values.
        
        #Assume that x_t is a numpy vector
        
        if self.alg_specification is None:
            print("Error: Must initialize an online algorithm before using this method. ",\
                  "You will need to use .initialize_IO_method.")
            return 0
        elif self.alg_specification == "Dong_implicit_update":
            if p_t is not None:
                assert type(p_t) is dict #making sure p_t is a dictionary
                
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
                
                (n,ph) = self.model_data_dimen['c']
                #Assert Statement to Make sure Passed in a column vector
                assert x_t.shape == (n,1),"Error: You did not pass a column vector of correct dimensions"
                self.noisy_decision_dong = x_t #putting the noisy decision in an 
                                            #attribute that can be shared among the other methods
                                            
        elif self.alg_specification == "BMPS_online_GD":
            #Make sure to reconstruct the objective function too when you update the 
            #subproblem model
            
            #Yeah I'll put the updating of c_t in the subproblem here as well
            #Currently in c_t_BMPS - passing a dictionary to this method, will
            #convert the dictionary to a column vector in gradient descent step later
            
            #So, we do have x_t in this algorithm, but the p_t and the x_t are
            #like "separately received" - so I think I should receive the x_t data
            #through this method (bc it is the method's job after all) but we can store
            #it in an attribute when we receive it, which will then be passed onto 
            #the gradient_step method
            
            if p_t is not None: #we assume it is a dictionary
                assert type(p_t) is dict #making sure passed in a dictionary
                
                ### Updating X(p_t) in BMPS_subproblem ###
                for (class_pname,user_pname) in self.model_data_names_mutable.items():
                    data = p_t[user_pname]
                    getattr(self.BMPS_subproblem,class_pname).clear() #clearing the attribute
                    getattr(self.BMPS_subproblem,class_pname).reconstruct(data) #reconstruct the attribute
                    
                    ### Obtain All of the Constraint Components of the BMPS_subproblem
                    ## Model and Reconstructing Them
                    for constr in self.BMPS_subproblem.component_objects(pyo.Constraint):
                        constr.reconstruct()
                
                ### Updating c_t in BMPS subproblem (using c_t_BMPS) ###
                getattr(self.BMPS_subproblem,'c').clear()
                getattr(self.BMPS_subproblem,'c').reconstruct(self.c_t_BMPS) #doesn't work
                #and would need to hack the code - not going to do
                #BACK UP: just delete the objective function and copy and paste the rule from
                #the _BMPS_methods file - the .c part DOES work
                
                #Deleting the objective function#
                #HOPING THIS KEEPS BETWEEN ITERATIONS
                #REALLY NEED TO CHECK THAT THIS KEEPS AND THAT THE 
                #INDEXED COMPONENTS ARE BEING reconstructed THE WAY I THINK THEY
                #ARE SUPPOSED TO BE
                self.BMPS_subproblem.del_component(self.BMPS_subproblem.obj_func)
                
                (n2,ph) = self.model_data_dimen['Q'] #to decide upon the objective rule
                (n,ph) = self.model_data_dimen['c']
                
                ###### Recreating the Objective Function Rule ######
                if n2 > 0: #if there is indeed a Q (we ALWAYS assume there is a c)
                    def obj_func_with_Q(model):
                        xt_Q_x_term = sum(sum(model.Q[i,j]*model.x[i]*model.x[j] for i in range(1,n2+1)) for j in range(1,n2+1))
                        return (0.5)*xt_Q_x_term + sum(model.c[j]*model.x[j] for j in range(1,n+1))
                    
                    self.BMPS_subproblem.obj_func = pyo.Objective(rule=obj_func_with_Q,sense=pyo.minimize)
                    
                elif n2 == 0: #if there is NO Q
                    def obj_func_without_Q(model):
                        return sum(model.c[j]*model.x[j] for j in range(1,n+1))
                    
                    self.BMPS_subproblem.obj_func = pyo.Objective(rule=obj_func_without_Q,sense=pyo.minimize)
                
                else:
                    print("Incorrect value for dim of Q.  Somehow you put in a negative value...")
                    return        
                
                #pdb.set_trace()
                #EXPR.expression_to_string(self.BMPS_subproblem.obj_func._data[None])
                
                
                ######THIS DOES WORK EVEN THOUGH THE CODE IS CONFUSED########
                #for obj in self.BMPS_subproblem.component_objects(pyo.Objective):
                #    obj.reconstruct() #using the objective rule 
                    
                ### Checking that Everything Has been Constructed at the End of the Update ###
                # Also going to check the bounds of the constraints
                for constr in self.BMPS_subproblem.component_objects(pyo.Constraint):
                    #DOES NOT WORK AS WELL AS WE THOUGHT!!!
                    assert constr._constructed == True, "Error in constraint construction (body)"
                    for c in constr:#LOOK AT THE JUPYTER NOTEBOOK!!!
                        #assert constr[c]._constructed == True, "Error in constraint construction (body)"
                        lb = constr[c].lower
                        ub = constr[c].upper
                        assert ((lb is not None) or (ub is not None)), "Error in constraint construction (LHS/RHS)"
                
                for obj in self.BMPS_subproblem.component_objects(pyo.Objective):
                    assert obj._constructed == True, "Error in construction of objective function"
                
            
            if x_t is not None:
                
                (n,ph) = self.model_data_dimen['c']
                assert x_t.shape == (n,1),"Error: You did not pass a column vector of corrected dimensions"
                self.x_t_BMPS = x_t #assume get a column vector
        
            
    def next_iteration(self,eta_factor=1,part_for_BMPS=1):
        
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
            
            #### Step 3: Calculate Batch Solution ####
            self.calculate_batch_sol(dimQ=self.model_data_dimen['Q'],dimc=self.model_data_dimen['c'],\
                                     dimA=self.model_data_dimen['A'],\
                                     dimD=self.model_data_dimen['D'])
            
            
        elif self.alg_specification == "BMPS_online_GD":
            if part_for_BMPS == 1: #part 1 for algorithm
                ### Step 0: Update Iteration Count ###
                self.BMPS_iteration_number = self.BMPS_iteration_number + 1
                #pdb.set_trace()
                ### Step 1: Project to F (stores the c_t in c_t_BMPS) ###
                self.project_to_F(dimc=self.model_data_dimen['c'],y_t=self.y_t_BMPS)
            
            elif part_for_BMPS == 2: #part 2 for algorithm
                ### Step 3 (with Step 2 being that parameters were updated) ###
                ### Solve the Subproblem X(p_t) ###
                self.solve_subproblem() #obtain xbar
                
                ### Step 4: Calculate Learning Rate ###
                #Based upon the self.if_use_diam flag (set up in initialization)#
                if self.if_use_diam == 1: #if we ARE using diameters
                    if self.dong_iteration_num < 2: #so if on first iteration, need to compute D
                        self.compute_diam_F(dimc=self.model_data_dimen['c'])
                    
                    self.compute_diam_X_pt(dimc=self.model_data_dimen['c']) #need to update each iteration
                    
                    eta_calculated = (1/math.sqrt(self.BMPS_iteration_number))*(self.D/self.G_max) 
                elif self.if_use_diam == 0: #if we are NOT using diameters
                    eta_calculated = (1/math.sqrt(self.BMPS_iteration_number))
                
                ### Step 5: Gradient Descent Step ###
                #receive_data will have put the x_t into x_t_BMPS
                self.y_t_BMPS = self.gradient_step(eta_t=eta_calculated,x_t=self.x_t_BMPS)
                
            else:
                print("Error: You can only enter 1 or 2 for the part_for_BMPS parameter")
                return
                
            
            
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
