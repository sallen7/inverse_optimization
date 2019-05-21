#### online_IO.py: Class for Online Inverse Optimization Methods ####
#5/19/2019

# This file contains the initialization method for and the mechanics methods
# for the Online_IO class.  Readers/users can see Section 1.2.2: The Online_IO Class
# in the Chapter documentation to find more information regarding these methods

# The Online_IO class supports online inverse optimization algorithms from
# the following papers: Dong, Chen, & Zeng 2018 and 
# B\"armann, Martin, Pokutta, & Schneider 2018

# Additional Notes:

#Big thanks to Qtrac Ltd.: http://www.qtrac.eu/pyclassmulti.html for explaining
#how to break up methods to multiple files.  We follow the procedure outlined
#in this website and import both the decorator function and the other
#functions/methods from the additional class files into this one, and then
#we decorate the Online_IO class. These are more helpful websites for decorators:

#https://realpython.com/primer-on-python-decorators/
#https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
#https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
#(have a good example with decorators here)

#Helpful for telling me about stale variables:
#https://stackoverflow.com/questions/55711079/what-does-it-mean-when-a-variable-returns-stale-true

#Can concatanate tuples using + operator
#https://www.digitalocean.com/community/tutorials/understanding-tuples-in-python-3

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

## Importing the decorator function and the methods/functions from the other files ##
from online_IO_files.online_IO_source.decorator_for_online_IO import func_that_adds_methods
from online_IO_files.online_IO_source._dong_chen_zeng_methods import dong_chen_zeng_funcs #importing the Dong, Chen, Zeng methods
from online_IO_files.online_IO_source._barmann_martin_pokutta_schneider_methods import barmann_martin_pokutta_schneider_funcs


## Decorating the Online_IO class ##
@func_that_adds_methods(dong_chen_zeng_funcs+barmann_martin_pokutta_schneider_funcs)
class Online_IO(): 
    
    def __init__(self,initial_model=None,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
                 dimQ=(0,0),dimc=(0,0),dimA=(0,0),dimD=(0,0),binary_mutable=[0,0,0,0,0,0],non_negative=0,\
                 feasible_set_C=None,var_bounds=None): 
        
        #METHOD DESCRIPTION: This initialization method sets up an instance of
        #the Online_IO class.  It puts values provided in attributes, constructs
        #various dictionaries from the provided values, and initializes attributes.
        
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
        
        ##### Dong_implicit_update Attributes #####
        self.KKT_conditions_model = None
        self.batch_model = pyo.ConcreteModel() #initiating the batch model
                                            #(hoping to have in future iterations of the
                                            #code)
        self.loss_model_dong = None
        self.noisy_decision_dong = None #holds the next noisy decision 
                                        #(if passed in, assume numpy column vector)
        self.noisy_decision_dict = {}
        
        self.dong_iteration_num = 0
        self.c_t_dong = None
        self.c_t_dict_dong = {}
        self.update_model_dong = None
        self.losses_dong = [] 
        self.opt_batch_sol = []
        
        ###### BMPS_online_GD Attributes #####
        self.var_bounds = var_bounds #if there are variable bounds, 
                                    #needs to be passed in as a tuple
                                    #FOR NOW, only enabling for BMPS
        self.BMPS_subproblem = None
        self.project_to_F_model = None
        self.D = None #these next two attributes are not relevant right now
        self.G_max = 0 #initializing max of diam(X_pt)
        self.c_t_BMPS = None
        self.xbar_t_BMPS = None
        self.x_t_BMPS = None
        self.y_t_BMPS = None
        self.BMPS_iteration_number = 0
        self.if_use_diam = 0 #UPDATE 5/8/2019 - we change this to 0
    
    
    def initialize_IO_method(self,alg_specification,alg_specific_params=None):
        #METHOD DESCRIPTION: Each algorithm has its own set up procedure 
        #before the first iteration.  This method takes care of that.
        
        if alg_specification=="Dong_implicit_update":  
            ### Step 0: Set Algorithm Name
            self.alg_specification = alg_specification 
            
            #### Step 1: Obtain c values to put into c_t_guess ####
            #We assume that the initial guess for c_t is in the parameter
            #that corresponds with c
            data = getattr(self.initial_model,self.model_data_names['c'])
            data = data.extract_values() #produces a dictionary
            self.c_t_dong = data 
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
            assert self.feasible_set_C is not None, "Error: you need to specify a feasible_set_C for BMPS_online_GD"
            
            self.alg_specification = alg_specification
            
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
        
        #METHOD DESCRIPTION: This method does the updating with p_t and puts the
        #external entity's x_t into the appropriate attribute (depending upon
        #the method)
        
        #We are assuming that if a parameter is mutable then the 
        #entire parameter block is mutable.  Users cannot just pass in single values.
        
        #Assume that x_t is a numpy vector
        
        if self.alg_specification is None:
            print("Error: Must initialize an online algorithm before using this method. ",\
                  "You will need to use .initialize_IO_method.")
            return 0
        
        elif self.alg_specification == "Dong_implicit_update":
            if p_t is not None:
                assert type(p_t) is dict #making sure p_t is a dictionary
                
                ### Iterates through the elements of the p_t dictionary ###
                ### and updates the parameters and then the constraints/expressions ###
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
                    
            if x_t is not None:
                # Putting the x_t in the appropriate place #
                (n,ph) = self.model_data_dimen['c']
                #Assert Statement to Make sure Passed in a column vector
                assert x_t.shape == (n,1),"Error: You did not pass a column vector of correct dimensions"
                self.noisy_decision_dong = x_t #putting the noisy decision in an 
                                            #attribute that can be shared among the other methods
                                            
        elif self.alg_specification == "BMPS_online_GD":
            
            if p_t is not None: #we assume it is a dictionary
                assert type(p_t) is dict #making sure passed in a dictionary
                
                ### Updating X(p_t) in BMPS_subproblem ###
                # Iterating through the values in the dictionary #
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
                getattr(self.BMPS_subproblem,'c').reconstruct(self.c_t_BMPS) 
                
                #Deleting the objective function#
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
                    
                ### Doing some Basic Checks at the End the Update ###
                # They won't necessary everything, but all of these things
                # need to be true for the model to be working well
                for constr in self.BMPS_subproblem.component_objects(pyo.Constraint):
                    assert constr._constructed == True, "Error in constraint construction (body)"
                    for c in constr:
                        lb = constr[c].lower
                        ub = constr[c].upper
                        assert ((lb is not None) or (ub is not None)), "Error in constraint construction (LHS/RHS)"
                
                for obj in self.BMPS_subproblem.component_objects(pyo.Objective):
                    assert obj._constructed == True, "Error in construction of objective function"
                
            
            if x_t is not None:
                
                (n,ph) = self.model_data_dimen['c']
                assert x_t.shape == (n,1),"Error: You did not pass a column vector of corrected dimensions"
                self.x_t_BMPS = x_t #assume get a column vector
        
            
    def next_iteration(self,eta_factor=1,part_for_BMPS=1,epsilon_for_DCZ=(1e-4)):
        
        # METHOD DESCRIPTION: This method carries out another iteration of 
        # the algorithm with which the instance of the object has been working 
        
        if self.alg_specification is None:
            print("Error: Must initialize an online algorithm before using this method. ",\
                  "You will need to use .initialize_IO_method.")
            return 0
        elif self.alg_specification == "Dong_implicit_update":
            self.dong_iteration_num = self.dong_iteration_num + 1 #increasing the iteration count
            ### Step 1: Calculating Loss ###
            loss_this_iteration = self.loss_function(y=self.noisy_decision_dong,\
                                            theta=self.c_t_dong)
            self.losses_dong.append(loss_this_iteration)
            
            #### Step 2: Update Rule ####
            if loss_this_iteration < epsilon_for_DCZ:
                #This means that we do not update self.c_t_dong
                #We do want to append the c_t that we are staying with onto
                #the dictionary
                print("Loss was below epsilon_for_DCZ.  Skipping update rule model")
                self.c_t_dict_dong[self.dong_iteration_num] = self.c_t_dong
            else:
                print("Loss was above epsilon_for_DCZ.  Utilizing update rule model")
                eta = eta_factor*(1/(math.sqrt(self.dong_iteration_num)))
                
                self.c_t_dong = self.update_rule_optimization_model(y=self.noisy_decision_dong,\
                                                    theta=self.c_t_dong,eta_t=eta)
                
                self.c_t_dict_dong[self.dong_iteration_num] = self.c_t_dong
            
            #### Step 3: Calculate Batch Solution ####
            # Something to do IN THE FUTURE #
            #self.calculate_batch_sol(dimQ=self.model_data_dimen['Q'],dimc=self.model_data_dimen['c'],\
            #                         dimA=self.model_data_dimen['A'],\
            #                         dimD=self.model_data_dimen['D'])
            
            
        elif self.alg_specification == "BMPS_online_GD":
            if part_for_BMPS == 1: #part 1 for algorithm
                ### Step 0: Update Iteration Count ###
                self.BMPS_iteration_number = self.BMPS_iteration_number + 1
                
                ### Step 1: Project to F (stores the c_t in c_t_BMPS) ###
                self.project_to_F(dimc=self.model_data_dimen['c'],y_t=self.y_t_BMPS)
            
            elif part_for_BMPS == 2: #part 2 for algorithm
                ### Step 3 (with Step 2 being that parameters were updated) ###
                ### Solve the Subproblem X(p_t) ###
                self.solve_subproblem() #obtain xbar
                
                ### Step 4: Calculate Learning Rate ###
                
                if self.if_use_diam == 0: #if we are NOT using diameters
                    eta_calculated = (1/math.sqrt(self.BMPS_iteration_number))
                
                ### Step 5: Gradient Descent Step ###
                #receive_data will have put the x_t into x_t_BMPS
                self.y_t_BMPS = self.gradient_step(eta_t=eta_calculated,x_t=self.x_t_BMPS)
                
            else:
                print("Error: You can only enter 1 or 2 for the part_for_BMPS parameter")
                return
                
            
            
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
