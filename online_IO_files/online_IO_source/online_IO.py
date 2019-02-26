#### Class for Online Inverse Optimization Methods ####
##These methods come from Dong et al. 2018 and B\"armann et al. 2018



import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity

class Online_IO(): #SHOULD WE RENAME THE FILE SLIGHTLY
    
    def __init__(self,initial_model,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
                 dimQ=(1,1),dimc=(1,1),dimA=(1,1),dimD=(1,1)):
        #TO DO: (1) Might want have the initialization of the names and dimensions
        #as a separate method?  Want to ask Tom's advice
        # (2) Fix the fact that right now you have to define a "dummy_name"
        # parameter in your model
        
        #FOR NOW we are going with the pyomo model implementation.
        #(Could always utilize numpy to create the parameters)
        #There is some really awesome functionality in numpy that would be 
        #awesome to utilize if possible.
        
        #So I can put a copy of the model into an attribute and then index
        #into it with different names, but I 
        
        self.initial_model = initial_model #load initial pyomo model into attribute
        self.obj_vals = {} #going to initialize as a dictionary to make easier to handle
                            #different types of objective functions
        self.alg_specification = None #not initialized until call "initialize_IO_method"
                                        #by setting as None, can use "is __ none" method
        self.model_data_names = {'Q':Qname,'c':cname,'A':Aname,'b':bname,'D':Dname,'f':fname}
        self.model_data_dimen = {'Q':dimQ,'c':dimc,'A':dimA,'D':dimD}
        
        ##Dong_implicit_update Attributes##
        self.KKT_conditions_model = None
        self.loss_model_dong = None
    
    
    def initialize_IO_method(self,alg_specification):
        ### Each algorithm has its own set up procedure before the first iteration
        if alg_specification=="Dong_implicit_update":
            ##FOR NOW assuming that we are passing in a Pyomo model
            
            self.alg_specification = alg_specification
            #We can use getattr() to pass in the parameter values into the 
            #compute_KKT_conditions function
            #Thank you to:
            #https://stackoverflow.com/questions/2612610/how-to-access-object-attribute-given-string-corresponding-to-name-of-that-attrib
            #https://docs.python.org/2/library/functions.html#getattr
            self.compute_KKT_conditions(getattr(self.initial_model,self.model_data_names['Q']),\
                getattr(self.initial_model,self.model_data_names['c']),\
                getattr(self.initial_model,self.model_data_names['A']),\
                getattr(self.initial_model,self.model_data_names['b']),\
                getattr(self.initial_model,self.model_data_names['D']),\
                getattr(self.initial_model,self.model_data_names['f']),\
                dimQ=self.model_data_dimen['Q'],dimc=self.model_data_dimen['c'],\
                dimA=self.model_data_dimen['A'],dimD=self.model_data_dimen['D'],\
                use_numpy=0) #we are passing in parameter objects into compute_KKT_conditions
        
        else:
            print("Error, we do not support inputted method. (It is possible",\
                    "that you spelled something wrong.)")
            return
        
            
    
    def receive_data(self,p_t,x_t,alg_specification):
        #The algorithm specification might get set by the initialization.
        #In the initialize_IO_method, might want to initialize an attribute
        #for the object so that we can here say alg_specification=self.SOMETHING
        
        #We will want this method to throw an error if people try to use it before the 
        #initialize_IO_method - that is where having "None" in the alg_specification
        #attribute is important
        print("working on")
        
        
    def compute_KKT_conditions(self,Q,c,A,b,D,f,dimQ=(1,1),dimc=(1,1),dimA=(1,1),\
                               dimD=(1,1),use_numpy=0,bigM=10000):
        
        #TO DO: 
        # (1) Need to deal with non-negativity constraints upon the x
        # (or we could force ppl to put these in Ax <= b.... but probably shouldnt...)
        # (2) Will have to deal with mutability stuff at some point...(just in general
        # somehow telling the class where the mutable data is will be the problem)
        #       (a) One idea might be to actually assign the parameter objects into
        #       the KKT_model pyomo model -> like KKT_model.A = A (need to see if can do)
        #       This MIGHT allow their already predefined mutability to come with them
        #       (since I'm making users define mutability before using my code)
        #       Will still need to get users to ID which param name is mutable
        #       In order for me to be able to update...
        #       Can test if all of this works by writing a new method to update!
        # (3) Might want to not define variables that we don't need to (us and vs)
        # (4) More cases for Stationarity conditions
        
        ##We are going to assume that the Pyomo model is of the following form:
        ## min f(x) = (1/2) x^t Q x + c^T x
        ## st Ax <= b
        ## Dx = f
        #We assume c,b,f are vectors and Q,A,D are matrices
        
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
            
            ## Establishing the KKT Stationarity Stuff ##
            #Determining the Stationarity Rule to Use Based upon Existance of Data
            #We will assume that c vector is always present (for now) but all
            #others will be up for grabs
            if n2==0 and m>0 and p>0:
                def KKT_stationarity_rule(model,i): #no Q
                    return c[i]\
                    + sum(A[j,i]*model.u[j] for j in range(1,m+1)) \
                    + sum(D[j,i]*model.v[j] for j in range(1,p+1)) == 0
                    
            elif n2==0 and m>0 and p==0: #no Q, no D
                def KKT_stationarity_rule(model,i):
                    return c[i] + sum(A[j,i]*model.u[j] for j in range(1,m+1)) == 0
                
            elif n2>0 and m>0 and p>0: #have all the data
                def KKT_stationarity_rule(model,i):
                    return sum(Q[i,j]*model.x[j] for j in range(1,n+1)) + c[i]\
                    + sum(A[j,i]*model.u[j] for j in range(1,m+1)) \
                    + sum(D[j,i]*model.v[j] for j in range(1,p+1)) == 0
                    
            elif n2==0 and m==0 and p>0: #no Q no A
                def KKT_stationarity_rule(model,i):
                    return c[i] + sum(D[j,i]*model.v[j] for j in range(1,p+1)) == 0
            
            KKT_model.stationary_conditions = pyo.Constraint(KKT_model.xindex,rule=KKT_stationarity_rule)
            
            ##### Step 2: Complementarity Constraints #####
            #We are going to do the disjunctive method ourselves
            if m>0:
                KKT_model.z = pyo.Var(KKT_model.uindex,domain=pyo.Binary) #same number of z vars as u vars
                
                ## Maintaining Feasibility of Ax <= b ##
                def Ax_b_feasibility_rule(model,i):
                    return sum(A[i,j]*model.x[j] for j in range(1,n+1)) <= b[i]
                
                KKT_model.Ax_b_feas = pyo.Constraint(KKT_model.uindex,\
                                                     rule=Ax_b_feasibility_rule)
                
                ## Disjunctive Constraints to Ensure Corresponding Rows of 
                ## u == 0 OR Ax == b
                def u_disjunctive(model,i):
                    return model.u[i] <= bigM*model.z[i]
                
                KKT_model.u_disjunc = pyo.Constraint(KKT_model.uindex,rule=u_disjunctive)
                
                def Ax_b_disjunctive(model,i):
                    return b[i] - sum(A[i,j]*model.x[j] for j in range(1,n+1)) <= bigM*(1-model.z[i])
                
                KKT_model.Ax_b_disjunc = pyo.Constraint(KKT_model.uindex,rule=Ax_b_disjunctive)
            
            ##### Step 3: Equality Constraints #####
            if p>0:
                def KKT_equality_constraints(model,i):
                    return sum(D[i,j]*model.x[j] for j in range(1,n+1)) == f[i]
                
                KKT_model.equality_conditions = pyo.Constraint(KKT_model.xindex,\
                                                    rule=KKT_equality_constraints)            
    
        elif use_numpy == 1: #means we are using numpy objects
            print("we will see about this one...")
            
        else:
            print("Error: This argument can only be 0 or 1")
            return
        
        
        #### Putting Model into Instance Attribute ####
        self.KKT_conditions_model = KKT_model #putting model into attribute
        
    def loss_function(self,y,u=0,theta=0,if_solve=1):
        #TO DO: 
        # (1) **Add in the ability to pass in theta and u (which is a BIG need)
        # This function is not at all doing its job until we can do this^
        # (a) Probably will be part of a larger class effort to deal with 
        # mutable data and the (signal,decision) pairs from Dong et al. and 
        # B\"armann et al.
        
        #This method constructs the l(y,u,theta) loss function defined in 
        #Dong et al.
        #In order to run this method, would need to run the 
        #initialize method first. 
        
        #We assume that y has the same dimensions as x, and we also assume that
        #y is inputted as a numpy column vector.
        
        loss_model = self.KKT_conditions_model #copying over the KKT_conditions_model
        
        ##### Step 1: Add in the Objective Function #####
        (n,ph) = np.shape(y) #getting dimensions of y
        
        def loss_objective_rule(model):
            return sum((y[j-1,0] - model.x[j])**2 for j in range(1,n+1))
        
        loss_model.obj_func = pyo.Objective(rule=loss_objective_rule)
        
        
        ##### Step 2: Solve the loss_model #####
        solver = SolverFactory("gurobi") #right solver to use because of 
                                        #nonlinear objective function and the 
                                        #possible binary variables
        
        results = solver.solve(loss_model,tee=True)
        print("This is the termination condition:",results.solver.termination_condition)
        
        ##Putting Model in Attribute##
        self.loss_model_dong = loss_model
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
