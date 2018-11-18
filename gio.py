####Class for General Inverse Optimization (GIO) Models from Chan et al. Paper######
#This class implements the solutions methods presented in Chan et al. (2018) for 
#their general inverse optimization (GIO) models.  
#There are three GIO models (1) based on p-norm (2) based on absolute duality gap
#(3) based on relative duality gap
#We have implemented (1) thus far for finding the optimal epsilon

#The class assumes a forward (traditional linear constrained model) of 
#           min c'x
#           st Ax >= b
#Therefore, the GIO class takes as INPUT the A and b matrices as well as a 
#feasible x0 value
#The goal of the implemented algorithms is to (from Chan et al. (2018)):
#For the general GIO method: "Given x^0, GIO(x^0) identifies a direction of
#perturbation, epsilon*, of minimimal distance to bring x^0 into the set X^OPT"
#with X^OPT as the set of points on the boundary of the feasible region that
#have a corresponding c vector that would make these feasible points optimal
#in the forward problem.
 
#Chan et al. also provide methods for obtaining this c, which are fairly
#simple to implement given that we have implemented the epsilon methods.

#In the GIO_p(self,p) method, the user is asked to supply the p-norm he/she/they
#wish to use.  We restrict the choice of p to the 1, 2, or infinity norms.

###### Additional Helpful Documentation ######
##https://software.sandia.gov/downloads/pub/coopr/CooprGettingStarted.html

import pdb #for debugging
import numpy as np #numpy is working a little differently in terms of object orientedness
                    #since I cannot make the methods work directly on an instance of the np class,
                    #I have to pass in - and doesn't modify the instances directly; like does not
                    #change the objects I pass into the function
                    
                    #objects are like little units/parcels that store attributes and methods
                    
import pyomo.environ as pyo
from pyomo.opt import SolverFactory #page 43 of the Oct 2018 documentation
import pyomo.kernel as pyok #to differentiate from pyomo.environment
                            #if I want to work from the lower level functionality
                            #of pyomo, I need to work in the kernel

#When Pyomo does a solve, it puts the solution values into attributes within the 
                            #object (I believe)

#Doc Statement (Python book said we should include)
"A class to find the c and epsilon values under different General Inverse Optimization (GIO) models"

               
###########  Function to Find i* in order to Calculate epsilon*  ###########
###We define this outside of the class in order to make it a global function
#def i_star(A,b,x0,p):   #need to figure out how to define methods that are not accessible to the user
#    residuals = np.transpose( (np.matmul(A,x0) - b) ) #need to transpose
#    if np.any(residuals<0)==True: #none of the residuals should be less than 0 because x^0 is a feasible point; the code should break if there is a negative residual
#        print("Error: Negative Residual and Thus Infeasible x^0")
#        return #to get out of the function entirely
#        
#    if p=='inf':
#        row_norms = np.transpose(np.linalg.norm(A,ord=np.inf,axis=1)) #have a special specification for infinity norm
#    elif p=='b': #this is for the relative norm case where we divide by absolute b 
#        #which we will sub in as ``row norms'' here just to keep the notation the same
#        row_norms = np.transpose(np.absolute(b)) #THIS COULD have issues because not sure what np.absolute does exactly
#                                    #although b is a column vector (we define it as a column vector)
#                                    #but we transpose the row vector in
#    else:
#        row_norms = np.transpose(np.linalg.norm(A,ord=p,axis=1)) #need to transpose #this is row-wise #should pick an A that wouldn't work
#    #Inspired by: https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
#    
#    ratios = np.divide(residuals,row_norms) 
#    
#    ######## Calculating 
#    istar = np.argmin(ratios) #remember indexes from 0
#    
#    
#    min_ratio = ratios[0,istar] #this is a 2D array so need to index accordingly
#    
#    return istar,min_ratio


class GIO():
    #Doc Statement (Python book said we should include)
    "A class to find the c and epsilon values under different General Inverse Optimization (GIO) models"
    
    def __init__(self,A,b,x0):  #start out with A, b, and x^0 (Chan et al. say they are "exogenously determined")
        #Inputs: A,b,x0 (details below)
        #We assume A is a matrix, b is a column vector, and x0 is a column vector
        #We also (for now) assume that they are numpy matrices/arrays
        
        #Attributes: We generate A, b, and x0 attributes as well as a bunch of lists in which
        #to put things for methods later on
        self.A = A
        self.b = b
        self.x0 = x0 #loading the data into attributes
        self.epsilon_p = []
        self.epsilon_a = []
        self.epsilon_r = [] #ep^* for the various GIO models; preallocating lists in case we end up appending
        self.x0_epsilon_p = []
        self.x0_epsilon_a = []
        self.x0_epsilon_r = [] #(x^0 - ep^*) for the various GIO models; preallocating lists in case we end up appending
        self.istar_multi = [] #storage for in case we project to multiple hyperplanes
        self.rho_p = []
        self.rho_a = []
        self.rho_r = [] #rho exact under the various models
                     
    def change_x0(self,new_x0): 
        #Method to change x0 (even though we can do with dot notation)
        #Input: new_x0
        #Output: None, assignment to x0 attribute
        self.x0 = new_x0
        
        
    def i_star(self,A,b,x0,q):   
        ####This method finds the minimum distance projection of x^0 onto the hyperplanes that
        ##define the boundary of the feasible region.
        ##INPUT: A (mxn numpy matrix), b (mx1 numpy column vector), x0 (nx1 numpy column vector), 
        ##q as the dual norm
        
        ##OUTPUT: istar (the constraint index where the distance between the projection and x0 is minimized),
        ## min ratio (the value associated with istar)
        ## IF there are multiple istar, a message is printed to the user, the first istar is put in the 
        ## istar variable, and the rest are written to the multi_istar attribute of the object
        
        ####Not a method users of the code need to worry about much####
        
        residuals = np.transpose( (np.matmul(A,x0) - b) ) #need to transpose
        if np.any(residuals<0)==True: #none of the residuals should be less than 0 because x^0 is a feasible point; the code should break if there is a negative residual
            print("Error: Negative Residual and Thus Infeasible x^0")
            return #to get out of the function entirely
            
        if q=='inf':
            row_norms = np.transpose(np.linalg.norm(A,ord=np.inf,axis=1)) #have a special specification for infinity norm
        elif q=='b': #this is for the relative norm case where we divide by absolute b 
            #which we will sub in as ``row norms'' here just to keep the notation the same
            row_norms = np.transpose(np.absolute(b)) #THIS COULD have issues because not sure what np.absolute does exactly
                                        #although b is a column vector (we define it as a column vector)
                                        #but we transpose the row vector in
        else:
            row_norms = np.transpose(np.linalg.norm(A,ord=q,axis=1)) #need to transpose #this is row-wise #should pick an A that wouldn't work
        #Inspired by: https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        
        ratios = np.divide(residuals,row_norms) 
        ratios = np.reshape(ratios,(np.size(ratios),))
        
        ######## Finding istar ########### 
        
        (istar,) = np.where(ratios == ratios.min()) #remember indexes from 0
        #The istar being returned is an array, whether a 1D or not
        #pdb.set_trace()
        #Additional resource for where: https://www.geeksforgeeks.org/numpy-where-in-python/
        #https://stackoverflow.com/questions/18582178/how-to-return-all-the-minimum-indices-in-numpy
        
        if np.size(istar) > 1:
            print("Under the",q,"loss metric, x^0 has been projected onto multiple",\
                  "hyperplanes.  For now, we will choose the first i index",\
                  "and will put the rest of the indices in the istar_multi",\
                  "attribute.")
            self.istar_multi.append(istar) #incase there have been things run already
            istar = istar[0] #picking the istar 
        else:
            istar = istar[0] #because istar is coming back as an array when we
                            #want it to be an index
        
        min_ratio = ratios[istar] #this is a 2D array so need to index accordingly
        
        return istar,min_ratio
    
    
    def GIO_p(self,p,if_append): 
        "A method that computes GIO according to the p-norm"
        "Input: The p indicates the norm (we only support 1,2, or infinity norms)"
        """The if_append argument indicates whether or not we want to replace any epsilon 
        and x0-epsilon values already generated or if we want to append the values onto a growing list.
        It allows us to store multiple values of GIO_p under different norms.  Takes a value of 'T' or 'F' 
        although wont actually append unless you put 'T' """
        
        if isinstance(p,str)==True: #making sure that if type in 'infinity', then gets converted
                                    #made use of: https://stackoverflow.com/questions/152580/whats-the-canonical-way-to-check-for-type-in-python
                                    #https://www.programiz.com/python-programming/methods/built-in/isinstance
            p = 'inf' 
        
        self.p = p #DO I WANT THIS? #MIGHT NEED TO ADD IN STUFF ABOUT FORCING 1,2, AND INFINITY - only those options

        ##########    Computing the Epsilon*    ########
        ##  Determining the Dual Norm for the Chosen p-Norm   ##
        if self.p==1:
            q = 'inf'
        elif self.p == 2:
            q = 2
        else: #presumably self.p = 'inf'
            q = 1
        
        [istar,min_ratio] = self.i_star(self.A,self.b,self.x0,q) #finding i*; want to pass
                                                            #a copy of the data into istar 
                                                            #to prevent istar from changing 
                                                            #the data
        
        ##  Actually Computing Epsilon* for Each p-Norm  ##
        if self.p == 1:
            A_row = self.A[istar,:] #grabbing the row corresponding to istar
            jstar = np.argmax(np.absolute(A_row)) #to find j*
            epsilon_constant = np.sign(A_row[jstar])*min_ratio #constant by which to multiply e_j*
            #For standard basis: https://stackoverflow.com/questions/11975146/efficient-standard-basis-vector-with-numpy
            #says np.zeros seems to be the fastest
            unitvector = np.zeros((np.size(A_row),1))  #should be the size of the row since this is number of DVs/vars          
            unitvector[jstar,0] = 1 #to make a unit vector
            epsilon = epsilon_constant*unitvector 
            
            ## Check for if column vs row vector ##
            (dim1,dim2) = np.shape(epsilon)
            if dim1 < dim2: 
                epsilon = np.transpose(epsilon) 
            
        elif self.p == 2:
            A_row = self.A[istar,:]
            A_unit_row = (1/np.linalg.norm(A_row))*A_row
            epsilon = min_ratio*A_unit_row #following the formula
            
            ep_size = np.size(epsilon)
            epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
            
            ## Check for if column vs row vector ##
            (dim1,dim2) = np.shape(epsilon)
            if dim1 < dim2: 
                epsilon = np.transpose(epsilon) 
            
        else: #presumably self.p = 'inf'
            A_row = self.A[istar,:]
            epsilon = min_ratio*np.sign(A_row)
            
            ep_size = np.size(epsilon)
            epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
            
            ## Check for if column vs row vector ##
            (dim1,dim2) = np.shape(epsilon)
            if dim1 < dim2: 
                epsilon = np.transpose(epsilon) 
        
        #######  Storing the Epsilon* as an Attribute  #######
        if if_append == 'T':
            self.epsilon_p.append(epsilon)
            self.x0_epsilon_p.append(self.x0 - epsilon)
        else:
            self.epsilon_p = [epsilon] 
            self.x0_epsilon_p = [self.x0 - epsilon] #since the method is a function, it has local variables
                                                #BUT we can access global variables now with the instances
              
    def GIO_abs_duality(self):
        ##Method that finds epsilon* according to the Absolute Duality Gap GIO Model##
        
        #### Gives the Exact Same Closed form Solution as the Infinity Norm ####
        [istar,min_ratio] = self.i_star(self.A,self.b,self.x0,1) #since dual of infinity is 1, and i* takes the dual norm
        A_row = self.A[istar,:]
        epsilon = min_ratio*np.sign(A_row)
            
        ep_size = np.size(epsilon)
        epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
        
        ## Check for if column vs row vector ##
        (dim1,dim2) = np.shape(epsilon)
        if dim1 < dim2: 
            epsilon = np.transpose(epsilon)
        
        self.epsilon_a = [epsilon] #epsilon_a for absolute duality gap
        self.x0_epsilon_a = [self.x0 - epsilon]
        #print("x_0-ep for absolute",self.x0 - epsilon)
        #print("x0_epsilon_a:",self.x0_epsilon_a)
    
    def GIO_relative_duality(self):
        ##Method that finds epsilon* according to the Relative Duality Gap GIO Model##
        [istar,min_ratio] = self.i_star(self.A,self.b,self.x0,'b')
        A_row = self.A[istar,:]
        epsilon = (1/np.linalg.norm(A_row,ord=1))*((np.dot(A_row,self.x0) - self.b[istar,0])*np.sign(A_row))
            
        ep_size = np.size(epsilon) #need to see if need this
        epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
                        
        ## Check for if column vs row vector ##
        (dim1,dim2) = np.shape(epsilon)
        if dim1 < dim2: 
            epsilon = np.transpose(epsilon)
        
        ### Storing Epsilon and x0-epsilon in Attributes ###
        self.epsilon_r = [epsilon]
        self.x0_epsilon_r = [self.x0 - epsilon]
    
    def GIO_all_measures(self):
        """This function runs all of the GIO models, including GIO_p
        for the p=1, 2, and inf norms"""
        self.GIO_p(1,'T')
        self.GIO_p(2,'T')
        self.GIO_p('inf','T')        
        self.GIO_abs_duality()
        self.GIO_relative_duality()
        
        #print("sup placeholder")
        #want to define a function that produces all of the measures
        #essentially is calling all of the other methods
        #will need to check that all of the attributes get filled
        
        #going to try to append onto x0_epsilon_p and epsilon_p
        #iterate through a list of the 1,2, and infinity norms
    
            
    def calculate_rho_p(self,p,if_append):
        ### This function calculates the goodness-of-fit/coefficient of complementarity metric
        ## under the p-norm 
        
        ##INPUT: p (for p norm) and if_append ('T' if you want the rho_p to be appended onto the list, otherwise
        ##we will just overwrite the rho_p list with a new list)
        
        ##OUTPUT: none, just writing to the rho_p attribute
        
        if isinstance(p,str)==True:
            p = 'inf' #making sure that, if a string was passed in for p, then 
                    #we are finding the infinity norm
        
        ######  Assume for now that rho_p overwrites stuff  ######
        self.GIO_p(p,'F') #'F' because we want to overwrite the stuff from before; 
                            #do need to remember that we will need to index into the list
        #pdb.set_trace()
        epsilon_star = self.epsilon_p
        epsilon_star = np.linalg.norm(epsilon_star[0],ord=p) #because it is in a list, need to index into the list
        
        ##### Need to Account for when we are too close to the boarder of the Feasible Region #####
        if epsilon_star < 1e-8: #Need to exit the function 
            if if_append == 'T':
                self.rho_p.append(1) #rho would be 1 because if exactly on the boundary then you
                                    #are in X^{OPT}, the ultimate sign of 'fit'
            else:
                self.rho_p = [1]
            return 
        
        #####  Calculating the Convex Programs #####
        
        #if p==2: #probably actually just need to specify this for the objective function
        A = self.A #copying to local variable because the local object "self" calls could get weird
        b = self.b
        x0 = self.x0
        (dim1,dim2) = np.shape(A) #getting the dimensions for the variable sets
        convex_prog = pyo.ConcreteModel()
        convex_prog.numvars = pyo.Param(initialize=dim2)
        convex_prog.varindex = pyo.RangeSet(1,dim2)
        convex_prog.Arowindex = pyo.RangeSet(1,dim1)
        convex_prog.ep = pyo.Var(convex_prog.varindex)
        
        ##### Feasibility Constraints #####
        def feas_rule(self,i): #we are assuming that b is a column vector with functioning 2 axes
                                #HOPEFULLY THE TWO SELVES ARE NOT CONFUSING??
                                #We are generating a local object, so hopefully we are all good
            return sum( A[i-1,j-1]*(x0[j-1,0]-self.ep[j]) for j in range(1,self.numvars+1)) >= b[i-1,0] #need to be careful
                                                                            #about mixing up the Pyomo vs Python indexing
        convex_prog.feas_constraint = pyo.Constraint(convex_prog.Arowindex,rule=feas_rule)
        
        ##### The ith constraint (equality) #####
        def equal_rule(self,index): #need to figure out how we are indexing (Pyomo or Python)
            return sum( A[index-1,j-1]*(x0[j-1,0]-self.ep[j]) for j in range(1,self.numvars+1) ) == b[index-1,0]
        #MIGHT NEED TO DOUBLE CHECK THESE INDICES
        
        convex_prog.equal_constraint = pyo.Constraint(convex_prog.Arowindex,rule=equal_rule) #assuming that we are Pyomo indexing
        convex_prog.equal_constraint.deactivate() # we will only activate the equality constraint
                                                #that is relevant to the given solve
        
        ##### Objective Function & Solution #####
        ##This will depend upon the norm we are imposing.##
        if p==2:
            def obj_rule_p2(model): #worked with "self" as well
                return pyo.sqrt( sum(model.ep[j]**2 for j in range(1,model.numvars+1)) )
            convex_prog.obj = pyo.Objective(rule=obj_rule_p2)
            solver = SolverFactory('ipopt') #need nonlinear because of the 
                                    #objective function
            
        ############ Solving the Convex Progs #############
        #This help ticket indicated that we needed to give the interior point method a starting point (and indicated 0 was a bad one)
        #https://projects.coin-or.org/Ipopt/ticket/205
            sum_of_epsilons = 0
            #solver = SolverFactory('ipopt')
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                for i in range(1,convex_prog.numvars+1):
                    convex_prog.ep[i] = 0.01 #have to give interior point algorithm a non-zero starting place
                
                convex_prog.equal_constraint[index].activate() #activating relevant equality constraint
                solver.solve(convex_prog)
                #print("Objective function value is",pyo.value(convex_prog.obj))
                sum_of_epsilons = sum_of_epsilons + pyo.value(convex_prog.obj)
                convex_prog.equal_constraint[index].deactivate()
        elif p==1:
            ###Since there are absolute values in the objective function for the L1 norm, we need to do a 
            ##transformation that will actually linearize the problem.
            #See documentation/chapter for references on this
            convex_prog.u = pyo.Var(convex_prog.varindex) #variables to replace x in the objective func
            
            ##Two rules to establish the extra constraints
            ##ConstraintList didn't like the convex_prog.varindex, hence
            ##the two rules
            def extra_abs_val_constraints_part1(model,i):
                return model.ep[i] <= model.u[i]
            
            def extra_abs_val_constraints_part2(model,i):
                return -1*model.ep[i] <= model.u[i]
            
            convex_prog.abs_val_constraints_part1 = pyo.Constraint(\
                                convex_prog.varindex,\
                                rule=extra_abs_val_constraints_part1)
            convex_prog.abs_val_constraints_part2 = pyo.Constraint(\
                                convex_prog.varindex,\
                                rule=extra_abs_val_constraints_part2)
            
            ###Defining the Objective Function###
            def obj_rule_p1(model):
                return sum(model.u[j] for j in range(1,model.numvars+1))
            convex_prog.obj = pyo.Objective(rule=obj_rule_p1)
            solver = SolverFactory('glpk') #we have a linear program now
                            #due to the transformation, so we can use an LP solver
            
            #pdb.set_trace() #Looks like everything is good
        ############ Solving the Convex Progs #############
        #This help ticket indicated that we needed to give the interior point method a starting point (and indicated 0 was a bad one)
        #https://projects.coin-or.org/Ipopt/ticket/205
            sum_of_epsilons = 0
            #solver = SolverFactory('ipopt')
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                for i in range(1,convex_prog.numvars+1):
                    convex_prog.u[i] = 0.01 #have to give interior point algorithm a non-zero starting place
                
                convex_prog.equal_constraint[index].activate() #activating relevant equality constraint
                solver.solve(convex_prog)
                #print("Objective function value is",pyo.value(convex_prog.obj))
                sum_of_epsilons = sum_of_epsilons + pyo.value(convex_prog.obj)
                convex_prog.equal_constraint[index].deactivate()
            
        
        ##### Calculating the Rho #####
        average_of_epsilons = (1/dim1)*(sum_of_epsilons) 
        rho = 1 - (epsilon_star/average_of_epsilons)
        
        if if_append == 'T':
            self.rho_p.append(rho)
        else:
            self.rho_p = [rho]
            
            
                   
    ###################### To be continued methods/functions #################################    
    def calculate_c(self):
        print("Actually calculates the c vector.  Shouldnt be hard, just need to do it")
    
    def rho_p_approx(self,p):
        print("Calculating approx rho_p")
    
    
    def GIO_linear_solve(self,type_c_constraint,type_ep_constraint):
        print("This function will serve the purpose of producing epsilon values",\
              "when we want to be able to put constraints upon c and epsilon.",\
              "We will implement a few options for constraints that can be placed on",\
              "c and epsilon that the user can pass in as options.",\
              "We will be using the absolute duality gap model")
        #Useful for when the closed form solution breaks down
        #Will need to generate an actual Pyomo model and do a linear solve.  
        #OR if users pass in a Pyomo model, then will need to figure out what to do
        
    def change_projection(self):
        print("This function will allow users to project onto a different",
              "hyperplane if there are multiple istar and the user wants to see another.",\
              "It will require that multi_istar be nonempty and the user would need to",\
              "specify from which GIO model the multi istar model came so that we can",\
              "update the appropriate attributes. Would need to call calculate_c")

        
        
            
            
        
                    
        
        
        
        
        
        
        
        
        
        
        
        
        