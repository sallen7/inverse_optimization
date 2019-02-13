####Class for General Inverse Optimization (GIO) Models from Chan et al. Paper######
#This class implements the solutions methods presented in Chan et al. (2018) for 
#their general inverse optimization (GIO) models.  
#There are three GIO models (1) based on p-norm (2) based on absolute duality gap
#(3) based on relative duality gap

#The class provides methods for 
#(1) solving these flavors of GIO models 
#(2) producing "goodness-of-fit"/coefficient of complementarity values
#for the three flavors
#(3) solving the structural constraints upon epsilon model

#The class assumes a forward (traditional linear constrained model) of 
#           min c'x
#           st Ax >= b
#Therefore, the GIO class takes as INPUT the A and b as well as a 
#feasible x0 value
 
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition


class GIO():
    #Doc Statement
    "A class to perform linear inverse optimization analysis under different General Inverse Optimization (GIO) models"
    
    def __init__(self,Amat,bvec,x0,num_eqs=None,num_vars=None,if_pyomo_params='F'):  
        
#        Inputs: A,b,x0 (details below)
#        
#        We assume A is a numpy matrix, b is a numpy column vector, and 
#        x0 is a numpy column vector UNLESS
#        the if_pyomo_params argument is set to 'T' (which indicates that the data being
#        passed into the initialization step are pyomo parameter components)
#        
#        Either way, we assume x0 is a numpy array
#        
#        IF this flag is set to 'T', then the user MUST provide num_eqs and num_vars
#        parameter values
        
        
        if if_pyomo_params == 'T':              
            Anumpy = np.zeros((num_eqs,num_vars))
            bnumpy = np.zeros((num_eqs,1))
            ### We are passing in a pyomo.core.base.param.IndexedParam (and it does work!)
            ### This means we will have to do some indexing manuvering
            for i in range(1,num_eqs+1):  
                bnumpy[i-1,0] = bvec[i] 
                for j in range(1,num_vars+1):
                    #pdb.set_trace()
                    Anumpy[i-1,j-1] = Amat[i,j]
            
            ### Assigning to Attributes ###
            self.A = Anumpy
            self.b = bnumpy            
        else: #if we are passing in numpy arrays
            self.A = Amat
            self.b = bvec       
            
        #Attributes: We generate A, b, and x0 attributes as well as a bunch of lists in which
        #to put things for methods later on
        self.x0 = x0 #loading the data into attributes
        self.epsilon_p = []
        self.epsilon_a = []
        self.epsilon_r = [] #ep^* for the various GIO models; preallocating lists in case we end up appending
        self.x0_epsilon_p = []
        self.x0_epsilon_a = []
        self.x0_epsilon_r = [] #(x^0 - ep^*) for the various GIO models; preallocating lists in case we end up appending
        self.istar_multi = [] #storage for in case we project to multiple hyperplanes
        self.c_p = []
        self.c_a = []
        self.c_r = [] #lists for storing the c cost vectors
        self.rho_p = []
        self.rho_a = []
        self.rho_r = [] #rho exact under the various models
        self.rho_p_approx = [] #rho approximate for the p norm GIO model 
        self.GIO_struc_ep = 0
        
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
        #Additional resource for where: https://www.geeksforgeeks.org/numpy-where-in-python/
        #https://stackoverflow.com/questions/18582178/how-to-return-all-the-minimum-indices-in-numpy
        
        if np.size(istar) > 1:
            print("Under the",q,"dual norm, x^0 has been projected onto multiple",\
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
    
    
    def GIO_p(self,p,if_append='F'): 
        
        #A method that computes GIO according to the p-norm
        #INPUT: The p indicates the norm (we only support 1,2, or infinity norms)
        #The if_append argument indicates whether or not we want to replace any epsilon 
        #and x0-epsilon values already generated or if we want to append the values onto a growing list.
        #It allows us to store multiple values of GIO_p under different norms.  Takes a value of 'T' or 'F' 
        #although wont actually append unless you put 'T' 
        
        
        if isinstance(p,str)==True: #making sure that if type in 'infinity', then gets converted
                                    #made use of: https://stackoverflow.com/questions/152580/whats-the-canonical-way-to-check-for-type-in-python
                                    #https://www.programiz.com/python-programming/methods/built-in/isinstance
            p = 'inf' 

        ##########    Computing the Epsilon*    ########
        ##  Determining the Dual Norm for the Chosen p-Norm   ##
        if p==1:
            q = 'inf'
        elif p == 2:
            q = 2
        else: #presumably p = 'inf'
            q = 1
        
        [istar,min_ratio] = self.i_star(self.A,self.b,self.x0,q) #finding i*; want to pass
                                                            #a copy of the data into istar 
                                                            #to prevent istar from changing 
                                                            #the data
        
        ##  Actually Computing Epsilon* for Each p-Norm  ##
        if p == 1:
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
            
        elif p == 2:
            A_row = self.A[istar,:]
            A_unit_row = (1/np.linalg.norm(A_row))*A_row
            epsilon = min_ratio*A_unit_row #following the formula
            
            ep_size = np.size(epsilon)
            epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
            
            ## Check for if column vs row vector ##
            (dim1,dim2) = np.shape(epsilon)
            if dim1 < dim2: 
                epsilon = np.transpose(epsilon) 
            
        else: #presumably p = 'inf'
            A_row = self.A[istar,:]
            epsilon = min_ratio*np.sign(A_row)
            
            ep_size = np.size(epsilon)
            epsilon = np.reshape(epsilon,(ep_size,1)) #turning from 1D to 2D vector
            
            ## Check for if column vs row vector ##
            (dim1,dim2) = np.shape(epsilon)
            if dim1 < dim2: 
                epsilon = np.transpose(epsilon) 
        
        #######  Storing the Epsilon* as an Attribute &  #######
        #######         Calculating the c vector         #######
        
        if if_append == 'T':
            self.epsilon_p.append(epsilon)
            self.x0_epsilon_p.append(self.x0 - epsilon)
            self.calculate_c_vector(istar,'p',if_append) #calculating c vector
        else:
            self.epsilon_p = [epsilon] 
            self.x0_epsilon_p = [self.x0 - epsilon] #since the method is a function, it has local variables
                                                #BUT we can access global variables now with the instances
            self.calculate_c_vector(istar,'p',if_append) #calculating c vector
              
    def GIO_abs_duality(self):
        
        ##Method that finds epsilon* according to the Absolute Duality Gap GIO Model##
        #Inputs: None
        
        #Purpose: Implements the $\ep^*$ and $\c^*$ components of Theorem 
        #\ref{chan_closed_form_theorem} for the GIO$_a(x^0)$ model %and thus 
        #finds $\ep^*_p$ and the corresponding $\c^*$
        
        #Results/Output: Places the resulting computed 
        #\textit{multi-dimensional} $\ep^*_a$ into the \url{epsilon_a} 
        #attribute, the computed $\x^0 - \ep^*_a$ into the \url{x0_epsilon_a} 
        #attribute, and the $\c^*$ in the \url{c_a} attribute
        
        
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
        
        ### Calculating the c vector ###
        self.calculate_c_vector(istar,'a','F')
    
    def GIO_relative_duality(self):
        
        ##Method that finds epsilon* according to the Relative Duality Gap 
        #GIO Model##
        #Inputs: None
    
        #Purpose: Implements the $\ep^*$ and $\c^*$ components of Theorem 
        #\ref{chan_closed_form_theorem} for the GIO$_r(x^0)$ model 
    
        #Results/Output: Places the resulting computed \textit{multi-dimensional}
        #$\ep^*_r$ into the \url{epsilon_r} attribute, the computed 
        #$\x^0 - \ep^*_r$ into the \url{x0_epsilon_r} attribute, and 
        #the $\c^*$ in the \url{c_r} attribute
        
        
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
        
        ### Calculating the c vector for the istar ###
        #pdb.set_trace()
        self.calculate_c_vector(istar,'r','F')
    
    def GIO_all_measures(self):
        
        #This function runs all of the GIO models, including GIO_p
        #for the p=1, 2, and inf norms
        
        #Inputs: None
    
        #Purpose: Calls \url{GIO_p} for $p=1,2,\infty$ (using 
        #\url{if_append=`T'} so that all of the values appear in the 
        #attributes), \url{GIO_abs_duality()}, and \url{GIO_relative_duality()}.  Thus one method call generates a complete panel of GIO model solves.
    
        #Results/Output: Generates all of the output described above 
        #for the individual methods
        
        self.GIO_p(1,'T')
        self.GIO_p(2,'T')
        self.GIO_p('inf','T')        
        self.GIO_abs_duality()
        self.GIO_relative_duality()
    
            
    def calculate_rho_p(self,p,if_append='F'):
        
        #Inputs: \url{p} specifies the type of norm (only $1,2,\infty$ supported)
        #and \url{if_append} indicates whether or not you would like the 
        #computed $\rho_p$ to be appended onto the current list in the 
        #\url{rho_p} attribute.  It is set to \url{`F'} by default, which 
        #means that the \url{rho_p} attribute is replaced by a list containing 
        #the $\rho_p$ calculation.  Set to \url{`T'} if you would like to append.
        
        #Purpose: Implements the calculation of $\rho_p$ according to 
        #the $p$ norm specified using the data in the \url{A}, \url{b}, 
        #and \url{x0} attributes
        
        #Results/Output: Places the resulting computed $\rho_p$ into the 
        #\url{rho_p} attribute (which is a list) according to the 
        #\url{if_append} parameter.  Does not return anything to the user
        
        
        if isinstance(p,str)==True:
            p = 'inf' #making sure that, if a string was passed in for p, then 
                    #we are finding the infinity norm
        
        ######  Assume for now that rho_p overwrites stuff  ######
        self.GIO_p(p,'F') #'F' because we want to overwrite the stuff from before; 
                            #do need to remember that we will need to index into the list
        if p=='inf':
            epsilon_star = self.epsilon_p
            epsilon_star = np.linalg.norm(epsilon_star[0],ord=np.inf) #because it is in a list, need to index into the list            
        else:
            epsilon_star = self.epsilon_p
            epsilon_star = np.linalg.norm(epsilon_star[0],ord=p) #because it is in a list, need to index into the list
        
        ##### Need to Account for when we are too close to the boarder of the Feasible Region #####
        if epsilon_star < 1e-12: #Need to exit the function 
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
            
            
            ############ Solving the Convex Progs #############
            #This help ticket indicated that we needed to give the interior point method a starting point (and indicated 0 was a bad one)
            #https://projects.coin-or.org/Ipopt/ticket/205
            sum_of_epsilons = 0
            #solver = SolverFactory('ipopt')
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                convex_prog.equal_constraint[index].activate() #activating relevant equality constraint
                solver.solve(convex_prog)
                #print("Objective function value is",pyo.value(convex_prog.obj))
                sum_of_epsilons = sum_of_epsilons + pyo.value(convex_prog.obj)
                convex_prog.equal_constraint[index].deactivate()
        
        elif p=='inf': #we have the infinity norm
            #### Similar to the p=1 norm, we have to do a transformation to convert the 
            ## max{|x_1|,...,|x_n|} into a linear form
            ## See documentation for notes on this
            convex_prog.t = pyo.Var([1]) #hopefully this produces one variable - NEED TO CHECK THIS!
            
            ## Two rules to establish the extra constraints
            def extra_inf_norm_constraints_part1(model,i):
                return model.ep[i] <= model.t[1]
            
            def extra_inf_norm_constraints_part2(model,i):
                return -1*model.ep[i] <= model.t[1]
            
            convex_prog.max_absval_constraints_part1 = pyo.Constraint(\
                                convex_prog.varindex,\
                                rule=extra_inf_norm_constraints_part1)
            convex_prog.max_absval_constraints_part2 = pyo.Constraint(\
                                convex_prog.varindex,\
                                rule=extra_inf_norm_constraints_part2)
            
            ########## Defining the Objective Function #########
            def obj_rule_p_inf(model):
                return model.t[1]
            convex_prog.obj = pyo.Objective(rule=obj_rule_p_inf)
            solver = SolverFactory('glpk') #we have a linear program now
                            #due to the transformation, so we can use an LP solver
            
            ############ Solving the Convex Progs #############
            #This help ticket indicated that we needed to give the interior point method a starting point (and indicated 0 was a bad one)
            #https://projects.coin-or.org/Ipopt/ticket/205
            sum_of_epsilons = 0
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                convex_prog.equal_constraint[index].activate() #activating relevant equality constraint
                solver.solve(convex_prog)
                #print("Objective function value is",pyo.value(convex_prog.obj))
                sum_of_epsilons = sum_of_epsilons + pyo.value(convex_prog.obj)
                convex_prog.equal_constraint[index].deactivate()
        else:
            print("Error with entered p value")
            return
            
        ##### Calculating the Rho #####
        average_of_epsilons = (1/dim1)*(sum_of_epsilons) 
        rho = 1 - (epsilon_star/average_of_epsilons)
        
        if if_append == 'T':
            self.rho_p.append(rho)
        else:
            self.rho_p = [rho]
            
    def calculate_rho_a(self):
        
        #Inputs: None
        
        #Purpose: Implements the calculation of $\rho_a$ using the data 
        #in the \url{A}, \url{b}, and \url{x0} attributes
        
        #Results/Output: Places the resulting computed $\rho_a$ into the 
        #\url{rho_a} attribute.  Does not return anything to the user
        
        
        ##### Calculate epsilon*_a #####
        self.GIO_abs_duality() 
        normed_epsilon_star_a = np.linalg.norm(self.epsilon_a[0],ord=np.inf) #use inf norm to get rid 
                                                                    #of the np.sign(A_row) in the epsilon_a
                                                                    #to recover the min ratio
                                                                    #that was calculated with .i_star(self.A,self.b,self.x0,1)
                
        ##### Need to Account for when we are too close to the boarder of the Feasible Region #####
        if normed_epsilon_star_a < 1e-12: #Need to exit the function 
            self.rho_a = [1] #rho would be 1 because if exactly on the boundary then you
                        #are in X^{OPT}, the ultimate sign of 'fit'
            return 
        
        ##### Calculate Ratios #####
        residuals = np.transpose( (np.matmul(self.A,self.x0) - self.b) )
        
        if np.any(residuals<0)==True: #none of the residuals should be less than 0 because x^0 is a feasible point; the code should break if there is a negative residual
            print("Error: Negative Residual and Thus Infeasible x^0")
            return #to get out of the function entirely
        
        ### Calculating the denominator: 1 norm for denominator ###  
        row_norms = np.transpose(np.linalg.norm(self.A,ord=1,axis=1)) 
        
        ratios = np.divide(residuals,row_norms) 
        ratios = np.reshape(ratios,(np.size(ratios),))
        
        sum_ratios = np.sum(ratios) #from experimenting, will sum no matter the dimensions of ratios
        (dim1,dim2) = np.shape(self.A)
        average_ratios = (1/dim1)*sum_ratios
        self.rho_a = [1-(normed_epsilon_star_a/average_ratios)]
        
    def calculate_rho_r(self):
        
        #Inputs: None
        
        #Purpose: Implements the calculation of $\rho_r$ using the data 
        #in the \url{A}, \url{b}, and \url{x0} attributes
        
        #Results/Output: Places the resulting computed $\rho_r$ into 
        #the \url{rho_r} attribute.  Does not return anything to the user
        
        
        [istar,min_ratio] = self.i_star(self.A,self.b,self.x0,'b') #assuming this is how
                            #we calculate istar
        
        ########### Calculating the |e_r^* - 1| (the Numerator) ############
        numerator = np.absolute( (np.dot(self.A[istar,:],self.x0)*(1/self.b[istar,0])) - 1)
        
        ### Need to Account for when we are too close to the boarder of the Feasible Region ###
        if numerator < 1e-12: #Need to exit the function 
            self.rho_r = [1] #rho would be 1 because if exactly on the boundary then you
                        #are in X^{OPT}, the ultimate sign of 'fit'
            return 
        
        
        ########### Calculating the Denominator #############
        (dim1,dim2) = np.shape(self.A)
        sum_denom = 0
        for i in range(dim1): 
            sum_denom = sum_denom + np.absolute( (np.dot(self.A[i,:],self.x0)*(1/self.b[i,0])) - 1)
        
        average_sum_denom = (1/dim1)*sum_denom
        #pdb.set_trace()
        
        ########### Calculating rho_r ##############        
        self.rho_r = [1-(numerator[0]/average_sum_denom[0])]
    
       
    def calculate_rho_p_approx(self,p):
        
        #Inputs: \url{p} specifies the type of norm (only $1,2,\infty$ 
        #supported)
        
        #Purpose: Implements the calculation of $\Tilde{\rho}_p$ using the 
        #data in the \url{A}, \url{b}, and \url{x0} attributes
        
        #Results/Output: Places the resulting computed $\Tilde{\rho}_p$ 
        #into the \url{rho_p_approx} attribute.  Does not return anything 
        #to the user
        
        
        if isinstance(p,str)==True: #making sure that if type in 'infinity' 
                                    #(or something like it), then gets converted
            p = 'inf'
        
        ############### Calculating the Numerator ###################
        self.GIO_p(p,'F')
        if p=='inf':
            numerator = np.linalg.norm(self.epsilon_p[0],ord=np.inf)
        else:
            numerator = np.linalg.norm(self.epsilon_p[0],ord=p)
        
        ### Need to Account for when we are too close to the boarder of the Feasible Region ###
        if numerator < 1e-12: #Need to exit the function 
            self.rho_p_approx = [1] #rho would be 1 because if exactly on the boundary then you
                        #are in X^{OPT}, the ultimate sign of 'fit'
            return 
        
        ############# Calculating the Denominator #####################
        #### Getting the Dual Norms ####
        if p==1:
            q = 'inf'
        elif p == 2:
            q = 2
        else: #presumably p = 'inf'
            q = 1
        
        ###### Getting the Min Projected Distance to Each Hyperplane (the Ratios) ######
        residuals = np.transpose( (np.matmul(self.A,self.x0) - self.b) )
        
        if np.any(residuals<0)==True: #none of the residuals should be less than 0 because x^0 is a feasible point; the code should break if there is a negative residual
            print("Error: Negative Residual and Thus Infeasible x^0")
            return #to get out of the function entirely
        
        if q == 'inf':
            row_norms = np.transpose(np.linalg.norm(self.A,ord=np.inf,axis=1)) 
        else:
            row_norms = np.transpose(np.linalg.norm(self.A,ord=q,axis=1))
        
        ratios = np.divide(residuals,row_norms) 
        ratios = np.reshape(ratios,(np.size(ratios),))
        
        ### Finding the Average ###
        (dim1,dim2) = np.shape(self.A)
        denominator = (1/dim1)*np.sum(ratios)
        
        ########### Calculating Rho_p Approximate ##############
        self.rho_p_approx = [1-(numerator/denominator)]
        
    def calculate_c_vector(self,istar,gio_model,if_append):
        
        #Inputs: \url{istar} index value, \url{gio_model} indicating 
        #whether \url{`p',`a',`r'}, and \url{if_append} to indicate whether 
        #or not for the \url{gio_model = `p'} option we would like values 
        #calculated to be appended
    
        #Purpose: Calculate $\c^*$ according to Theorem 
        #\ref{chan_closed_form_theorem}. This method is called by the 
        #other \url{GIO} methods, which means it is an internal method 
        #to the \url{GIO} class.
    
        #Results/Output: Places the computed $\c^*$ into the appropriate 
        #attribute based upon the \url{gio_model} input. Makes decision on 
        #whether or not to append based upon \url{if_append} input.  Does 
        #not return anything to the user.
        
        
        c_vector = self.A[istar,:]*(1/np.linalg.norm(self.A[istar,:],ord=1))
        ###Want to output a column numpy vector###
        c_vector = np.reshape(c_vector,(np.size(c_vector),1))
        #ratios = np.reshape(ratios,(np.size(ratios),))
        if gio_model == 'p':
            if if_append =='T':
                self.c_p.append(c_vector)
            else:
                self.c_p = [c_vector]                
        elif gio_model == 'a':
            self.c_a = [c_vector]
        elif gio_model == 'r':
            self.c_r = [c_vector]
    
    def GIO_structural_epsilon_setup(self):
        
        #Inputs: None
    
        #Purpose: Sets up a \url{pyomo} model with the $A(\x^0 - \ep) \geq b$ 
        #and all possible $\mathbf{a}'_i(\x^0 - \ep) = b_i$ constraints.  
        #All of the equality constraints are ``deactivated'' such that, on 
        #the $i$th solve, the $i$th equality constraint can be activated 
    
        #Results/Output: Places the \url{pyomo} model in the \url{GIO_struc_ep}
        #attribute. Does not return anything to the user.        
        
        ########### Defining the Model (the Feasible Region Part) ############
        
        A = self.A #copying to local variable because the local object "self" calls could get weird
        b = self.b
        x0 = self.x0
        (dim1,dim2) = np.shape(A) #getting the dimensions for the variable sets
        convex_prog = pyo.ConcreteModel()
        convex_prog.numvars = pyo.Param(initialize=dim2)
        convex_prog.numeqs = pyo.Param(initialize=dim1)
        convex_prog.varindex = pyo.RangeSet(1,dim2)
        convex_prog.Arowindex = pyo.RangeSet(1,dim1)
        convex_prog.ep = pyo.Var(convex_prog.varindex)
        
        ##### Feasibility Constraints #####
        def feas_rule(self,i): #we are assuming that b is a column vector with functioning 2 axes
            return sum( A[i-1,j-1]*(x0[j-1,0]-self.ep[j]) for j in range(1,self.numvars+1)) >= b[i-1,0] 
        convex_prog.feas_constraint = pyo.Constraint(convex_prog.Arowindex,rule=feas_rule)
        
        ##### The ith constraint (equality) #####
        def equal_rule(self,index): 
            return sum( A[index-1,j-1]*(x0[j-1,0]-self.ep[j]) for j in range(1,self.numvars+1) ) == b[index-1,0]
        
        convex_prog.equal_constraint = pyo.Constraint(convex_prog.Arowindex,rule=equal_rule) #assuming that we are Pyomo indexing
        convex_prog.equal_constraint.deactivate() # we will only activate the equality constraint
                                                #that is relevant to the given solve
                    
        #### Putting the Model in Right Attribute ####
        self.GIO_struc_ep = convex_prog
        
        
    def GIO_structural_epsilon_solve(self,p,FLAG=1984000):
        
        #Inputs: \url{p} to indicate the $p$-norm under which we are 
        #working and \url{FLAG} is the large number we internally place 
        #into the array containing the objective function values for each
        #of the $m$ solves.  We have a default setting for the value, but 
        #the user can change the \url{FLAG} if he/she/they believe their 
        #specific application warrants it.  The objective with \url{FLAG} 
        #is that it be larger than any of the actual $\ep^i$ values.
    
        #Purpose: Solve the \url{GIO_struc_ep} model (with any additional 
        #constraints on $\ep$ added in by the user) according to the $p$ 
        #norm specified.
    
        #Results/Output: Places the $\ep^*$ in \url{epsilon_p}, $\x^0 - \ep^*$ 
        #in \url{x0_epsilon_p}, $\c^*$ in \url{c_p} and, if relevant, places 
        #multiple $i^*$ indices in the \url{istar_multi} attribute.  Also 
        #places the calculated $\rho_p$ into \url{rho_p}. Does not return 
        #anything to the user.
        
        
        ##### Objective Function & Solution #####
        ##This will depend upon the norm we are imposing.##
        
        (dim1,dim2) = np.shape(self.A) #getting the dimensions of A
        container_for_obj_vals = np.zeros((dim1,)) #since dim1 represents the number of equations
        
        if isinstance(p,str)==True:
            p = 'inf' #making sure that, if a string was passed in for p, then 
                    #we are finding the infinity norm
        
        #################### Finding the ep^i for GIO_struc_ep ######################## 
        
        if p==2:
            def obj_rule_p2(model): #worked with "self" as well
                return pyo.sqrt( sum(model.ep[j]**2 for j in range(1,model.numvars+1)) )
            self.GIO_struc_ep.obj = pyo.Objective(rule=obj_rule_p2)
            solver = SolverFactory('ipopt') #need nonlinear because of the 
                                    #objective function
            #pdb.set_trace()
            ############ Solving the Convex Progs #############
            #This help ticket indicated that we needed to give the interior point method a starting point (and indicated 0 was a bad one)
            #https://projects.coin-or.org/Ipopt/ticket/205
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                for i in range(1,self.GIO_struc_ep.numvars+1):
                    self.GIO_struc_ep.ep[i] = 0.01 #have to give interior point algorithm a non-zero starting place
                
                self.GIO_struc_ep.equal_constraint[index].activate() #activating relevant equality constraint
                results = solver.solve(self.GIO_struc_ep)
                #### Checking for Feasibility ####
                if str(results.solver.termination_condition) == "infeasible":
                    print("We have infeasibility for constraint=",str(index-1),".  Putting FLAG in the ",\
                          "container_for_obj_vals vector")
                    container_for_obj_vals[index-1] = FLAG
                else: 
                    container_for_obj_vals[index-1] = pyo.value(self.GIO_struc_ep.obj)
                
                #### Deactivating the Constraint we Needed to Deactivate ####
                self.GIO_struc_ep.equal_constraint[index].deactivate()
        elif p==1:
            ###Since there are absolute values in the objective function for the L1 norm, we need to do a 
            ##transformation that will actually linearize the problem.
            #See documentation/chapter for references on this
            self.GIO_struc_ep.u = pyo.Var(self.GIO_struc_ep.varindex) #variables to replace x in the objective func
            
            ##Two rules to establish the extra constraints
            ##ConstraintList didn't like the convex_prog.varindex, hence
            ##the two rules
            def extra_abs_val_constraints_part1(model,i):
                return model.ep[i] <= model.u[i]
            
            def extra_abs_val_constraints_part2(model,i):
                return -1*model.ep[i] <= model.u[i]
            
            self.GIO_struc_ep.abs_val_constraints_part1 = pyo.Constraint(\
                                self.GIO_struc_ep.varindex,\
                                rule=extra_abs_val_constraints_part1)
            self.GIO_struc_ep.abs_val_constraints_part2 = pyo.Constraint(\
                                self.GIO_struc_ep.varindex,\
                                rule=extra_abs_val_constraints_part2)
            
            ###Defining the Objective Function###
            def obj_rule_p1(model):
                return sum(model.u[j] for j in range(1,model.numvars+1))
            self.GIO_struc_ep.obj = pyo.Objective(rule=obj_rule_p1)
            solver = SolverFactory('glpk') #we have a linear program now
                            #due to the transformation, so we can use an LP solver
            
            ############ Solving the Convex Progs #############
            
            
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                #for i in range(1,self.GIO_struc_ep.numvars+1):
                    #self.GIO_struc_ep.u[i] = 0.01 
                
                self.GIO_struc_ep.equal_constraint[index].activate() #activating relevant equality constraint
                #pdb.set_trace()
                results = solver.solve(self.GIO_struc_ep)
                #### Checking for Feasibility ####
                if str(results.solver.termination_condition) == "infeasible":
                    print("We have infeasibility for constraint=",str(index-1),".  Putting FLAG in the ",\
                          "container_for_obj_vals vector")
                    container_for_obj_vals[index-1] = FLAG
                else: 
                    container_for_obj_vals[index-1] = pyo.value(self.GIO_struc_ep.obj)
                
                
                self.GIO_struc_ep.equal_constraint[index].deactivate()
        
        elif p=='inf': #we have the infinity norm
            #### Similar to the p=1 norm, we have to do a transformation to convert the 
            ## max{|x_1|,...,|x_n|} into a linear form
            ## See documentation for notes on this
            self.GIO_struc_ep.t = pyo.Var([1])  
            
            ## Two rules to establish the extra constraints
            def extra_inf_norm_constraints_part1(model,i):
                return model.ep[i] <= model.t[1]
            
            def extra_inf_norm_constraints_part2(model,i):
                return -1*model.ep[i] <= model.t[1]
            
            self.GIO_struc_ep.max_absval_constraints_part1 = pyo.Constraint(\
                                self.GIO_struc_ep.varindex,\
                                rule=extra_inf_norm_constraints_part1)
            self.GIO_struc_ep.max_absval_constraints_part2 = pyo.Constraint(\
                                self.GIO_struc_ep.varindex,\
                                rule=extra_inf_norm_constraints_part2)
            
            ########## Defining the Objective Function #########
            def obj_rule_p_inf_struc(model):
                return model.t[1]
            self.GIO_struc_ep.obj = pyo.Objective(rule=obj_rule_p_inf_struc)
            solver = SolverFactory('glpk') #we have a linear program now
                            #due to the transformation, so we can use an LP solver
            
            ############ Solving the Convex Progs #############
            
            constraint_indices = [1+k for k in range(0,dim1)] #want dim1 because want number of rows
            for index in constraint_indices:
                self.GIO_struc_ep.equal_constraint[index].activate() #activating relevant equality constraint
                #pdb.set_trace()
                results = solver.solve(self.GIO_struc_ep)
                #### Checking for Feasibility ####                
                if str(results.solver.termination_condition) == "infeasible":
                    print("We have infeasibility for constraint=",str(index-1),".  Putting FLAG in the ",\
                          "container_for_obj_vals vector")
                    container_for_obj_vals[index-1] = FLAG
                else: 
                    container_for_obj_vals[index-1] = pyo.value(self.GIO_struc_ep.obj)                
                #solver.solve(self.GIO_struc_ep)
                #container_for_obj_vals[index-1] = pyo.value(self.GIO_struc_ep.obj)
                self.GIO_struc_ep.equal_constraint[index].deactivate()
                #pdb.set_trace()
        else:
            print("Error with entered p value")
            return
    
        ####### Find the Minimal Element in the Set #########  
        ###We will call this istar_struc because this is the GIO_struct_ep model
        #pdb.set_trace()
        ##Need to first do a feasibility check##
        if container_for_obj_vals.min() == FLAG:
            print("Error, entire problem infeasible")
            return 
        
        ########### Finding istar ###########
        (istar_struc,) = np.where(container_for_obj_vals == container_for_obj_vals.min()) #remember indexes from 0        
        
        if np.size(istar_struc) > 1:
            print("Multiple feasible epsilon^i are minimal",\
                  "For now, we will choose the first i index",\
                  "and will put the rest of the indices in the istar_multi",\
                  "attribute.")
            self.istar_multi = [istar_struc] #replaces previous istar ()
            istar_struc = istar_struc[0] #picking the istar 
        else:
            istar_struc = istar_struc[0] #because istar is coming back as an array when we
                            #want it to be an index
        
        
        ##################### Solve the Mathematical Program One More Time #########################
        if p==2:
            solver = SolverFactory('ipopt')
            for i in range(1,self.GIO_struc_ep.numvars+1):
                self.GIO_struc_ep.ep[i] = 0.01 #have to give interior point algorithm a non-zero starting place
                
            self.GIO_struc_ep.equal_constraint[istar_struc+1].activate() #activating relevant equality constraint
                                                    #NEED that + 1 because python indexes from 0 and pyomo indexes
                                                    #from 1
            solver.solve(self.GIO_struc_ep)
            
            ####Obtaining the Values for the Epsilon Vector#####
            epsilon = np.zeros((dim2,1)) #since dim2 is the number of variables
            #pdb.set_trace()
            for i in range(1,dim2+1): #since pyomo models and python are on different indexing systems
                epsilon[i-1,0] = self.GIO_struc_ep.ep[i].value                
            
        elif p==1:
            solver = SolverFactory('glpk')
            for i in range(1,self.GIO_struc_ep.numvars+1):
                self.GIO_struc_ep.u[i] = 0.01 #have to give interior point algorithm a non-zero starting place
                
            self.GIO_struc_ep.equal_constraint[istar_struc+1].activate() #activating relevant equality constraint
            solver.solve(self.GIO_struc_ep)
            ####Obtaining the Values for the Epsilon Vector#####
            epsilon = np.zeros((dim2,1)) #since dim2 is the number of variables
            for i in range(1,dim2+1): #since pyomo models and python are on different indexing systems
                epsilon[i-1,0] = self.GIO_struc_ep.ep[i].value            
            
        elif p=='inf':
            solver = SolverFactory('glpk')
            self.GIO_struc_ep.t[1] = 0.01 #resetting for the heck of it - shouldn't cause any problems
            
            self.GIO_struc_ep.equal_constraint[istar_struc+1].activate() #activating relevant equality constraint
            solver.solve(self.GIO_struc_ep)
            
            ####Obtaining the Values for the Epsilon Vector#####
            epsilon = np.zeros((dim2,1)) #since dim2 is the number of variables
            for i in range(1,dim2+1): #since pyomo models and python are on different indexing systems
                epsilon[i-1,0] = self.GIO_struc_ep.ep[i].value            
        
        
        ########## Storing Things in Attributes ###########
        self.epsilon_p = [epsilon]
        self.x0_epsilon_p = [self.x0 - epsilon]
        
        ########## Calculating the Rho Part #################
        #istar_struc, container_for_obj_vals
        #we can just use the container_for_obj_vals to calculate rho
        ###Need to do specialized sum in case of those infeasible values###
        sum_of_obj_vals = 0
        num_feasible = 0
        for i in range(dim1):
            if container_for_obj_vals[i] == FLAG:
                continue
            else:
                sum_of_obj_vals = sum_of_obj_vals + container_for_obj_vals[i]
                num_feasible = num_feasible + 1
            
        rho_struc = 1-( container_for_obj_vals[istar_struc]/(sum_of_obj_vals*(1/num_feasible)) )
        self.rho_p = [rho_struc] #storing in the rho_p (there is no rho_approx)
        
        ########## Calculating c ##########
        self.calculate_c_vector(istar_struc,'p','F') #so the c will be put in the c_p attribute              
    
    
    ###################### To be continued methods/functions ################################# 
    
    def GIO_structural_c_setup(self):
        print("working on, in development") 
        
    
    def GIO_structural_c_solve(self):           
        print("working on, in development")
                   
        
            
            
        
                    
        
        
        
        
        
        
        
        
        
        
        
        
        