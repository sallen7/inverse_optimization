#### experiment_dong_consumer_behavior_text_Online_GIO.py ####

# 5/20/2019

# This file runs the "learning consumer preferences" (DCZ 2018)
# computational experiment we carried out from  
# the Dong, Chen, & Zeng 2018 paper.

# Readers can find more information about running this experiment by going to
# Section 1.4.4 Validation/Usage and scrolling to the part that says
# Computational Experiment.

# Additional Notes:

# DCZ = Dong, Chen, & Zeng


import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import time
import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pickle
import math

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing
#from experiment_dong_consumer_behavior_gen_data import num_samples,c_dict #NOTE: if you comment this out
                                                        #then it will actually use the data you already
                                                        #produced when you ran the generating data script
                                                        #ALSO:^don't need to introduce module sequence because we run this script within the
                                                        #experiments directory

###### Step -1: Define some variables from data generation script ##########

num_samples = 1000
c_dict = {1:-1.180,2:-1.733,3:-1.564,4:-0.040,5:-2.443,6:-1.055,\
          7:-4.760,8:-5.00,9:-1.258,10:-4.933} #made r negative since we will want to minimize



############## Step 0: Loading the Data & Set Up Model #################
###    Load Data    ###
loading_data_info = open("dong_p_t.pickle","rb")

p_t_samples = pickle.load(loading_data_info) 
loading_data_info.close()

loading_data_info2 = open("dong_y_t.pickle","rb")
y_t_samples = pickle.load(loading_data_info2)
loading_data_info2.close()

#########    Set up Model    ###############
#a. Sets and Vars
cb_model = pyo.ConcreteModel()
cb_model.varindex = pyo.RangeSet(1,10)
cb_model.x = pyo.Var(cb_model.varindex) #EDIT 5/4/2019,domain=pyo.NonNegativeReals)
cb_model.numvars = pyo.Param(initialize=10)
cb_model.eqindex = pyo.RangeSet(1,1)

#b. Parameters
cb_model.p_t = pyo.Param(cb_model.eqindex,\
                        cb_model.varindex,initialize=0) 

cb_model.bscalar = pyo.Param(cb_model.eqindex,initialize={1:40})

diag_vec = np.array([2.360,3.465,3.127,0.0791,4.886,2.110,\
        9.519,9.999,2.517,9.867]) 
Q = np.diag(diag_vec) #THIS IS A POSITIVE DEFINITE MATRIX (DCZ wrote a positive def
                        #matrix in their supplemental material when they
                        #meant to write a negative definite)

def Q_param_rule(model,i,j):
    return Q[i-1,j-1]

cb_model.Qmat = pyo.Param(cb_model.varindex,\
                cb_model.varindex,rule=Q_param_rule)

c_dict_dummy = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}

cb_model.cvec = pyo.Param(cb_model.varindex,\
                               initialize=c_dict_dummy)


##### Step 1: Create an Instance of the Online_IO Class #####
##### & Initiate the Dong_implicit_update Algorithm #####

time_0_process = time.process_time() #actual start of the algorithm
                                    #Thanks to: https://docs.python.org/3/library/time.html
time_0_counter = time.perf_counter()

online_cb = Online_IO(cb_model,Qname='Qmat',cname='cvec',Aname='p_t',\
          bname='bscalar',Dname='None',fname='None',dimQ=(10,10),dimc=(10,1),\
          dimA=(1,10),dimD=(0,0),binary_mutable=[0,0,1,0,0,0],non_negative=1,\
          feasible_set_C=(-5,0)) #NEW feasible_set_C added 5/12/2019

online_cb.initialize_IO_method("Dong_implicit_update")

 
##### Step 2-3: Iterate Through the Data and, for each iteration, #####
##### Run receive_data and next_iteration in Sequence #####

for i in range(1,num_samples+1):
    
    #### Step 2: Update Model with New Data ####
    online_cb.receive_data(p_t={"p_t":p_t_samples[i]},x_t=y_t_samples[i]) #receive data sample
                                                                    #remember that the p_t needs to be named by the parameter
    
    ## The next few statements do some basic checks.  They aren't comprehensive, but ##
    ## they are useful to showcase because they indicate some approaches to debugging pyomo models ##
    ## See documentation for sources regarding the pyomo package and its documentation (and some sites I found helpful) ##
    
    ##Checking that p_t gets put into the KKT_model##
    assert online_cb.KKT_conditions_model.A.extract_values() == p_t_samples[i]

    ### Checking some Pyomo flags for Constraint Integrity ###
    for constr in online_cb.KKT_conditions_model.component_objects(pyo.Constraint):
        assert constr._constructed == True, "Error in constraint construction (body)" 
        for c in constr:
            lb = pyo.value(constr[c].lower)
            ub = pyo.value(constr[c].upper)
            assert ((lb is not None) or (ub is not None)), "Error in constraint construction (LHS/RHS)"
    
    #### Step 3: Perform Update Rule ####
    online_cb.next_iteration(eta_factor=5) #go to the next iteration


##### TIMING ######    
print("Process_time Run time (seconds) for the script: ")             
time_1_process = time.process_time() - time_0_process
print(time_1_process) 

print("Perf_counter Run time (seconds) for the script: ")             
time_1_counter = time.perf_counter() - time_0_counter
print(time_1_counter)


######################  Step 4: Graphing Stuff ##############################

##### Step 4a: Graph the Loss ##### 

losses = np.array(online_cb.losses_dong)
cumsum_losses = np.cumsum(losses)
average_cumsum_losses = np.divide(cumsum_losses,np.arange(1,num_samples+1))

losses_scatter = plt.scatter(np.arange(1,num_samples+1),losses,c='c',\
                    label="loss each time step")
losses_scatter = plt.plot(np.arange(1,num_samples+1),average_cumsum_losses,'r',\
                linewidth=2,label="average cumulative loss")
losses_scatter = plt.plot(np.arange(1,num_samples+1),0.2083*np.ones((num_samples,)),\
                          '--k',label="variance of noise")

plt.legend()
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.title('Loss over the Iterations',fontsize=20)
plt.show()

##### Step 4b: Graph 2-Norm Between c_t and c_true #####
    
#c_dict has the true (negated) r
c_ph = np.zeros((10,))
for j in range(1,11):
    c_ph[j-1] = c_dict[j]

c_error = np.zeros((num_samples,))

for i in range(1,num_samples+1):
    ph = online_cb.c_t_dict_dong[i]
    ph_np = np.zeros((10,))
    for j in range(1,11):
        ph_np[j-1] = ph[j]
    c_error[i-1] = np.linalg.norm(ph_np - c_ph)
     
error_in_c_graph = plt.semilogy(np.arange(1,num_samples+1),c_error)
plt.ylim((10**(-1),10))
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('log(2-norm(c_t - c_true))',fontsize=15)
plt.title('Diff between c_t and c_true (logged)',fontsize=20)
plt.show()

     






#######################################################################
####################################################################
##### Step 4a: Graph the Loss #####
#print("This is our loss vector:")
#print(online_cb.losses_dong)   #the loss dots are all over for Dong as well  
#
#losses = np.array(online_cb.losses_dong) 
#losses_scatter = plt.scatter(np.arange(1,num_samples+1),losses)
#plt.xlabel('Iterations',fontsize=20)
#plt.ylabel('Loss',fontsize=20)
#plt.title('Loss over the Iterations',fontsize=20)
#plt.show()
#
#
###### Step 4b: Graph 2-Norm Between c_t and c_true #####
#    
##c_dict has the true r (negated true vec)
#c_ph = np.zeros((10,))
#for j in range(1,11):
#    c_ph[j-1] = c_dict[j]
#
#c_error = np.zeros((num_samples,))
#
#for i in range(1,num_samples+1):
#    ph = online_cb.c_t_dict_dong[i]
#    ph_np = np.zeros((10,))
#    for j in range(1,11):
#        ph_np[j-1] = ph[j]
#    c_error[i-1] = np.linalg.norm(ph_np - c_ph)
# 
##pdb.set_trace()
#error_in_c_graph = plt.plot(np.arange(1,num_samples+1),c_error)
#plt.xlabel('Iterations',fontsize=30)
#plt.ylabel('2-norm(c_t - c_true)',fontsize=30)
#plt.title('Diff Between c_t and c_true',fontsize=30)
#plt.show()
#
###### Step 4c: Average Regret against the Bound #####
#online_loss_np = np.array(online_cb.losses_dong)
#batch_loss_np = np.array(online_cb.opt_batch_sol)

############################################################






## Cumsum Stuff ##
#cumsum_online_loss = np.cumsum(online_loss_np)
#cumsum_batch_loss = np.cumsum(batch_loss_np)
#
#regret = cumsum_online_loss - cumsum_batch_loss
#avg_regret = np.divide(regret,np.arange(1,num_samples+1))
#
#bound_func = lambda x: 1/(np.sqrt(x)) #following the lead of BMPS 2018
#
#pdb.set_trace()
### Graphing ##
#avg_regret_plot = plt.semilogy(np.arange(1,num_samples+1),avg_regret,label='average regret')
#avg_regret_plot = plt.semilogy(np.arange(1,num_samples+1),bound_func(np.arange(1,num_samples+1)))
#plt.xlabel('Iterations',fontsize=20)
#plt.ylabel('Average Regret (log scale)',fontsize=20)
#plt.title('Average Regret and Bound on Regret',fontsize=20)
#plt.show()
####NEED TO COME BACK TO THIS AND SEE IF i NEED TO CHANGE ANYTHING ABOUT THE
### GRAPHS - need to test if stuff works
        

#NEED TO ALSO MAKE A GRAPH WITH r!
#probably can just build that as another list attribute
#into the class 



