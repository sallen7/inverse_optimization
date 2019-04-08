#### File to Run the Experiment ####

#LEAVING OFF: My code doesn't appear to be consistent with the data
#It failed in like 9 iterations and now it is failing in 11.  We are
#getting some kind of infeasible solution

#ACTUALLY LEFT OFF: I'm checking my update rule function right now for if
#everything is updating correctly (note that the KKTconditionsmodel is
#indeed always with c set to 0 (since we arent updating those c values)))

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

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing
from experiment_dong_consumer_behavior_gen_data import num_samples,c_dict

##### Step 0: Loading the Data & Set Up Model #####
###    Load Data    ###
pickle_in = open("dong_p_t.pickle","rb")

p_t_samples = pickle.load(pickle_in) #it worked
                                    #will need to look into
                                    #the large data overhead
                                    #like if loading all this
                                    #data is gonna work out
                                    #AND need to see how the 
                                    #overwrite stuff will work
pickle_in = open("dong_y_t.pickle","rb")
y_t_samples = pickle.load(pickle_in)

#pdb.set_trace()

###    Set up Model    ###
#a. Sets and Vars
cb_model = pyo.ConcreteModel()
cb_model.varindex = pyo.RangeSet(1,10)
cb_model.x = pyo.Var(cb_model.varindex,domain=pyo.NonNegativeReals)
cb_model.numvars = pyo.Param(initialize=10)
cb_model.eqindex = pyo.RangeSet(1,1)

#b. Parameters
cb_model.p_t = pyo.Param(cb_model.eqindex,\
                        cb_model.varindex,initialize=0,mutable=True) 
                                                    #Zero should work because should be 
                                                    #updated BEFORE any solution stuff happens
                                                    #initializing with
                                                        #a dummy value
                                                    #MAKES SENSE TO PUT 5
                                                    #or 25 as start because those
                                                    #are the end of the ranges
                                                    #Putting 0 makes problem
                                                    #UNBOUNDED
                                                    #???For cold start, Dong
                                                    #initiated as a vec of 0s

cb_model.bscalar = pyo.Param([1],initialize={1:40})

diag_vec = np.array([2.360,3.465,3.127,0.0791,4.886,2.110,\
        9.519,9.999,2.517,9.867]) #BUT ALL OF THESE VALUES ARE +
Q = np.diag(diag_vec) #THIS IS A POSITIVE DEFINITE MATRIX

def Q_param_rule(model,i,j):
    return Q[i-1,j-1]

cb_model.Qmat = pyo.Param(cb_model.varindex,\
                cb_model.varindex,rule=Q_param_rule)

c_dict_dummy = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}#{1:-1.180,2:-1.733,3:-1.564,4:-0.040,5:-2.443,6:-1.055,\
          #7:-4.760,8:-5.00,9:-1.258,10:-4.933} #CANNOT have the solution as our c_dict
                  #for the actual model

cb_model.cvec = pyo.Param(cb_model.varindex,\
                               initialize=c_dict_dummy)


##### Step 1: Create an Instance of the Online_IO Class #####
##### & Initiate the Dong_implicit_update Algorithm #####

#Note that initial guess of c is vector of 1s

time_0 = time.clock() #actual start of the algorithm

online_cb = Online_IO(cb_model,Qname='Qmat',cname='cvec',Aname='p_t',\
          bname='bscalar',Dname='None',fname='None',dimQ=(10,10),dimc=(10,1),\
          dimA=(1,10),dimD=(0,0),binary_mutable=[0,0,1,0,0,0],non_negative=1)

online_cb.initialize_IO_method("Dong_implicit_update")


##### Step 2-3: Iterate Through the Data and, for each iteration, #####
##### Run receive_data and next_iteration in Sequence #####

for i in range(1,num_samples+1):
    
    #### Step 2: Update Model with New Data ####
    online_cb.receive_data(p_t={"p_t":p_t_samples[i]},x_t=y_t_samples[i]) #receive data sample
                                                                    #remember that the p_t needs to be named by the parameter
    
    ##Maybe can assert here that the parameter p_t gets put into the KKT_model##
    assert online_cb.KKT_conditions_model.A.extract_values() == p_t_samples[i]
    
    ##Making sure all the constraints were updated appropriately
    #Could make this a part of the online_IO class potentially?
    for constr in online_cb.KKT_conditions_model.component_objects(pyo.Constraint):
        assert constr.body != None, "Constraints not being re-rendered correctly"
    
    #### Step 3: Perform Update Rule ####
    online_cb.next_iteration(eta_factor=5) #go to the next iteration
    
print("Run time (seconds) for the script: ")             
time_1 = time.clock() - time_0
print(time_1)      

#WANT TO MAKE SURE THAT THE SETTING OF THE C fixed variable values is
#happening correctly
#BASICALLY any time values are being changed in the model, need to make
#sure that the model is not messing up
    
#Also getting errors about the stuff I'm doing with the loss_func
    #and the update_rule in terms of the replacing objectives, etc
    #a. Wondering if we need to clear something?
    #b. Overall, should put a pdb.set_trace() and just make sure everything
    #is clearing the way I'm expecting it to clear

##### Step 4: Graph the Loss #####
print("This is our loss vector:")
print(online_cb.losses_dong)   #the loss dots are all over for Dong as well  

losses = np.array(online_cb.losses_dong) 
losses_scatter = plt.scatter(np.arange(1,num_samples+1),losses)
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.title('Loss over the Iterations',fontsize=20)
plt.show()


##### Step 4b: Graph 2-Norm Between c_t and c_true #####
    
#c_dict has the true r
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
 
#pdb.set_trace()
error_in_c_graph = plt.plot(np.arange(1,num_samples+1),c_error)
plt.xlabel('Iterations',fontsize=30)
plt.ylabel('2-norm(Theta_t - True_Theta)',fontsize=30)
plt.title('Diff Between Theta_t and True Theta',fontsize=30)
plt.show()


        

#NEED TO ALSO MAKE A GRAPH WITH r!
#probably can just build that as another list attribute
#into the class 



