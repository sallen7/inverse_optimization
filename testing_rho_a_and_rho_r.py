##### Generating the Heat Map to Validate Code #####

import pdb #for debugging
import numpy as np
import time
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pyomo.environ as pyo
from pyomo.opt import SolverFactory #page 43 of the Oct 2018 documentation
from gio import GIO
#from testing_rho_p import generate_rho_p
#from joblib import Parallel, delayed #for parallelization

##################### Data ############################### 
A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

################ Generating the Heat Map ###############################

##### Set Up #####
time_0 = time.clock()
density = 70
type_duality = 'r' #'a' or 'r'
half_dense = int(density/2)
x_vals = np.linspace(0,5,density)
y_vals = np.linspace(0,4,density)

(mesh_x,mesh_y) = np.meshgrid(x_vals,y_vals) #meshing
mesh_z = np.zeros((density,density)) #creating the container to hold the rho_p values
                                    #for each of the x0 values 

#### Defining the Function ####
#Might eventually want an argument for the calculate_rho_p func
def generate_rho_a_r(x1,y1,i,j,type_duality): #originally didn't have i and j as arguments so they were taken 
                            #as global since weren't defined in the function
    global mesh_z #making the mesh_z global so that we can actually define this function
    x0_1 = np.array([[x1],[y1]])
    residuals_1 = np.dot(A,x0_1) - b
    if np.any(residuals_1<0)==True:
        print("Not feasible, moving on...") #and since there is already a 0 in the 
                                            #mesh_z matrix, we actually are pretty good
    else:        
        #### For duality gap models ####
        generating_GIO = GIO(A,b,x0_1)
        if type_duality == 'a':
            generating_GIO.calculate_rho_a()  
            print("This is testmod rho_a:",generating_GIO.rho_a) #runs extremely quickly
            mesh_z[i,j] = generating_GIO.rho_a[0] #storing the rho_p in the proper place of mesh_z
        elif type_duality == 'r':
            generating_GIO.calculate_rho_r()  
            print("This is testmod rho_a:",generating_GIO.rho_r) #runs extremely quickly
            mesh_z[i,j] = generating_GIO.rho_r[0] #storing the rho_p in the proper place of mesh_z
        else:
            print("ERROR: invalid type_duality argument for function generate_rho_a_r")
            return
                                                                                
                               
########### Generating the Rho ###########                                                                     
for i in range(0,density):
    for j in range(0,density):
        x1 = mesh_x[i,j]
        y1 = mesh_y[i,j]
        generate_rho_a_r(x1,y1,i,j,type_duality)
        

##### Creating the Graphic #####
#Adopted the colorbar parts of the code from: https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
#Link below also helped with understanding plt.scatter()
#https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html

colormap_option = plt.cm.get_cmap('rainbow') 
rho_map = plt.scatter(mesh_x,mesh_y,c=mesh_z,vmin=0,vmax=1,cmap=colormap_option)
plt.colorbar(rho_map)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Rho_p for Chan et al. (2018) Example 1, duality=' + type_duality)

print("Run time (seconds) for the script: ")             
time_1 = time.clock() - time_0
print(time_1) #Took 76 seconds for density of 30 (with no loop unrolling)
              #the image doesn't get very fine grained; so going to need more points
              #We were at like 45 seconds for 20
     
        
plt.show() #want the graphic to show after the time gets recorded
            #need a way for this to work in Jupyter
            
####Testing the Rho_approx_inf matrix against the Absolute Duality Matrix####
if type_duality == 'a':
    mesh_z_inf_rho_approx = np.load('inf_norm_rho_approx_mesh.npy')
    diff_between = mesh_z_inf_rho_approx - mesh_z
    print("This is the 1 norm of the difference:",np.linalg.norm(diff_between,ord=1))
    print("This is the 2 norm of the difference:",np.linalg.norm(diff_between,ord=2))
    print("This is the inf norm of the difference:",np.linalg.norm(diff_between,ord=np.inf))

            
