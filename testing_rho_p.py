##### Generating the Heat Map to Validate Code #####
#This code produces heatmaps for rho_p and rho_p_approx
#for Example 1 from Chan et al.'s 2018 paper

#Users can change the "p" parameter to decide which map they
#would like to produce, along with the density parameter
#to determine the number of x^0 values to be calculated

#Homework from AMSC660 was very helpful with regard to the matplotlib
#commands/code.

import pdb #for debugging
import numpy as np
import time
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from gio import GIO

##################### Data ############################### 
A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

################ Generating the Heat Map ###############################

##### Set Up #####
time_0 = time.clock()
density = 70
p = 1 #for setting the norm
half_dense = int(density/2)
x_vals = np.linspace(0,5,density)
y_vals = np.linspace(0,4,density)

(mesh_x,mesh_y) = np.meshgrid(x_vals,y_vals) #meshing
mesh_z = np.zeros((density,density)) #creating the container to hold the rho_p values
                                    #for each of the x0 values 
mesh_z_approx = np.zeros((density,density))

#### Defining the Function ####
#Might eventually want an argument for the calculate_rho_p func
def generate_rho_p(x1,y1,i,j,p): #originally didn't have i and j as arguments so they were taken 
                            #as global since weren't defined in the function
    global mesh_z #making the mesh_z global so that we can actually define this function
    global mesh_z_approx
    x0_1 = np.array([[x1],[y1]])
    residuals_1 = np.dot(A,x0_1) - b
    if np.any(residuals_1<0)==True:
        print("Not feasible, moving on...") #and since there is already a 0 in the 
                                            #mesh_z matrix, we actually are pretty good
    else:        
        #pdb.set_trace()
        #### For p norm rho ####
        generating_GIO = GIO(A,b,x0_1)
        generating_GIO.calculate_rho_p(p,'F')  
        print("This is generating_GIO rho_p:",generating_GIO.rho_p) 
        mesh_z[i,j] = generating_GIO.rho_p[0] #storing the rho_p in the proper place of mesh_z
        ### For p norm rho approximate ####
        generating_GIO.calculate_rho_p_approx(p)
        print("This is generating_GIO rho_p approximate:",generating_GIO.rho_p_approx)
        #pdb.set_trace()
        mesh_z_approx[i,j] = generating_GIO.rho_p_approx[0] #storing the rho_p_approx in the mesh_z_approx
                               
#### Generating the Rho ####                                                                     
for i in range(0,density):
    for j in range(0,density):
        x1 = mesh_x[i,j]
        y1 = mesh_y[i,j]
        generate_rho_p(x1,y1,i,j,p)
        

############### Creating the Graphic #########################
#Adopted the colorbar parts of the code from: https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
#Link below also helped with understanding plt.scatter()
#https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html

plt.subplot(131)
colormap_option = plt.cm.get_cmap('rainbow') 
rho_map = plt.scatter(mesh_x,mesh_y,c=mesh_z,vmin=0,vmax=1,cmap=colormap_option)
plt.colorbar(rho_map)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Rho_p for Chan et al. (2018) Example 1, p=' + str(p))

plt.subplot(132)
colormap_option = plt.cm.get_cmap('rainbow') 
rho_map = plt.scatter(mesh_x,mesh_y,c=mesh_z_approx,vmin=0,vmax=1,cmap=colormap_option)
plt.colorbar(rho_map)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Rho_p_approx, p=' + str(p))


plt.subplot(133)
colormap_option = plt.cm.get_cmap('rainbow') 
diff_max = np.amax(mesh_z-mesh_z_approx)
rho_map = plt.scatter(mesh_x,mesh_y,c=(mesh_z-mesh_z_approx),vmin=0,vmax=diff_max,cmap=colormap_option)
plt.colorbar(rho_map)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('rho_p - rho_p_approximate, p=' + str(p))



print("Run time (seconds) for the script: ")             
time_1 = time.clock() - time_0
print(time_1)      
        
plt.show() #want the graphic to show after the time gets recorded

####Saving Numpy Array for rho_r Validation#####

#if p == 'inf':
#    np.save('inf_norm_rho_approx_mesh.npy',mesh_z_approx)
