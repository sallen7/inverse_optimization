##### Generating the Heat Map to Validate Code #####

import pdb #for debugging
import numpy as np
import time
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pyomo.environ as pyo
from pyomo.opt import SolverFactory #page 43 of the Oct 2018 documentation
from gio import GIO
#from joblib import Parallel, delayed #for parallelization

##################### Data ############################### 
A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

#### Just testing that Methods Run for One Point ####
#testmod = GIO(A,b,x0)
#testmod.calculate_rho_p('inf','F')
#print("This is testmod rho_p p=inf",testmod.rho_p)

#p=1: [0.54929577464788737]
#p=2:  [0.56450853266162038]
#p=inf: [0.58974358974358942] #they are diff numbers

#testmod.calculate_rho_a()
#print("This is testmod rho_a:",testmod.rho_a)
#rho_a: [0.58208955223880599] (I think it makes sense that decently similar to
#p=inf)

#testmod.calculate_rho_r()
#print("This is testmod rho_r:",testmod.rho_r)
#rho_r: [array([ 0.68421053])]  -> this is a bit higher than the others

#testmod.calculate_rho_p_approx(2)
#print("This is testmod rho_p approximate:",testmod.rho_p_approx)
#rho_p approximate: [0.56450853266162049] #pretty similar to the exact bc
#we are positioned in the feasible region such that the projections to
#the hyperplanes are likely still feasible (or almost there)

#also note that rho_approx <= rho


################ Generating the Heat Map ###############################

##### Set Up #####
time_0 = time.clock()
density = 70
#diff_max = 0.03#0.028
p = 'inf' #for setting the norm
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
        #####NOW NEED TO MAKE ANOTHER GRAPH WITH THE DIFFERENCE BETWEEN THE TWO (maybe a side by side )                                                                        
                               
#### Generating the Rho ####                                                                     
for i in range(0,density):
    for j in range(0,density):
        x1 = mesh_x[i,j]
        y1 = mesh_y[i,j]
        generate_rho_p(x1,y1,i,j,p)
        

##### Creating the Graphic #####
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
print(time_1) #Took 76 seconds for density of 30 (with no loop unrolling)
              #the image doesn't get very fine grained; so going to need more points
              #We were at like 45 seconds for 20
     
        
plt.show() #want the graphic to show after the time gets recorded
            #need a way for this to work in Jupyter




##########################################################################
##########Trying to Parallelize#########
##DEPENDING WHAT I HAVE TIME FOR: It would be cool to write a script or a small
##function that could take a series of x0 and basically do the parallelized checking that I
#want to do here for the various parts of the module

#OR, I can just make this script available on GITHUB and encourage people to reuse the code

#http://numba.pydata.org/
#https://scicomp.stackexchange.com/questions/19586/parallelizing-a-for-loop-in-python
#https://joblib.readthedocs.io/en/latest/parallel.html
#https://stackabuse.com/parallel-processing-in-python/ - might just go with parallel processes
#https://stackoverflow.com/questions/3474382/how-do-i-run-two-python-loops-concurrently - or do parallel queues as the
    #stackoverflow states
#https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop - do need to be careful
    #about the shared memory stuff - that "universal lock" that has come up in a few of these links could get triggered
#If end up using joblib, will need to download it with conda
    
    #Probably should do more other coding before do much parallel coding
    #probably should go more on the back burner??

#def compute_z(i,j,mesh_x,mesh_y,mesh_z):
#    x1 = mesh_x[i,j]
#    y1 = mesh_y[i,j]
#    x0_1 = np.array([[x1],[y1]])
#    residuals_1 = np.dot(A,x0_1) - b
#    if np.any(residuals_1<0)==True:
#        print("Not feasible, moving on...")
#    else:        
#        #pdb.set_trace()
#        generating_GIO = GIO(A,b,x0_1)
#        generating_GIO.calculate_rho_p(2,'F')
#        print("This is testmod rho_p",generating_GIO.rho_p)
#        mesh_z[i,j] = generating_GIO.rho_p[0] #storing the rho_p in the proper place of mesh_z
#








