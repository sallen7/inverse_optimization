##### Generating the Heat Map to Validate Code #####

import pdb #for debugging
import numpy as np
import time
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pyomo.environ as pyo
from pyomo.opt import SolverFactory #page 43 of the Oct 2018 documentation
from gio import GIO
#from joblib import Parallel, delayed #for parallelization

A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])

#testmod = GIO(A,b,x0)
#testmod.calculate_rho_p(1,'F')
#print("This is testmod rho_p p=1",testmod.rho_p)
#
#testmod.calculate_rho_p(2,'F')
#print("This is testmod rho_p p=2",testmod.rho_p)
#testmod = GIO(A,b,x0)
##pdb.set_trace()
#
#testmod.calculate_rho_p(2,'F') #remember that RHO_P is a 'list'
#
#print("This is testmod rho_p",testmod.rho_p) #seems to work on this one point

##### Generating the Heat Map ######
##Need to generate a mesh with (0,5) on the x and (0,4) on the y
##Can test the residual of each potential x0 from the mesh to see if Ax^0 >= b
##If not, then we just discard the point
##I think we DO want to "create a new GIO" for each x0 - will make parallelizing easier
    ##We are afterall doing 4 nonlinear solves (since we are doing the 2 norm)
## (although we can do some loop unrolling to do instruction-wise parallelization)




time_0 = time.clock()
density = 30
half_dense = int(density/2)
x_vals = np.linspace(0,5,density)
y_vals = np.linspace(0,4,density)

(mesh_x,mesh_y) = np.meshgrid(x_vals,y_vals) #meshing
mesh_z = np.zeros((density,density)) #creating the container to hold the rho_p values
                                    #for each of the x0 values
for i in range(0,density):
    for j in range(0,density):
        x1 = mesh_x[i,j]
        y1 = mesh_y[i,j]
        x0_1 = np.array([[x1],[y1]])
        residuals_1 = np.dot(A,x0_1) - b
        if np.any(residuals_1<0)==True:
            print("Not feasible, moving on...")
        else:        
            #pdb.set_trace()
            generating_GIO = GIO(A,b,x0_1)
            generating_GIO.calculate_rho_p(1,'F')
            print("This is testmod rho_p",generating_GIO.rho_p)
            mesh_z[i,j] = generating_GIO.rho_p[0] #storing the rho_p in the proper place of mesh_z




        ####Want to try to run the process below in parallel to the one above - need to 
        #Read the links below
#        x2 = mesh_x[i,j+half_dense] #might have to double check the indexing here
#        y2 = mesh_y[i,j+half_dense]
#        x0_2 = np.array([[x2],[y2]])
#        residuals_2 = np.dot(A,x0_2) - b
#        if np.any(residuals_2<0)==True:
#            print("Not feasible, moving on...")
#        else:        
#            #pdb.set_trace()
#            generating_GIO_2 = GIO(A,b,x0_2)
#            generating_GIO_2.calculate_rho_p(2,'F')
#            print("This is testmod rho_p",generating_GIO_2.rho_p)
#            mesh_z[i,j+half_dense] = generating_GIO_2.rho_p[0] #storing the rho_p in the proper place of mesh_z




#Adopted the colorbar parts of the code from: https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
colormap_option = plt.cm.get_cmap('RdYlBu') 
rho_map = plt.scatter(mesh_x,mesh_y,c=mesh_z,vmin=0,vmax=1,cmap=colormap_option)
#Link below also helped with understanding plt.scatter()
#https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
plt.colorbar(rho_map)

print("Run time (seconds) for the script: ")             
time_1 = time.clock() - time_0
print(time_1) #Took 76 seconds for density of 30 (with no loop unrolling)
              #the image doesn't get very fine grained; so going to need more points
              #We were at like 45 seconds for 20
     
plt.show() #want the graphic to show after the time gets recorded





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








