### Generating Some Test Models ###

import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

#Should go back and turn maybe the A mat or b vec into a mutable param

#Messing with the KKT condition stuff first is a good idea because will set
#us up well for manipulating pyomo models for the entire class

toyprob1 = pyo.ConcreteModel()
toyprob1.x = pyo.Var([1,2]) #two x variables
toyprob1.b_param = pyo.Param([1,2],initialize={1:4,2:6},mutable=True)
toyprob1.exper_mat = pyo.Param([1,2],[1,2],initialize={(1,1):1,(1,2):2,(2,1):3,(2,2):4})
#doesn't seem like you can just update with a straight dictionary
#maybe if I can do a hybrid numpy/pyomo thing
toyprob1.constraint1 = pyo.Constraint(expr=toyprob1.x[1]-toyprob1.x[2] >= toyprob1.b_param[1])
toyprob1.constraint2 = pyo.Constraint(expr=2*toyprob1.x[1]+toyprob1.x[2] == toyprob1.b_param[2]) #bounds should be like 0

toyprob1.obj_func = pyo.Objective(expr=5*toyprob1.x[1]-3*toyprob1.x[2])

toyprob1.pprint()

#Can I just provide a new dictionary to the mutable param object?

pdb.set_trace()

print("These are the constraints")
#The following loop is heavily based upon code from the Pyomo documentation
#https://media.readthedocs.org/pdf/pyomo/latest/pyomo.pdf
for i in toyprob1.component_objects(pyo.Constraint,active=True): #not sure if we want descend_into=True
    print("Constraint name", i)
    for index in i: #what happens when index into a generator?
        #pdb.set_trace() #want to get at the tuple that the documentation talks about
                        #NEED TO LEARN what a "generator" is
        print("Constraint index (could be None if this constraint object has one element):", index)
        if i[index].lower is None:
            #Thanks to https://stackoverflow.com/questions/3965104/not-none-test-in-python
            #and https://docs.quantifiedcode.com/python-anti-patterns/readability/comparison_to_none.html 
            #for the recommendation
            print("This constraint is not bounded below, thus is a <= constraint")
        else:
            print("this is lower bound :",i[index].lower.value)
            
        if i[index].upper is None:
            print("This constraint is not bounded above, thus it is a >= constraint")
        else:
            print("this is the upper bound :", i[index].upper.value)
        
        print("Equality constraint?",i[index].equality)
            
            
            
            
#        print("this is lower bound of ",i,", number",index,":",i[index].lower.value)
#        pdb.set_trace() #problem if the upper value is infinity - then get a "none" value 
#            #but we can use "p i[index].upper is None" to get a True statement
#        print("this is upper bound of ",i,", number",index,":",i[index].upper.value)
#        print("Equality constraint?",i[index].equality)
        
##Next, mess with the parameter objects##        

        
        
###PROGRESS####
#p i[index]
#<pyomo.core.base.constraint.SimpleConstraint object at 0x000002EAE828EA08>
#(Pdb) p i[index].lower
#<pyomo.core.expr.numvalue.NumericConstant object at 0x000002EADF775D38>
#(Pdb) p i[index].lower.value
#4.0 
        
#(Pdb) p i[index].equality
#False


#pdb.set_trace()

## Solving the Model ##
#solver = SolverFactory('glpk')
#solver.solve(toyprob1)