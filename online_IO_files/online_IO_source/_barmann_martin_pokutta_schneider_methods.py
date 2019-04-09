### Methods for B\"armann, Martin, Pokutta, & Schneider 2018 ###

import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.mpec as pyompec #for the complementarity
import math

def compute_learning_rate(self):
    pass

def project_to_F(self):
    pass

def solve_subproblem(self):
    pass

def gradient_step(self):
    pass

barmann_martin_pokutta_schneider_funcs = (compute_learning_rate,project_to_F,solve_subproblem,gradient_step)



