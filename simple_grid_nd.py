# MDP for non-deterministic simple grid example
# Luca Iocchi 2015-2017

import random

# MDP definition for simple non-deterministic grid world

X = ['S0', 'S1', 'F', 'S3', 'S4', 'G']
A = ['R', 'L', 'U', 'D']

deltaS = { ('S0','R'): ['S1','F'],  ('S0','U'): ['S3'],
           ('S1','R'): ['F'],  ('S1','L'): ['S0'],  ('S1','U'): ['S4'],
           ('S3','R'): ['S4','G'],  ('S3','D'): ['S0'],
           ('S4','R'): ['G'],  ('S4','L'): ['S3'],  ('S3','D'): ['S0']
         }


reward = { ('S4', 'R', 'G'): 100, ('S3', 'R', 'G'): 100 }

x0 = X[0]
xg = X[5]
Sxf = [X[2], X[5]] # final states

# transition function at planning time
def delta(x, a):
    if deltaS.has_key((x,a)):
        return deltaS[(x,a)]
    else:
        return [x]


def P(x1,a,x2): # transition probability
    Sx2 = delta(x1,a)
    if (x2 in Sx2):
        if len(Sx2)==1:
            return 1
        else:
            return 1.0/len(Sx2)
    else:
        return 0

        

def r(x1,a,x2): # reward value
    if (reward.has_key((x1,a,x2))):
        return reward[(x1,a,x2)]
    else:
        return 0

# transition function at execution time
def delta_exec(x, a):
    Sx2 = delta(x,a)
    return random.choice(Sx2)

        
