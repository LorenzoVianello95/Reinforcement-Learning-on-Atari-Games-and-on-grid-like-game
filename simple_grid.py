# MDP for simple grid example
# Luca Iocchi 2015-2017


# MDP definition for simple deterministic grid world

X = ['S0', 'S1', 'S2', 'S3', 'S4', 'G']
A = ['R', 'L', 'U', 'D']

deltaS = { ('S0','R'): 'S1',  ('S0','U'): 'S3',
           ('S1','R'): 'S2',  ('S1','L'): 'S0',  ('S1','U'): 'S4',
           ('S2','L'): 'S1',  ('S2','U'): 'G',
           ('S3','R'): 'S4',  ('S3','D'): 'S0',
           ('S4','R'): 'G',   ('S4','L'): 'S3',  ('S3','D'): 'S0' 
           #, ('G' ,'L'): 'S4',  ('G', 'D'): 'S2',
        }

reward = { ('S4', 'R'): 100, ('S2', 'U'): 100 }

x0 = X[0] # initial state
xg = X[5] # goal state
Sxf = [X[5]] # set of final states

# transition function at planning time
def delta(x, a):
    if deltaS.has_key((x,a)):
        return deltaS[(x,a)]
    else:
        return x

# transition function at execution time
def delta_exec(x, a):
    return delta(x,a)

    
def P(x1,a,x2): # transition probability    
    if (delta(x1,a) == x2):
        return 1
    else:
        return 0

        
def r(x1,a,x2): # reward value
    if (reward.has_key((x1,a))):
        return reward[(x1,a)]
    else:
        return 0

