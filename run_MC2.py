"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
	self.q_table.loc[s, a] += self.lr * (q_target - q_predict) # update


env = gym.make('Berzerk-ram-v0')
env = env.unwrapped
RL = SarsaTable(actions=list(range(env.action_space.n)))

total_steps = 0
k=[] # quanto sta in vita
m=[] # quanto guadagna in tot vita
rr=[] #reward per ogni singola esistenza 

for episode in range(5000):
        # initial observation
        observation = env.reset()
	vite = 3
	stayAlive=0
        # RL choose action based on observation
        action = RL.choose_action(str(observation))
	r=0

        while True:
            # fresh env
            #env.render()
	
            # RL take action and get next observation and reward
            observation_, reward, done,info = env.step(action)
	    r=r+reward			
	    nlife=info['ale.lives']
	    if vite-nlife != 0:
		   reward= reward-10
	    
	    if done:
		   reward= reward-10

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
            	print(episode,':stay alive',stayAlive)
	    	k.append(stayAlive)
		rr.append(r)
               	break

            observation = observation_
            total_steps += 1
	    vite=nlife
	    stayAlive +=1

    # end of game
print('game over')

plt.plot(k,'r--', rr, 'bs')
plt.show()

