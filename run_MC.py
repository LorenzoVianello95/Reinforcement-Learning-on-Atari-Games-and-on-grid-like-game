"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt

env = gym.make('Berzerk-ram-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
#print(env.observation_space.high)
#print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
k=[] # quanto sta in vita
m=[] # quanto guadagna in tot vita
rr=[] #reward per ogni singola esistenza 

for i_episode in range(500):

    observation = env.reset()
    ep_r = 0
    vite=3
    stayAlive=0
    r=0

    while True:
        #env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        r=r+reward

	nlife=info['ale.lives']
	if vite-nlife != 0:
		reward=-10
	#elif vite-nlife == 0:
	#	reward= reward+30
	
        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2), 'stay alive',stayAlive)
	    k.append(stayAlive)
	    m.append(round(ep_r, 2))
            rr.append(r)
            break

        observation = observation_
        total_steps += 1
	vite=nlife
	stayAlive +=1

#RL.plot_cost()
print(max(k))
print(sum(k) / float(len(k)))
print(max(m))
print(sum(m) / float(len(m)))
plt.plot(k,'r--', rr, 'bs',m,'g^')
plt.show()
plt.show()
