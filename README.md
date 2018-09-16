				REINFORCEMENT LEARNING

For my project I built a grid type environment in which there is an agent( red rectangle), some bad cells(blacks rectangles) and one target( yellow ). Only the agent can move, while the others objects are fixed.
Every time that the agent hits one black cell it  loses the match, instead if it goes in the yellow cell it wins. If one of this two situations is true the game ends and the grid is recreated with a different configuration of black and yellow cells.
The various configurations are randomly chosen using the numpy library, the only constrains I put are that the Agent starts moving always from the cell upper left and the yellow can’t be in the same cell of the black.
The agent can move only in 4 direction(up, down, left, right) and if it hits the borders of the map nothing change.




























My approach is the DeepQ Learning, I made several attempts, changing reward function, observation, parameters and configurations.
The Q learning approach is a technique that helps to calculate the expected utility of the available actions for a given state. It uses the function Q(s,a) that returns the expected utility executing action a while the state is S.
Deep Q learning is an application of Q learning in deep learning, the system uses a neural network to represent and calculate the Q function. It uses  an  iterative update that adjusts Q while the program is executing.


My project consists of 3 programs:

- In Maze.py you build the environment, agents and obstacles

- RunThis.py is the place where the environment is launched and information is collected to be sent to the neural network

-Brain.py is the neural network that processes all data with the deepQ approach.


OBSERVATION and REWARD:

I have made several attempts by changing the configurations of the network and passing different values of Observation and Reward, but those that have given me better results in the end are:

As Observations I passed the distances between the agent and the other objects in the map, I considered the distances projected along the two vertical and horizontal axes: so with 5 obstacles and a target I have observations composed of 12 numerical elements.

As far as Reward is concerned, my configuration associates the following values to the Reward:
+10 if you reach the target,
-5 if an obstacle is reached,
+1 if you are particularly close to the target,
-1 if you hit one of the walls of the map.
This last value in particular allowed to speed up the iterations because the agent often leaned against the walls and lengthened the research a lot.

I also tried other more complex mechanisms such as implementing a sort of obstacle avoidance algorithm and giving a positive reward to my agent each time he made a similar move to the algorithm, but the results were not good.


PROGRAMS:

I attach multiple versions of the same program, these differ mainly in neural networks: the results I obtained are quite different:
the first of the two has an initial phase of learning that allows its to reach values around 75%, then he stabilizes and remains more or less stable around that value;
The second one, after a learning phase, reaches values higher than 75%, exceeding 90%, but subsequently has a decline and falls to very low values.



NEURAL NETWORK ARCHITECTURES

My best network is built using Tensorflow and Pandas libraries;  inside it contains two identical neural networks, one used for evaluation and one to determine the best actions.
Both takes as input the states and as output a placeholder that have size equal to the number of actions and is proportional to the reward.
During the execution of the training we want to minimize the loss function defined as follows:

self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

That is equivalent to:







where q_eval are my labels while q_target are the results of the execution of the Q function:
q_target is calculated in my program as:

q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


The first Neural network is trained every time an action is performed and the reward is collected. So we have one input layer that receives 12 or 14 information and 2 hidden layers. At  the output layer there are 4 node (4 actions).

The other network is the complete model, it periodically saves the weights of the twin network;
it is used to predict an action given a state.  
During the execution of the program, the Agent interacts with the network through functions such as choose_action () and learn(), in this way it trains the weights of the networks. The action can be randomly chosen with an epsilon probability that decreases during program execution.


SOME RESULTS THAT I OBTAIN and BEST PARAMETERS:

These are some graphs that illustrate the various learning curves while the parameters of the neural network vary.
The graphs are obtained by averaging every 100 iterations of the result achieved, where 10 is the maximum and -1 or -5 is the minimum.
The parameter of the network that most affects the operation is certainly the epsilon greedy that decreases throughout the execution, so you must choose a decrease suitable for the number of iterations made. On average, to achieve good results, it takes from 500 to 1000 iterations.




In this graph is possible to see how the learning work, first attempts made more errors then last and the number decrease with the time.








In this image we can notice how sometimes after an initial learning peak that reaches the maximum value that is 10 there is a lowering phase.








Here, on the other hand, it is clear how, after a while, the average of the results obtained is around a value, we must take into account that these values are obtained by making 5000 iterations;
the value 6 in the axis of the ordinates means that the target is reached on average 0.75% of the time.
The values given below refer to the same circumstances:



Below you can see how often the target is reached (value 10) compared to when it is not reached (-5) in the last iterations:


(4945, ' ', 10)(4946, ' ', -5)(4947, ' ', 10)(4948, ' ', 10)(4949, ' ', -5)(4950, ' ', 10)
(4951, ' ', 10)(4952, ' ', 10)(4953, ' ', 10)(4954, ' ', 10)(4955, ' ', -5)(4956, ' ', 10)(4957, ' ', 10)
(4958, ' ', -5)(4959, ' ', 10)(4960, ' ', 10)(4961, ' ', -5)(4962, ' ', -5)(4963, ' ', -5)(4964, ' ', 10)
(4965, ' ', -5)(4966, ' ', -5)(4967, ' ', 10)(4968, ' ', 10)(4969, ' ', 10)(4970, ' ', 10)(4971, ' ', 10)
(4972, ' ', 10)(4973, ' ', 10)(4974, ' ', 10)(4975, ' ', 10)(4976, ' ', 10)(4977, ' ', -5)(4978, ' ', 10)
(4979, ' ', 10)(4980, ' ', 10)(4981, ' ', 10)(4982, ' ', 10)(4983, ' ', -5)(4984, ' ', -5)(4985, ' ', -5)
(4986, ' ', 10)(4987, ' ', 10)(4988, ' ', 10)(4989, ' ', 10)(4990, ' ', 10)(4991, ' ', -5)(4992, ' ', 10)
(4993, ' ', 10)(4994, ' ', -5)(4995, ' ', 10)(4996, ' ', -5)(4997, ' ', -5)(4998, ' ', 10)(4999, ' ', 10)











SECOND PART: HYBRID APPROACH

As a second part of my project I tried to develop a hybrid method in which the system learns for itself, but it can also be operated by a human user and learn from the moves that he performs.

I made a first attempt in which the moves made by the user were not in any way valued unlike those performed in autonomous driving (so no change in the attribution of rewards).
In particular, in this first attempt I always and immediately reached the target to the agent, in 100 iterations of which the first 20 were done in manual guidance I got these results:
[The green graph indicates human interaction.]


No human interaction, flat green graph, irregular red graph because the agent is exploring the map.











Human interaction present in the first iterations, always reaches the target then maximum value, once the intervention stops the red graph returns irregular but maintains quite high values (first phase around 50% and later around 30%)




One thing to take into account when making this change is the epsilon greedy, in fact in my first experiments I had kept it unchanged, but then I realized a conceptual error of my reasoning, if I perform these tests in the first 100 iterations, the epsilon will be maximum and very close to 1 so my agent would not choose the best action anyway. In the simulation proposed here, epsilon has immediately value = 0.2.

A problem of this configuration is, however, that need for many iterations carried out by the human operator otherwise it can happen many times that the agent is in a state never seen before, and having a very low epsilon does not have the ability to escape, because its tendency is no longer to visit states not yet visited, and will therefore find itself in a stalemate.

To do a lot of iterations it is necessary a human operator that repeats the game many times, to speed up the times I wrote a small obstacles avoidance algorithm that works well enough and takes the place of the human operator. In this way I can perform 300 iterations maneuvered and the subsequent autonomous.





In the figure we can see that, once the work done manually has finished, the learning is still above average, which means that even the manual phase has contributed to learning.










CONCLUSIONS:

So concluding reinforcement learning is a good way to solve this problem and can lead to good results;
The hybrid method can be applied to the Reinforcement Learning in order to reduce the number of iterations necessary for learning and to limit negative examples to a minimum. This case is particularly useful in situations where a negative situation can damage the equipment



