[//]: # (Image References)

[dqn]: https://user-images.githubusercontent.com/23042512/45259824-d8441780-b38a-11e8-94f9-6391923aa2f7.png "DQN Training"
[ddqn]: https://user-images.githubusercontent.com/23042512/45259828-e5610680-b38a-11e8-9885-75135a438094.png "DDQN Training"
[pixel dqn]: https://user-images.githubusercontent.com/23042512/45857880-aed2a680-bd0f-11e8-9fa5-c6f886449e04.png "Pixel DQN Training"
[ddqn video]: https://user-images.githubusercontent.com/23042512/45858185-ff96cf00-bd10-11e8-9b25-a16d5153d56b.gif "DDQN Test Video"
[dqn pixel video]: https://user-images.githubusercontent.com/23042512/45858306-a54a3e00-bd11-11e8-9c45-8d49ff23450f.gif "DQN Pixel Test Video"

# Project 1: Navigation (Banana Collection Agent)

![Pixel DQN Test Video][dqn pixel video]

Agent Performance Using Raw Pixels!

### Introduction
Reinforcement learning has long be thought to be an important tool in achieving human level Artificial Intelligence (AI). While we are still far away from anything remotely like human level AI, the advent of deep learning has significantly improved the performance of traditional reinforcement learning algorithms. In this article, we will look at my implementation for the banana collection project in the Udacity Deep Reinforcement Learning Nanodegree program. 

Using a simplified version of the Unity Banana environment, the objective of the project is to design an agent to navigate (and collect bananas!) in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The agent's observation space is 37 dimensional and the agent's action space is 4 dimensional (forward, backward, turn left, and turn right). The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. As a Udacity challenge, and as detailed below, I also trained the agent using just the raw input image pixels as the observations -- although this invoved a very different network and a lot more time commitment!

Before diving into the technical details, let us briefly cover the basics of Reinforcement learning.

******************************************
### Fundamentals of Reinforcement Learning
In broad terms, machine learning can be divided into three categories: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning is when we train a model directly from the ground truth and the model gets immediate feedback as to whether its prediction is correct or not. Unsupervised learning is training the model without having any knowledge of what is the correct answer, i.e. there is no supervisor to learn form. Reinforcement learning is training an agent to perform a particular behavior, but the feedback of whether the agent's behavior is correct or not is received after many time steps -- i.e. there is a delayed feedback (unlike in supervised learning where there is an immediate feedback). It is basically a method of sequential decision making where the model of the environment is unknown and the agent has to learn the optimal behavior (to get maximum rewards) in that environment. If the environment's model is known, then it is known as a planning problem. There are many different planning algorithms such as Dynamic Programming, Uniform Cost Search, A* etc. Strictly speaking, regardless of whether the environment's model is known or not, it is all under the domain of reinforcement learning. However, in common parlance, reinforcement learning is typically used for situations where the environment's model is unknown.

### Markov Decision Process (MDP)
The basic framework for a reinforcement learning problem is defined using a Markov Decision Process (MDP). In an MDP, an agent interacts with the environment by taking actions, and in return, the agent gets rewards and the next state information from the environment. And the process continues. The goal of the agent is to maximize its rewards. Essentially the goal is to find a policy that tells what actions to take from each state that will result in the maximum amount of rewards. It is called Markvov because the state incorporates all the information necessary about the past (i.e. history of all the visited states) such that the future is independent of the past given the present state. The problem however is that in many real world situations, the enviromnemt is only partially observable, i.e. we end up with a Partially Observable Markov Decision Process (POMDP). And when the environment is only partially observable, the Markov propety no longer holds as the future is dependent on the past, even given the present state. To convert a POMDP into an MDP, the state can be augment such that the current state is a combination of observations from a few times steps in the past. But this doesn't always work as it is not clear how many time steps in the past to look at. Another way is to use some sort of a recurrent model (e.g RNN) that combines all the historical information (using some learned mathematical relationship) to turn partial observations into full state representations; although, this tends to increase the overall complexity.

Once given an MDP, the goal of an agent (using RL algorithms) is to take the optimal actions to get the highest possible rewards. The MDP is considered solved once we have this optimal policy. However, in order to get the optimal policy, for each state, we need to determine what is the value of that state using the current policy. And by definition, the value vpi(s) of a state s is the expected total reward (with the appropriate discounting factor) obtained by following the policy pi from that state onwards. The q-value qpi(s,a) is the state-action value function that is formally defined as the expected total reward (with the appropriate discounting factor) obtained by taking action a from state s, and then following the policy pi thereafter. There are two commonly used value-based RL algortihms to solve MDPs: Monte-Carlo and Temporal Difference learning methods. 

### Monte-Carlo Learning
Intuitively, Monte-Carlo (MC) learning works as follows: we start off with a random policy from a starting state s and taking action a and then follow the policy pi until termination (i.e. reaching the terminal state). The q-value of state s and action a is then updated using the following equation:

qpi(s,a) = qpi(s,a) + alpha(Gt-qpi(s,a)) ..... EQ. 1, where Gt is the total rewards obtained for the episode and alpha is the learning rate.

This same process of updating the q-value is done for all the states encountered in the episode. Then the policy is updated such that for each state (that has been previously visited), we pick the action with the highest q-value. But given the q-values for the policy pi are not accurate as they're only an estimate, this will be a greedy policy that can be suboptimal. To address it, we use an epsilon-greedy policy where with a probability epsilon (eps), we randomly pick an action from state s and with probability 1-eps, we follow the greedy policy. This helps balance exploration-exploitation tradeoff that is so important in reinforcement learning. The process of iteratively updating the eps-greedy policy and the q-values of all the visited states eventually leads to q-value convergence to the optimal policy's q-values. One downside of the MC method is that it only works for episodic tasks (i.e. tasks that terminate). Moreover, it also takes longer time to learn than other algorithms.

### Temporal-Difference Learning
Temporal-Difference (TD) learning is another type of Reinforcement Learning algorithm that combines the best of Monte-Carlo (MC) learning and Dynamic Programming (DP). Like DP, we bootstrap by updating the q-values after one step instead of waiting until the episode terminates. This allows for the algorithm to converge faster and be computationally efficient. Moreover, like MC, for each state, TD only takes/samples a single action. That is unlike DP, where we do full action sweep at each step. This further makes it computationally efficient. There are a few different variants of the TD learning algorithm: sarsa, Q-learning, and expected sarasa. And they all differ in how the TD target in their respective update equations is calculated. The q-value update equation for sarsa is given by:

qpi(s,a) = qpi(s,a) + alpha(qpi(s',a') - qpi(s,a)) ..... EQ. 2, where a' is the action taken according to policy pi from state s' and the other parameters are as defined previously.

And the update equation for q-learning is:
qpi(s,a) = qpi(s,a) + alpha(max_a{qpi(s',a)} - qpi(s,a)) ..... EQ. 3, where max_a{qpi(s',a)} is the maximum q-value over all the actions at state s'.

Once the q-values are updated, the policy is inturn updated in the same was as with MC method above, i.e. using eps-greedy policy.

sarsa is an online learning algorithm because for the TD target (i.e. qpi(s',a')), the action a' is chosen based upon the policy we are trying to learn/improve, i.e. eps-greedy policy pi. Whereas Q-learning is an off-policy algorithm becasue the the TD target (i.e. max_a{qpi(s',a)}) is chosen based upon the greedy policy, which is not same as the eps-greedy policy (pi) we are trying to learn. And there are pros and cons to both approaches[1].

The above equations, however only work for tabular world cases where the state space is finite. In continuous environments, discretizing the statespace can quickly run into the curse of dimensionality problem. To address it, we instead use a function approximator to model the q-values. Using a function approximator, we update the weights of the function approximator and so the update equation for Q-learning becomes:

w = qpi(s,a,w) + alpha(max_a{qpi(s',a,w)} - qpi(s,a,w))*grad_w(qpi(s,a,w)) ..... EQ. 4,
where grad_w(qpi(s,a,w)) is the gradient of qpi(s,a,w) with respect to weights w.

And the policy update is same as in the tabular case where for each visited state, with probability eps we select a random action and with probability 1-eps we select an action with the maximum q-value (i.e. greedy policy). For linear function approximators, this approach works rather well in practice. That is the learning algorithm doesn't oscillate and instead converges to the optimal policy. 

### DQN
For nonlinear function approximators like neural networks, the above approach can run into instabilities. To help improve convergence, two modifications can be made and the resulting algorithm is known as Deep Q-Network (DQN) [2].

1. Fixed Q Targets: In the above Q-learning algorithm using a function approximator, the TD target is also dependent on the network parameter w that we are trying to learn/update, and this can lead to instabilities. To address it, a separate network with identical architecture but different weights is used. And the weights of this separate target network are updated every 100 or so steps to be equal to the local network that is continuously being updated.

2. Experience Replay: Updating the weights as new states are being visited is fraught with problems. One is that we don't make use of past experiences. An experience is only used once and then discared. An even worse problem is that there is inherent correlation in the states being visited that needs to be broken; otherwise, the agent will not learn well. Both of these issues are addressed using experience replay where we have a memory buffer where all the experience tuples (i.e. state, action, reward, and next state) are stored. And to break correlation, at each learning step, we randomly sample experiences from this buffer. This also helps us learn from the same experience multiple times, and this is especially useful when encountering some rare experiences.

### Double DQN
DQN is based upon Q-learning algorithm with a deep neural network as the function approximator. However, one issue that Q-learning suffers from is the over estimation of the TD target due to the term max_a{qpi(s',a)}. The expected value of max_a{qpi(s',a)} is always greater than or equal to the max_a of the expected value of qpi(s',a). As a result, Q-learning ends up overstimating the q-values thereby degrading learning efficiency. To address it, we use the double Q-learning algoritm [1] where there are two separate q-tables. And at each time step, we randomly decide which q-table to use and use the argmax a from one q-table to evaluate the q-value of the other q-table. Refer to [1] for more details.

Double DQN [3] is the implementation of double Q-learning using a deep neural network as the function approximator. Note that it is not a direct implementation of double Q-learning using a deep neural network, it is slightly different in terms of how the two networks are used. Refer to [3] and **`model.py`** for implementation details.
**************************

### Agent Training
Having discussed some of the fundamentals, we are now in a position to dive into my implementation of this project.

Just to refresh, the vector observation space is in a 37-dimensional continuous space corresponding to 35 dimensions of ray-based perception of objects around agent's forward direction and 2 dimensions of velocity. The 35 dimensions of ray perception are broken down as: 7 rays projecting from the agent at the following angles (and returned back in the same order): [20, 90, 160, 45, 135, 70, 110] where 90 is directly infront of the agent. Each ray is 5 dimensional and it projected into the scene. If it encounters one of four detectable objects (i.e. good banana, bad banana, wall, agent), the value at that posiiton in the array is set to 1. Finally there is a distance measure which is a fraction of the ray length. Each ray is [Banana, Wall, Bad Banana, Agent, Distance]. For example, [0,1,1,0,0,0.2] means there is a bad banana detected 20% of the distance along the ray with a wall behind it. The velocity of the agent is two dimensional: left/right velocity (usually near 0) and forward/backward velocity (0 to 11.2). Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The observation space is essentially fully observable because it includes information regarding the type of obstacle, the distance to obstable, and the agent's velocity. As a result, we don't need to augment the observations to make it fully observable. Instead, we can directly use the incoming observations as our state representation. The function approximator used is a 3 layer fully connected neural network with the input layer dimension being 37, the first hidden layer being 64 dimensional, the second hidden layer also being 64 dimensional, and the final output layer being 4 dimensional -- for each of the 4 different actions. The model is trained using Stochastic-Gradient Descent algorithm (specifically the Adam optimizer) to update the weights using Equation 4 above. Refer to **`model.py`** for implementation details.

I first tried training the agent using the DQN algorithm, and after training the agent for 1000 episodes, the average reward over 100 episodes is achieved to be around 13. To further improve the model performance, I retrained the agent using the double DQN algorithm. After training for 2000 episodes, the 100 episode reward increased to around 17 as shown below.
 
![DDQN Training][ddqn]

Compared to DQN, the double DQN algortihm resulted in higher rewards and smoother behaviour. Below is a video of the agent's performance using the double DQN algorithm.

![DDQN Test Video][ddqn video]


### Challenge: Training Using Raw Input Pixels
As a challenge, Udacity encouraged us to train the agent directly using raw pixels that the agent "sees." That is with no feature extractor that converts the raw input pixels into a 37 dimensional observation space. Given we are dealing with raw input pixels, the model now needs to be more complex than a simple fully connected network. I implemented the model using three convolutional layers (each followed by maxpool, batch normalization for faster training, and relu nonlinearity). The output of the 3rd convolutional layer (which can be thought of as feature representation) is fed into two fully connected layers. Given the observations are just raw pixes, it is pretty safe to assume it is not the full representation of the environment state, i.e. the state is only partially observable. 

This assumption makes intuitive sense because without using a recurrent network as the input of the overall network, a single frame of pixel cannot (as an example) represent the agent's velocity information. Velocity is the first derivative of distance, and thus we need atleast two adjacent frames (and the corresponding action taken) to represent it. To get acceleration information (which is the second derivative of distance), we need 3 frames, and so on and so forth for higher order derivatives. This helps motivate why we need to augment the input state. While the underlying problem is a POMDP, to approximately turn it into an MDP, I augmented the input observations as follows: From the experience buffer (which is basically a sequential collection of all the raw pixels, actions, next state raw pixels, and rewards) an augmented state is created as follows: augmented_state = [pix_t-1, a_t-1, pix_t, a_t, pix_t+1], where pix_t-1, pix_t, and pix_t+1 are the raw input images from 1 time step earlier, the current time step, and the next time step's observation, respectively. Moreover, a_t-1 and a_t are the actions taken at the prvious time step and the current time step, respectively. This augmented state is what is fed as the input to the CNN, and the network is then trained end-to-end.

This agent was trained using the DQN architecture and the agent was trained for 3000 episodes. Below is the agent performance as it is trained:

![Pixel DQN Training][pixel dqn]

The final trained agent achieves an average reward of about 11 over a course of 100 episodes. The video below shows the performance of this agent when only observing raw input pixels. Refer to model_pixels.py for implementation details.

![Pixel DQN Test Video][dqn pixel video]

As an aside, in theory we can solve this partial observability issue using the Math heavy POMDP framework, but it is computationally intractable for such a high dimensional observation space. Another option is to convert the POMDP problem into an MDP by using the entire history of all observations and actions as our state representation. However, this is also computationally intractable, not to mention the huge amount of memory needed. Given we have some decent idea as to what needs to be included in the state space, e.g. agent's velocity, acceleration, type of obstacles, distance to obstaces etc, and given we have a powerful function approximator, we can get most of the environment's state information using the current, the previous, and the next image frames and their corresponding actions. Hence the above state augmentation methodology was used.


### References:
[1] Sutton and Barto RL Book
[2] Deep Mind DQN paper
[3] Deep Mind Double DQN paper


