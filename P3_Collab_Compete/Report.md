[image1]: https://user-images.githubusercontent.com/23042512/48657451-cc597f00-e9e5-11e8-8332-bf97ee7da5f8.gif "Trained Agent Perf"
[image2]: https://user-images.githubusercontent.com/23042512/48657452-cc597f00-e9e5-11e8-8776-37a144f24702.png "Trained Agent Scores"
[image3]: https://user-images.githubusercontent.com/23042512/48657452-cc597f00-e9e5-11e8-8776-37a144f24702.png "MADDPG Algorithm"

![Trained Agent][image1]

## Introduction

This report discusses my implementation for the third project in Udacity's Deep Reinforcement Learning Nanodegree. In particular, the objective of this project is to solve the Tennis environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space is 24-dimensional consisting of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically: After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores, and this yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## MADDPG

This environment is quite interesting compared to single agent environments. It requires the training of two separate agents, and the agents need to collaborate under certain situations (like don't let the ball hit the ground) and compete under other situations (like gather as many points as possible). Just doing a simple extension of single agent RL by independently training the two agents does not work very well because the agents are independently updating their policies as learning progresses, the environment appears non-stationary from the viewpoint of any one agent. While we can have non-stationary Markov processes, the convergence guarantees offered by many RL algorithms such as Q-learning requires stationary environments. While there are many different RL algorithms for multi agent settings, for this project I chose to use the Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm [1].

The primary motivation behind MADDPG is that if we know the actions taken by all agents, the environment is stationary even as the policies change since P(s'|s,a<sub>1</sub>,a<sub>2</sub>,&pi;<sub>1</sub>,&pi;<sub>2</sub>) = P(s'|s,a<sub>1</sub>,a<sub>2</sub>) = P(s'|s,a<sub>1</sub>,a<sub>2</sub>,&pi;'<sub>1</sub>,&pi;'<sub>2</sub>) for any &pi;<sub>i</sub> &ne; &pi;'<sub>i</sub>. This is not the case if we do not explicitly condition on the actions of other agents, as done by most traditional RL algorithms [1].

In MADDPG, each agent's critic is trained using the observations and actions from all the agents, whereas each agent's actor is trained using just its own observations. This allows the agents to be effectively trained without requiring other agents' observations during inference (because the actor is only dependent on its own observations). Here is the gist of the MADDPG algorithm [1]:

![MADDPG Algorithm][image3]

For each agent's actor, I used a two-layer neural network with 24 units in the input layer, 256 in the first hidden layer, 128 units in the second hidden layer, and 2 units in the output layer. For each agent's critic, I used a two-layer neural network with 48 units in the input layer, 256 units in the first hidden layer (and the actions are concatenated with the output of the input layer), 128 units in the second hidden layer, and 1 unit in the output layer.

The network was trained using Adam optimizer with elu non-linearity for faster training. I performed a few things to help speed up the learning process: 1) for the first 300 episodes, no learning occurred. The agents were just doing random exploration using the Ornstein-Uhlenbeck (OU) noise process. 2) There after learning started. But for each step in the environment, agent's performed three iterations of learning. Furthermore, the additive noise introduced by the OU process (for better exploration) was gradually decayed down to 50% of the noise amount. 

In terms of hyperparameters used, the actor network's learning rate was 1e-4 and the critic's was 3e-4. This allowed the critic to learn a little faster than the actor since the actor network's learning relies on the critic network. For the target networks, a soft update factor of &tau;=2e-3 was used. A batch size of 256 was used. Additionally, a discount factor of 0.99 was used force the agents to be "cognizant" of their actions' long term consequences. Given the behavior policy used is stochastic, due to the additive OU noise, it helped the network generalize well such that regularization was not needed.

For implementation details, [please refer to my github code for details.](https://github.com/gtg162y/DRLND/blob/master/P2_Continuous_Actions/Continuous_Control_UdacityWorkspace.ipynb).

The agent's learning performance is as shown below.
![MADDPG Algorithm][image2]


References:
[1]: MADDPG paper
