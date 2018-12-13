-------------------------------------------------------------------

## Ideas for Future Improvement:
1. Use parameter space noise rather than noise on action. https://vimeo.com/252185862 https://github.com/jvmancuso/ParamNoise
2. We can use prioritised experience buffer. https://github.com/Damcy/prioritized-experience-replay
3. Different replay buffer for actor/critic
4. Try adding dropouts in critic network
5. Turn off OU noise and use random noise
6. You should also try implementing some other algorithms like A3C and PPO. Following are some useful posts.
    [Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
    
    [Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO)](https://medium.com/@sanketgujar95/trust-region-policy-optimization-trpo-and-proximal-policy-optimization-ppo-e6e7075f39ed)
