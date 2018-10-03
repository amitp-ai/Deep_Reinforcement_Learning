##### RANDOM PROCESSES: RANDOM WALK VS OUP ###########
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class RandomWalk(object):
    """Random Walk process."""

    def __init__(self, size, seed, mu=0., sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.sigma * np.array([random.random()-0.5 for i in range(len(x))])
        self.state = x + dx
        return self.state
        
        
class OUNoise(object):
    """Ornstein-Uhlenbeck process. Compared to random walk, this random process forces the samples to be near the mean value."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random()-0.5 for i in range(len(x))])
        self.state = x + dx
        return self.state


OUP = OUNoise(1, seed=0)
RW = RandomWalk(1, seed=0)

OUP_list = []
RW_list = []
for i in range(1000):
    RW_list.append(RW.sample())
    OUP_list.append(OUP.sample())

plt.plot(RW_list)
plt.show()
plt.plot(OUP_list)
plt.show()

##### /END RANDOM PROCESSES: RANDOM WALK VS OUP ###########
