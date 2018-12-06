#MADDPG
def seeding(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, full_state_size, full_action_size, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of both agents states
            action_size (int): Dimension of both agents actions
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+full_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_state, full_action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.elu(self.fcs1(full_state))
        x = torch.cat((xs, full_action), dim=1)
        x = F.elu(self.fc2(x))
        return self.fc3(x)


class MADDPG(object):
    '''The main class that defines and trains all the agents'''
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE) # Replay memory
        self.maddpg_agents = [DDPG(state_size, action_size), DDPG(state_size, action_size)] #create agents
        
        
    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def step(self, full_states, full_actions, full_rewards, full_next_states, full_dones):
        #for stepping maddpg
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # index 0 is for agent 0 and index 1 is for agent 1
        full_states = np.reshape(full_states, newshape=(1,-1)) #agent0 is 0:state_size, agent1 is state_size:state_size*2
        full_actions = np.reshape(full_actions, newshape=(1,-1)) #agent0 is 0:action_size, agent1 is action_size:action_size*2
        full_rewards = np.reshape(full_rewards, newshape=(1,-1)) #agent0 is 0, agent1 is 1
        full_next_states = np.reshape(full_next_states, newshape=(1,-1)) #agent0 is 0:state_size, agent1 is state_size:state_size*2
        full_dones = np.reshape(full_dones, newshape=(1,-1)) #agent0 is 0, agent1 is 1

        # Save experience / reward
        self.memory.add(full_states, full_actions, full_rewards, full_next_states, full_dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)    
            
            
    def learn_maddpg(self, experiences, gamma):
        #for learning MADDPG ver 1
        # index 0 is for agent 0 and index 1 is for agent 1
        full_states, full_actions, full_rewards, full_next_states, full_dones = experiences
        
        #zero out the gradients first
        for agent in self.maddpg_agents:
            agent.zero_out_grads()
        
        critic_full_next_actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            strt = self.state_size*agent_id
            stp = strt + self.state_size
            agent_next_state = full_next_states[:,strt:stp]
            critic_full_next_actions.append(agent.actor_target.forward(agent_next_state))
        critic_full_next_actions = torch.cat(critic_full_next_actions, dim=1)

        retain_graph_value = True
        for agent_id, agent in enumerate(self.maddpg_agents):
            strt = self.state_size*agent_id
            stp = strt + self.state_size
            agent_state = full_states[:,strt:stp]

            strt = self.action_size*agent_id
            stp = strt + self.action_size
            actor_full_actions = full_actions.clone() #create a deep copy
            actor_full_actions[:,strt:stp] = agent.actor_local.forward(agent_state)
            
            experiences = (full_states, actor_full_actions, full_actions, full_rewards[:,agent_id], \
                            full_dones[:,agent_id], full_next_states, critic_full_next_actions)
            
            if agent_id == 1: retain_graph_value = False #don't retain graph after second agent
            agent.learn(experiences, gamma, retain_graph_value)
            
        for agent in self.maddpg_agents:
            agent.update_parameters() #update both agents' parameters after first computing both agents' gradients


        

    def step_double_ddpg(self, full_states, full_actions, full_rewards, full_next_states, full_dones):
        #stepping for double ddpg. 
        # But not not sure if it will work because for the same state, the two agents actions will be different as well as their rewards. But it does work indeed.
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # index 0 is for agent 0 and index 1 is for agent 1
        for agent_id in range(self.num_agents):
            state = full_states[agent_id,:].reshape(1,-1)
            action = full_actions[agent_id,:].reshape(1,-1)
            reward = full_rewards[agent_id]
            next_state = full_next_states[agent_id,:].reshape(1,-1)
            done = full_dones[agent_id]
            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            self.learn(GAMMA)
            
    def learn_double_ddpg(self, gamma):
        #for learning double DDPG
        # index 0 is for agent 0 and index 1 is for agent 1
        for agent_id, agent in enumerate(self.maddpg_agents):            
            agent.learn(self.memory.sample(), gamma)

    def learn_double_ddpg_2(self, experiences, gamma):
        #for learning Doouble DDPG ver 2
        # index 0 is for agent 0 and index 1 is for agent 1
        full_states, full_actions, full_rewards, full_next_states, full_dones = experiences

        for agent_id, agent in enumerate(self.maddpg_agents):
            strt = self.state_size*agent_id
            stp = strt + self.state_size
            agent_state = full_states[:,strt:stp]
            agent_next_state = full_next_states[:,strt:stp]

            strt_a = self.action_size*agent_id
            stp_a = strt_a + self.action_size
            agent_action = full_actions[:,strt_a:stp_a]
            
            agent_reward = full_rewards[agent_id]
            agent_done = full_dones[agent_id]
            
            experiences = (agent_state, agent_action, agent_reward, agent_next_state, agent_done)
            agent.learn(experiences, gamma)
  
    def learn(self, experiences, gamma):
        #for learning Double DDPG ver 3
        # use with step_maddpg
        # index 0 is for agent 0 and index 1 is for agent 1
        full_states, full_actions, full_rewards, full_next_states, full_dones = experiences

        full_next_actions = []
        full_curr_actions = []
        for agent in self.maddpg_agents:
            with torch.no_grad():
                full_next_actions.append(agent.actor_target.forward(full_next_states))
                full_curr_actions.append(agent.actor_local.forward(full_states))
        full_next_actions = torch.cat(full_next_actions, dim=1)
        full_curr_actions = torch.cat(full_curr_actions, dim=1)
        
        losses = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            agent_reward = full_rewards[agent_id]
            agent_done = full_dones[agent_id]
            
            experiences = (full_states, full_actions, agent_reward, full_next_states, full_curr_actions, full_next_actions, agent_done)
            losses.append(agent.learn(experiences, gamma))


         # Minimize the loss
        for agent_id, agent in enumerate(self.maddpg_agents):
            agent.critic_optimizer.zero_grad()
            agent.actor_optimizer.zero_grad()

        for agent_id, agent in enumerate(self.maddpg_agents):
            losses[agent_id][0].backward(retain_graph=True) #critic loss
            losses[agent_id][1].backward(retain_graph=True) #actor loss

        for agent_id, agent in enumerate(self.maddpg_agents):
            agent.critic_optimizer.step()
            agent.actor_optimizer.step()
            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target, TAU)
            agent.soft_update(agent.actor_local, agent.actor_target, TAU)     

            
    def act(self, full_states):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            #action = agent.act(np.expand_dims(full_states[agent_id,:], axis=0))
            action = agent.act(full_states.reshape(1,-1)) #for fully double ddpg
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions

    def save_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local_' + str(agent_id) + '.pth')

    def load_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agents):
            #Since the model is trained on gpu, need to load all gpu tensors to cpu:
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))

            agent.eps = EPS_END #initialize to the final epsilon value upon training


class DDPG(object):
    """Interacts with and learns from the environment.
    There are two agents and the observations of each agent has 24 dimensions. Each agent's action has 2 dimensions.
    Will use two separate actor networks (one for each agent using each agent's observations only and output that agent's action).
    The critic for each agents gets to see the actions and observations of all agents. """
    
    def __init__(self, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state for each agent
            action_size (int): dimension of each action for each agent
        """
        self.state_size = state_size
        self.action_size = action_size
        
        network_adjustmment = 2 #1 for double DDPG and 2 for MADDPG

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(network_adjustmment*state_size, action_size).to(device)
        self.actor_target = Actor(network_adjustmment*state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(network_adjustmment*state_size, network_adjustmment*action_size).to(device)
        self.critic_target = Critic(network_adjustmment*state_size, network_adjustmment*action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_critic)

        # Noise process
        self.noise = OUNoise(action_size) #single agent only
        self.eps = EPS_START
    
        # Make sure target is initialized with the same weight as the source (found on slack to make big difference)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        #add noise according to epsilon probability
        if add_noise and (np.random.random() < self.eps):
            #actions += self.noise.sample()
            actions = 2*np.random.rand(1,self.action_size) - 1
            
            #update the exploration parameter
            self.eps -= EPS_DECAY
            if self.eps < EPS_END:
                self.eps = EPS_END
            #self.noise.reset() #not sure if need to do this here

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn_maddpg(self, experiences, gamma, retain_graph_value):
        #for MADDPG
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        full_states, actor_full_actions, full_actions, rewards, dones, full_next_states, critic_full_next_actions = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get Q values from target models
        Q_targets_next = self.critic_target(full_next_states, critic_full_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        #self.critic_optimizer.zero_grad() #don't zero out here
        critic_loss.backward(retain_graph=retain_graph_value)
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        #self.critic_optimizer.step() #don't update here

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local.forward(full_states, actor_full_actions).mean() #-ve b'cse we want to do gradient ascent
        # Minimize the loss
        #self.actor_optimizer.zero_grad() #don't zero out here
        actor_loss.backward(retain_graph=retain_graph_value)
        #self.actor_optimizer.step() #don't update here

        # ----------------------- update target networks ----------------------- #
        #self.soft_update(self.critic_local, self.critic_target, TAU) #don't update here
        #self.soft_update(self.actor_local, self.actor_target, TAU) #don't update here                    


    def zero_out_grads(self):
        #for maddpg
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        
    def update_parameters(self):
        #for maddpg
        
        #update actor and critic networks
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        
        # update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)  
        

    def learn_ddpg_ver1(self, experiences, gamma):
        #for double ddpg
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0) #clip the gradient for the critic network (Udacity hint)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     


    def learn(self, experiences, gamma):
        #for fully double ddpg
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, actions_pred, actions_next, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()

        return (critic_loss, actor_loss)
                

        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
