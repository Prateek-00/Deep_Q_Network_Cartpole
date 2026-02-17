import torch 
import torch.nn as nn 
import torch.optim as optim 
import random 
import numpy as np 
from src.model import DQN 
from src.replay_buffer import ReplayBuffer

class DQN_Agent():

    def __init__(self,state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99 # future reward impotance 
        self.epsilon = 1.0 # start full exploration 
        self.epsilon_min = 0.01 # minimum exploration 
        self.epsilon_decay = 0.995 #exploration reduce rate 
        self.learning_rate = 5e-4
        self.batch_size = 64

        # use GPU if availavle else use CPU 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #main learning network 
        self.policy_net = DQN(state_size,action_size).to(self.device)

        #stable reference network 
        self.target_net = DQN(state_size,action_size).to(self.device)

        #intially same weights to policy and target 
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #Target network training mode me nahi hai
        self.target_net.eval()

        #using optiizers for updating weights 
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr= self.learning_rate)

        #replay memory 
        self.memory = ReplayBuffer(10000)

    def select_action(self,state):

        #exploration condition 
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        #exploitation 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():# no gradient calculation during inference 
            q_values = self.policy_net(state)

        return torch.argmax(q_values).item()
    
    def train(self):

        # return if enough memory is not present 
        if len (self.memory)< self.batch_size:
            return 
        
        # random batch samples 
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        #convert numpy array into torch tensors 
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        #current Q values 
        current_q = self.policy_net(states).gather(1,actions)

        #target Q values 
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1-dones) *self.gamma *max_next_q

        #Loss calculation 
        #loss = nn.MSELoss()(current_q,target_q)

        #using Huber loss(more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q,target_q)


        #Backpropagation 
        # Clear previously accumulated gradients from the optimizer
        # PyTorch accumulates gradients by default, so we reset them before backprop
        self.optimizer.zero_grad()

        # Perform backpropagation to compute gradients of the loss
        # with respect to the policy network parameters
        loss.backward()

        #using gradient cliiping for stability 
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),1.0)

        # Update the policy network weights using the computed gradients
        # Applies one optimization step using Adam optimizer
        self.optimizer.step()

    def update_target(self):
        #copy weights from policy to target 
        self.target_net.load_state_dict(self.policy_net.state_dict())