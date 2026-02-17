import random   # for randomness in sampling 
import numpy as np # for mathematical task handling 
from collections import deque # efficient queue Structure 

class ReplayBuffer():

    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity) # fixed size memory old data auto delete 

    def store(self, state, action, reward, next_state, done):
        # storing transistion in memory 
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size) # selecting the random batch 

        states, actions, rewards ,next_states, dones = zip(*batch)
        # zip unpacks the tuples in separate array

        return(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.memory) # returns how many samples in memory 