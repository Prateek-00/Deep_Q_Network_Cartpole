import torch #pythorch main python library
import torch.nn as nn #neural network module 

class DQN(nn.Module):  #DQN class inherits from nn.MOdule 
    def __init__(self,state_size,action_size):
        super(DQN,self).__init__() # parent class constructor call

        #Sequential will ensure layers will run one by one 
        self.network = nn.Sequential(

            nn.Linear(state_size,128), # input layer 4 states so 128 neurons 
            nn.ReLU(), # activation function add non non linearity 

            nn.Linear(128,128),# hidden layers for deeper feature learning
            nn.ReLU(), # again activation function 

            nn.Linear(128,action_size) # output layers 2 Q values( left, right )
        )

    def forward(self ,x):
        return self.network(x) # forward input to network and return output 