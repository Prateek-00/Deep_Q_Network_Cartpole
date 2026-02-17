import gymnasium as gym # gymnasium for environment 
import numpy as np #for reward tracking and averaging 
import torch # for saving the model 
import os #for safety purpose 
import random # for reproduibility 
from src.agent import DQN_Agent # import our custom DQN_agent 

#creating cartpole enviornment 
env = gym.make("CartPole-v1")

# Fixing  random seed for reproducibility
seed= 123  # change this manually for different runs 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)

# fetching state size from observation space 
# cartpole has a 4 values as a state 
# [cart_position , cart_velocity, pole_angle, pole_velocity ]

state_size = env.observation_space.shape[0]

# action space has 2 actions 
# 0 for left 
# 1 for right 

action_size = env.action_space.n

# initializing  our agent 
agent = DQN_Agent(state_size,action_size)

#we start training here 

episodes = 1000 # mximum training episodes 
reward_history = [] #we are storing rewards for plotting the graphs 

#step based target updating 
global_step = 0
target_update_steps = 200

print("Starting DQN Training...\n")

#just in case if folder does not exist 
os.makedirs("saved_models", exist_ok=True)

# Track best model based on avg reward
best_avg_reward = -float("inf")

# MAIN TRAINING LOOP 
for episode in range (episodes):

    #reset the enviornment at start of each episode 
    state, _ = env.reset()
    
    done = False
    total_reward = 0 # track total reward for this episodes 

    while not done:
        
        #selecion of action 
        action  = agent.select_action(state)

        #taking action in Enviornment
        next_state, reward, terminated, truncated, _ = env.step(action)

        #episode ends if truncated or terminated 
        done = terminated or truncated
        
        #storing the experience in memory 
        agent.memory.store(state, action, reward, next_state, done)

        # training the agent 
        agent.train()

        #move to the next state 
        state = next_state

        #accumulating the reward 
        total_reward += reward

        # Step-based target update
        global_step += 1
        if global_step % target_update_steps == 0:
            agent.update_target()
    
    #storing the episode rewards 
    reward_history.append(total_reward)

    # Rolling average calculation
    if len(reward_history) >= 100:
        avg_reward = np.mean(reward_history[-100:])
    else:
        avg_reward = np.mean(reward_history)

    #printing the episodes result 
    print(f"Episode {episode+1} | Reward: {total_reward} | Avg(100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")


    # Saving the best performing model
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(
            agent.policy_net.state_dict(),
            f"saved_models/best_seed_{seed}.pth"
        )
    
    #Episode-level epsilon decay
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    #early stopping conditions 
    if len(reward_history) >=100 and avg_reward >=475 :
        print("\nEnvironment Solved! 🎉")
        print(f"Average Reward (last 100 episodes): {avg_reward}")
        break

#saving reward history for plotting 
np.save(f"rewards_seeds_{seed}.npy",reward_history)

print("\nTraining Complete. Best saved model details given below")
print(f"Best Avg(100): {best_avg_reward:.2f}")

#close enviornment 
env.close()