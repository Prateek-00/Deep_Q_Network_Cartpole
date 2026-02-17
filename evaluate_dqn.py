#In This script we load our pre  trained DQN model and runs it in a visible CartPole environment .
#so we can watch the trained agent perform.

#Requirements:
#- Model must be saved at: saved_models/dqn_cartpole.pth
#- pygame must be installed (pip install "gymnasium[classic-control]")

#importing necesaary library 
import gymnasium as gym # library of environment setup
import torch # library for model 
from src.agent import DQN_Agent # custom DQNagent

# Creating CartPole Environment (with live window)
# render_mode="human" opens a graphical window for simulation 
env = gym.make("CartPole-v1", render_mode="human")

# Select Seed for Evaluation

print("\n--------------------------------------------------")
print("The model was trained on three different seeds:")
print("  • 42")
print("  • 7")
print("  • 123")
print("--------------------------------------------------")

valid_seeds = [42, 7, 123]

while True:
    try:
        seed = int(input("Enter the seed number you want to evaluate: "))

        if seed in valid_seeds:
            break
        else:
            print("Invalid seed. Please choose from 42, 7, or 123.\n")

    except ValueError:
        print("Please enter a valid numeric seed value.\n")


env.reset(seed=seed)
env.action_space.seed(seed)
torch.manual_seed(seed)


# Get state and action dimensions from environment
state_size = env.observation_space.shape[0]   # 4 state values
action_size = env.action_space.n              # 2 possible actions

# Initializng  Agent
agent = DQN_Agent(state_size, action_size)

# Loading trained weights into policy network
model_path = f"saved_models/best_seed_{seed}.pth"

agent.policy_net.load_state_dict(
    torch.load(model_path, map_location=agent.device)
)

# Set policy network to evaluation mode
agent.policy_net.eval()

# VERY IMPORTANT:
# Disablng the exploration copletely during evaluation
# We want pure greedy action selection so 
agent.epsilon = 0.0

#  Multiple Evaluation Episodes
num_episodes = 20
total_rewards = []

for episode in range(num_episodes):

    # Reset environment at start of each episode
    state, _ = env.reset(seed=seed + episode) #Controlling  randomness + different initial states.
    done = False
    episode_reward = 0

    while not done:

        # Select action using trained policy
        action = agent.select_action(state)

        # Perform action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Episode ends if terminated or truncated
        done = terminated or truncated

        # Move to next state
        state = next_state

        # Accumulate reward
        episode_reward += reward

    print(f"Episode {episode + 1} Reward: {episode_reward}")
    total_rewards.append(episode_reward)

#Calculate and Print Average Reward
average_reward = sum(total_rewards) / num_episodes
print(f"\nAverage Reward over {num_episodes} episodes: {average_reward}")

#Close Environment Properly
env.close()