# Deep Q-Network (DQN) from Scratch — CartPole-v1

In this project, I implemented a **Deep Q-Network (DQN)** agent from scratch using **PyTorch** to solve the **CartPole-v1** environment from Gymnasium.

The goal of this project was not just to run an RL algorithm, but to **understand how DQN actually works internally** by coding every important component manually.

---

## 📌 Problem Statement

CartPole is a simple control problem where:

- A pole is attached to a moving cart.
- The cart can move **left or right**.
- The goal is to **keep the pole balanced upright** as long as possible.

The agent starts with no knowledge and learns balancing only through **trial-and-error interaction** with the environment.

---

## 🧠 What I Implemented

This project includes a full DQN pipeline built manually:

- Neural Network to approximate Q-values
- Experience Replay Buffer
- Target Network for stable learning
- Epsilon-Greedy exploration strategy
- Training loop with batch learning
- Evaluation script for trained model
- Multi-seed training to check consistency

No RL libraries like Stable-Baselines were used — everything is written from scratch.

---

## ⚙️ How the Algorithm Works (Simple Explanation)

1️⃣ The agent observes the current state from the environment.

2️⃣ The neural network predicts Q-values for both actions:


Move Left | Move Right


3️⃣ Using epsilon-greedy policy:
- Sometimes it **explores** (random action)
- Sometimes it **uses what it learned** (best action)

4️⃣ The experience is stored in memory:


(state, action, reward, next_state, done)


5️⃣ A random batch is sampled from memory to train the network.

6️⃣ A separate **Target Network** is updated periodically to keep learning stable.

---

## 🔁 Why Experience Replay Was Used

If we train only on recent steps:
- Data becomes highly correlated
- Learning becomes unstable

Replay Buffer fixes this by:
- Mixing old and new experiences
- Making training more stable and efficient.

---

## 🎯 Why a Target Network Was Needed

If we update and predict using the same network, the learning target keeps changing, which makes training unstable.

So we compute targets using a fixed network:



Target Q = reward + gamma * max(Q_target(next_state))


This helps stabilize training.

---

## 📈 Training Setup

- Episodes: 500+
- Discount Factor (gamma): 0.99
- Batch Size: 64
- Optimizer: Adam
- Loss Function: MSE Loss
- Epsilon decayed gradually to reduce exploration over time.

---

## 🧪 Multi-Seed Experiment

The model was trained using different random seeds to verify learning consistency:

| Seed | Purpose |
|------|---------|
| 7    | Initial run |
| 42   | Stability check |
| 123  | Reproducibility check |

This ensures results are not dependent on lucky initialization.

---

## 📊 Training Behavior Observed

- Early episodes → Mostly random movements  
- Middle phase → Agent starts learning corrective actions  
- Later episodes → Agent balances pole consistently  

Learning progression:


Random → Learning Corrections → Stable Balance


---

## 📂 Project Structure
```

Deep_Q_Network_CartPole/
│
├── src/
│ ├── agent.py
│ ├── model.py
│ └── replay_buffer.py
│
├── saved_models/
│
├── train_dqn.py
├── evaluate_dqn.py
├── notebook.ipynb
│
├── rewards_seeds_7.npy
├── rewards_seeds_42.npy
├── rewards_seeds_123.npy
│
├── .gitignore
└── README.md

```
---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository
```
git clone https://github.com/YOUR_USERNAME/Deep_Q_Network_CartPole.git
cd Deep_Q_Network_CartPole

```
2️⃣ Create Virtual Environment

```
python -m venv dqn_env
dqn_env\Scripts\activate

```
3️⃣ Install Dependencies

```
pip install torch gymnasium numpy matplotlib

```
4️⃣ Train the Model

```
python train_dqn.py

```

5️⃣ Evaluate the Model

```
python evaluate_dqn.py

```

🛠️ What I Learned From This Project

Reinforcement Learning works very differently from supervised learning.

Stability is one of the biggest challenges in RL.

Experience Replay is essential for meaningful learning.

Target Networks prevent divergence during training.

Hyperparameters significantly affect performance.

Implementing from scratch gives deeper understanding than using prebuilt libraries.

👨‍💻 Author

Prateek Shukla
