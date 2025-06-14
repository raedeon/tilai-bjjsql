import torch
import torch.nn as nn
import numpy as np
from til_environment import gridworld
from til_environment.flatten_dict import FlattenDictWrapper
from wrapper import RewardShapingWrapper
from pettingzoo.utils import aec_to_parallel
from model import CNNDQN
import random
from collections import deque
import os
import imageio
import csv

# === Hyperparameters ===
GAMMA = 0.95
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 1000000000

# === Environment Setup ===
def make_env(seed=42):
    return gridworld.parallel_env(
        novice=True,
        render_mode="rgb_array",
        rewards_dict={
            gridworld.RewardNames.SCOUT_RECON: +2.0,
            gridworld.RewardNames.SCOUT_MISSION: +5.0,
            gridworld.RewardNames.SCOUT_CAPTURED: -35.0,
            gridworld.RewardNames.GUARD_CAPTURES: +50.0,
            gridworld.RewardNames.STATIONARY_PENALTY: 0,
            gridworld.RewardNames.WALL_COLLISION: -1.5,
            gridworld.RewardNames.AGENT_COLLIDER: -0.2,
            gridworld.RewardNames.SCOUT_STEP: -0.01,
            gridworld.RewardNames.GUARD_STEP: -0.01,
            gridworld.RewardNames.GUARD_TRUNCATION: -10.0, # From 2
            gridworld.RewardNames.SCOUT_TRUNCATION: +2.0,
        },
        env_wrappers=[RewardShapingWrapper]
    )

# === Observation Encoding ===
def process_observation(obs_dict):
    viewcone = np.array(obs_dict["viewcone"], dtype=np.uint8)
    channels = [(viewcone & (1 << i)) >> i for i in range(8)]
    viewcone_tensor = np.stack(channels, axis=0).astype(np.float32)
    aux = np.array([
        obs_dict["direction"],
        obs_dict["location"][0],
        obs_dict["location"][1],
        obs_dict["scout"],
        obs_dict["step"] / 100.0
    ], dtype=np.float32)
    return viewcone_tensor, aux

# === DQN Agent Setup ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, viewcone, aux, action, reward, next_viewcone, next_aux, done):
        self.buffer.append(((viewcone, aux), action, reward, (next_viewcone, next_aux), done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        viewcones, auxes, actions, rewards, next_viewcones, next_auxes, dones = zip(*[
            (s[0][0], s[0][1], s[1], s[2], s[3][0], s[3][1], s[4]) for s in samples
        ])
        return (
            torch.tensor(np.array(viewcones), dtype=torch.float32),
            torch.tensor(np.array(auxes), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.int64),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_viewcones), dtype=torch.float32),
            torch.tensor(np.array(next_auxes), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# === Main Training Loop ===
env = make_env()
env.reset()
n_agents = len(env.agents)
agent_ids = env.agents
output_dim = env.action_spaces[agent_ids[0]].n

policy_net = CNNDQN(output_dim)
target_net = CNNDQN(output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(MEMORY_SIZE)
epsilon = EPSILON_START
start_episode = 0
best_total_reward = float('-inf')
best_scout_reward = float('-inf')
best_guard_reward = float('-inf')

# Load checkpoint
if os.path.exists("last.pt"):
    print("Loading checkpoint from last.pt...")
    checkpoint = torch.load("last.pt")
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    target_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_episode = checkpoint.get("episode", 0)
    epsilon = checkpoint.get("epsilon", EPSILON_START)
    best_total_reward = checkpoint.get("best_reward", float('-inf'))
    best_scout_reward = checkpoint.get("best_scout_reward", float('-inf'))
    best_guard_reward = checkpoint.get("best_guard_reward", float('-inf'))
    print(f"Resumed from episode {start_episode} with epsilon {epsilon:.4f}")

train_scout_next = True

for episode in range(start_episode, NUM_EPISODES):
    obs, _ = env.reset(seed=random.randint(0, 2**32 - 1))
    done = {agent: False for agent in env.agents}
    state_batch = {}
    learning_reward = 0
    
    all_agents = ['player_0', 'player_1', 'player_2', 'player_3']
    
    if train_scout_next:
        scout_agent = next(agent for agent in env.agents if obs[agent]["scout"] == 1)
        learning_agent_idx = env.agents.index(scout_agent)
    else:
        guard_agents = [agent for agent in env.agents if obs[agent]["scout"] == 0]
        learning_agent_idx = env.agents.index(random.choice(guard_agents))
        
    current_learning_agent = env.possible_agents[learning_agent_idx]
    initial_obs = obs[current_learning_agent]
    role = "Scout" if initial_obs["scout"] == 1 else "Guard"
    frames = []

    #step_count = 0
    while not all(done.values()):
        #print(step_count)
        #step_count += 1
        actions = {}
        for i, agent in enumerate(all_agents):
            viewcone_tensor, aux = process_observation(obs[agent])
            state_batch[agent] = (viewcone_tensor, aux)

            if i == learning_agent_idx:
                if random.random() < epsilon:
                    actions[agent] = random.randint(0, output_dim - 1)
                else:
                    with torch.no_grad():
                        v = torch.tensor(viewcone_tensor).unsqueeze(0)
                        a = torch.tensor(aux).unsqueeze(0)
                        q_vals = policy_net(v, a)
                        actions[agent] = q_vals.argmax().item()
            else:
                if random.random() < 0.2:  # semi-random: mostly forward
                    actions[agent] = random.randint(0, output_dim - 1)
                else:
                    actions[agent] = 0  # biased to go forward

        next_obs, rewards, term, trunc, _ = env.step(actions)
        for i, agent in enumerate(all_agents):
            v1, a1 = state_batch[agent]
            v2, a2 = process_observation(next_obs[agent])
            if i == learning_agent_idx:
                #print(rewards[agent])
                replay_buffer.push(v1, a1, actions[agent], rewards[agent], v2, a2, term[agent] or trunc[agent])
                learning_reward += rewards[agent]
        obs = next_obs
        done = {agent: term[agent] or trunc[agent] for agent in all_agents}
        
        frame = env.render()
        frames.append(frame)

        if len(replay_buffer) >= BATCH_SIZE:
            vc_batch, aux_batch, actions_, rewards_, next_vc_batch, next_aux_batch, dones = replay_buffer.sample(BATCH_SIZE)
            q_vals = policy_net(vc_batch, aux_batch).gather(1, actions_.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_actions = policy_net(next_vc_batch, next_aux_batch).argmax(dim=1)
                next_q_vals = target_net(next_vc_batch, next_aux_batch).gather(1, next_actions.unsqueeze(1)).squeeze()

            expected_q_vals = rewards_ + GAMMA * next_q_vals * (1 - dones)
            loss = nn.MSELoss()(q_vals, expected_q_vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode}, Learning Agent: {current_learning_agent} ({role}), Reward: {learning_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    # Log to CSV
    csv_path = "results.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Episode", "Agent", "Role", "Reward", "Epsilon"])
        writer.writerow([episode, current_learning_agent, role, round(learning_reward, 2), round(epsilon, 4)])
    
    train_scout_next = not train_scout_next
    
    imageio.mimsave(f"gifs/episode{episode}_{current_learning_agent}.gif", frames, fps=5)

    # Rotate which agent is being trained
    learning_agent_idx = (learning_agent_idx + 1) % n_agents

    torch.save({
        "model_state_dict": policy_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "epsilon": epsilon,
        "best_reward": best_total_reward,
        "best_scout_reward": best_scout_reward,
        "best_guard_reward": best_guard_reward
    }, "last.pt")

    # Save best overall model (based on learning agent's own reward)
    if learning_reward > best_total_reward:
        best_total_reward = learning_reward
        torch.save({
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "best_reward": best_total_reward,
            "best_scout_reward": best_scout_reward,
            "best_guard_reward": best_guard_reward
        }, "best.pt")
        print(f"[SAVE] New BEST model @ Ep {episode} | Reward: {learning_reward:.2f} | Role: {role}")

    # Save named checkpoint for high-reward episodes
    if learning_reward >= 350:
        torch.save({
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "reward": learning_reward
        }, f"models/checkpoint_reward{int(learning_reward)}_ep{episode}.pt")
        print(f"[SAVE] Checkpoint saved for reward {learning_reward:.2f} at episode {episode}")

    # Save best Scout model
    if role == "Scout" and learning_reward > best_scout_reward:
        best_scout_reward = learning_reward
        torch.save({
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "best_reward": best_total_reward,
            "best_scout_reward": best_scout_reward,
            "best_guard_reward": best_guard_reward
        }, "best_scout.pt")
        print(f"[SAVE] New BEST SCOUT model @ Ep {episode} | Reward: {learning_reward:.2f}")

    # Save best Guard model
    if role == "Guard" and learning_reward > best_guard_reward:
        best_guard_reward = learning_reward
        torch.save({
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "best_reward": best_total_reward,
            "best_scout_reward": best_scout_reward,
            "best_guard_reward": best_guard_reward
        }, "best_guard.pt")
        print(f"[SAVE] New BEST GUARD model @ Ep {episode} | Reward: {learning_reward:.2f}")
        
    # Save every 500 episodes
    if episode % 500 == 0:
        torch.save({
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "best_reward": best_total_reward,
            "best_scout_reward": best_scout_reward,
            "best_guard_reward": best_guard_reward
        }, f"model_ep{episode}.pt")
        print(f"[SAVE] Periodic model checkpoint at episode {episode}")
