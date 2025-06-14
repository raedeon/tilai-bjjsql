# docker run -p 5004:5004 bjjsql-rl:v0

"""Manages the RL model."""
import torch
import numpy as np
from typing import Tuple, List, Dict, Union
#from model import DQN
from model import CNNDQN
from gymnasium.spaces import flatten, Dict as DictSpace, Discrete, Box
from wrapper import RewardShapingWrapper

class RLManager:
    def __init__(self):
        # Initialise single shared DQN policy
        obs_dim, act_dim = 144, 5
        #self.policy = DQN(obs_dim, act_dim)
        self.policy = CNNDQN(act_dim)

        # Load trained model weights
        # checkpoint_reward201_ep7011
        
        checkpoint = torch.load("best.pt", map_location=torch.device("cpu"))
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()

        # Sample observation space needed for flattening
        from til_environment import gridworld
        from til_environment.flatten_dict import FlattenDictWrapper

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
        }

        env = gridworld.parallel_env(
            novice=True,
            render_mode=None,
            rewards_dict=rewards_dict,
            env_wrappers=[RewardShapingWrapper
                          , FlattenDictWrapper]
        )
        env.reset()
        agent_ids = env.agents
        self.obs_space = env.unwrapped.observation_space(agent_ids[0])

    def rl(self, observation: Dict[str, Union[List[List[int]], int, List[int]]]) -> int:
        """Returns next action given an observation."""
        #flat_obs = flatten(self.obs_space, observation)
        #obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

        #with torch.no_grad():
            #logits = self.policy(obs_tensor)
            #action = torch.argmax(logits, dim=-1).item()
        #return action
        
        def process_observation(obs):
            viewcone = np.array(obs["viewcone"], dtype=np.uint8)
            channels = [(viewcone & (1 << i)) >> i for i in range(8)]
            viewcone_tensor = np.stack(channels, axis=0).astype(np.float32)  # shape (8, 7, 5)

            aux = np.array([
                obs["direction"],
                obs["location"][0],
                obs["location"][1],
                obs["scout"],
                obs["step"] / 100.0
            ], dtype=np.float32)

            return viewcone_tensor, aux
        
        viewcone_tensor, aux_tensor = process_observation(observation)

        viewcone_tensor = torch.tensor(viewcone_tensor, dtype=torch.float32).unsqueeze(0)  # shape (1, 8, 7, 5)
        aux_tensor = torch.tensor(aux_tensor, dtype=torch.float32).unsqueeze(0)            # shape (1, 5)

        with torch.no_grad():
            logits = self.policy(viewcone_tensor, aux_tensor)
            action = torch.argmax(logits, dim=-1).item()
        return action
