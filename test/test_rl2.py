import json
import os
import requests
from dotenv import load_dotenv
from til_environment import gridworld
import imageio

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

NUM_ROUNDS = 8

def main(novice: bool):
    env = gridworld.env(env_wrappers=[], render_mode="rgb_array", novice=novice)
    env.reset()
    print(env.agents)
    print("Possible agents:", env.possible_agents)
    print("Current agents:", env.agents)
    _agent = env.possible_agents[0]
    rewards = {agent: 0 for agent in env.possible_agents}

    for round_idx in range(NUM_ROUNDS):
        #frames = []
        print(f"\n=== Starting Round {round_idx + 1} ===")
        env.reset()
        _ = requests.post("http://localhost:5005/reset")

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            observation = {
                k: v if isinstance(v, int) else v.tolist() for k, v in observation.items()
            }

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == _agent:
                payload = {"instances": [{"observation": observation}]}
                print(f"\nSending observation to RL manager for agent '{agent}':")
                #print(json.dumps(payload, indent=2))

                response = requests.post("http://localhost:5004/rl", data=json.dumps(payload))
                print("Received response from RL manager:")
                print(response.text)

                predictions = response.json()["predictions"]
                action = int(predictions[0]["action"])
            else:
                action = env.action_space(agent).sample()
            env.step(action)
            #frame = env.render()
            #frames.append(frame)
        #imageio.mimsave(f"episode{round_idx}.gif", frames, fps=1)

    env.close()
    print(f"\nTotal rewards: {rewards[_agent]}")
    print(f"Score: {rewards[_agent] / NUM_ROUNDS / 100:.2f}")

if __name__ == "__main__":
    main(TEAM_TRACK == "novice")
