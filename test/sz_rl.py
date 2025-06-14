import json
import os
import requests
from dotenv import load_dotenv
from til_environment import gridworld
import imageio # Already imported!
from datetime import datetime # To make filenames unique

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

NUM_ROUNDS = 8

def main(novice: bool):
    # Ensure render_mode is set to "rgb_array" for visual output
    env = gridworld.env(env_wrappers=[], render_mode="rgb_array", novice=novice)
    env.reset()
    print(env.agents)
    print("Possible agents:", env.possible_agents)
    print("Current agents:", env.agents)
    _agent = env.possible_agents[0] # Assuming this is your agent you want to control
    rewards = {agent: 0 for agent in env.possible_agents}

    # Create a directory for GIFs if it doesn't exist
    output_dir = "evaluation_gifs"
    os.makedirs(output_dir, exist_ok=True)

    for round_idx in range(NUM_ROUNDS):
        # List to store frames for the current round
        frames = []
        print(f"\n=== Starting Round {round_idx + 1} ===")
        env.reset()
        _ = requests.post("http://localhost:5004/reset")

        # Capture the initial frame
        frames.append(env.render())

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            # Convert numpy arrays in observation dict to list for JSON serialization
            observation = {
                k: v if isinstance(v, int) else v.tolist() for k, v in observation.items()
            }

            # Accumulate rewards for all agents in the environment
            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == _agent: # This is the agent your policy controls
                payload = {"instances": [{"observation": observation}]}
                print(f"\nSending observation to RL manager for agent '{agent}':")
                # print(json.dumps(payload, indent=2)) # Uncomment for verbose observation debugging

                response = requests.post("http://localhost:5004/rl", data=json.dumps(payload))
                print("Received response from RL manager:")
                print(response.text)

                predictions = response.json()["predictions"]
                action = int(predictions[0]["action"])
            else: # Other agents (guards) act randomly
                action = env.action_space(agent).sample()

            env.step(action)
            
            # Capture the frame AFTER the step
            frames.append(env.render()) 
            
            if termination or truncation:
                break # Break out of the agent_iter loop if the episode ends

        # Save the captured frames as a GIF for the current round
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_filename = os.path.join(output_dir, f"evaluation_round_{round_idx + 1}_{timestamp}.gif")
        
        # Use imageio.v3.imwrite for modern imageio versions
        # Check your imageio version if this causes issues, you might need imageio.mimsave for older versions.
        imageio.v3.imwrite(gif_filename, frames, duration=25, loop=0) # duration in ms, loop=0 means infinite loop
        print(f"Saved evaluation GIF for Round {round_idx + 1} to: {gif_filename}")


    env.close()
    print(f"\nTotal rewards for controlled agent ({_agent}): {rewards[_agent]}")
    # Adjusting score calculation based on your original script's output
    print(f"Score: {rewards[_agent] / NUM_ROUNDS / 100:.2f}")

if __name__ == "__main__":
    main(TEAM_TRACK == "novice")