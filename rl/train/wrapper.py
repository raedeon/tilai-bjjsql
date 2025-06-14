import functools
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import AECEnv, ActionType, AgentID, ObsType


class RewardShapingWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        super().__init__(env)
        self.position_history = {}  # Track recent positions for each agent
        self.survival_steps = {}
        self.visited_tiles = {}

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.position_history = {}
        self.survival_steps = {}
        self.visited_tiles = {}

    def step(self, action: ActionType):
        agent = self.agent_selection
        obs = self.observe(agent)
        current_loc = tuple(obs["location"])
        step = obs["step"]
        #print(f"Action: {action}")

        # === Apply actual action ===
        super().step(action)
        
        # === Skip if agent is done or missing ===
        if agent not in self.rewards or agent not in self.agents:
            return

        history = self.position_history.get(agent, [])
        visited = self.visited_tiles.get(agent, set())
        
        # === Scout reward shaping ===
        if obs["scout"] == 1:
            # A. Penalise standing still
            if len(history) >= 1 and current_loc == history[-1]:
                self.rewards[agent] -= 0.8

            # B. Penalise revisiting previous tile
            elif len(history) >= 2 and current_loc == history[-2]:
                self.rewards[agent] -= 0.1

            # C. Bonus if near recon/mission tiles
            if self.near_tile(obs["viewcone"], tile_types={2}):  # Recon
                pass
                #self.rewards[agent] += 0.05
            if self.near_tile(obs["viewcone"], tile_types={3}):  # Mission
                self.rewards[agent] += 0.15

            # D. Penalise if guard seen in viewcone
            if self.guard_visible(obs["viewcone"]):
                self.rewards[agent] -= 0.15

            # E. Survival bonus every 10 steps
            prev_steps = self.survival_steps.get(agent, 0)
            if (step // 10) > (prev_steps // 10):
                self.rewards[agent] += 0.3
            self.survival_steps[agent] = step
            
            # F. Tile visit reward shaping
            if current_loc not in visited:
                self.rewards[agent] += 0.08
                visited.add(current_loc)
            else:
                self.rewards[agent] -= 0.05

            self.visited_tiles[agent] = visited

        
        # === Guard reward shaping ===
        else:
            # === A. Penalise back-and-forth ===
            if len(history) >= 1 and current_loc == history[-1]:
                self.rewards[agent] -= 0.9  # Stayed in same position
            elif len(history) >= 2 and current_loc == history[-2]:
                self.rewards[agent] -= 0.8  # A → B → A pattern
            elif len(history) >= 1 and current_loc != history[-1]:
                self.rewards[agent] += 0.4  # Moved to a new tile
                
            # B. Track visited tiles
            if current_loc not in visited:
                self.rewards[agent] += 1
                visited.add(current_loc)
            else:
                self.rewards[agent] -= 0.1

            # C. Bonus for seeing scout
            if self.scout_visible(obs["viewcone"]):
                self.rewards[agent] += 0.5
                
            self.visited_tiles[agent] = visited
            
            # D. Penalise unnecessary turning
            #if action in [2, 3]:  # turn left or turn right
                #self.rewards[agent] -= 0.1

            # E. Bonus for valid forward move
            #if action == 0:
                #self.rewards[agent] += 0.1
                
            # F. Manhattan distance between a Guard and the Scout
            if self.scout_visible(obs["viewcone"]):
                distance = self.env.get_info(agent)["manhattan"]
                prev_distance = self.survival_steps.get(agent, distance)

                if distance < prev_distance:
                    self.rewards[agent] += 0.3
                elif distance > prev_distance:
                    self.rewards[agent] -= 0.2

                self.survival_steps[agent] = distance
                
        self.detect_and_penalise_guard_competition(obs, agent)
                
        # Update position history
        history.append(current_loc)
        if len(history) > 3:
            history = history[-3:]
        self.position_history[agent] = history

    def observe(self, agent: AgentID) -> ObsType:
        return super().observe(agent)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return super().observation_space(agent)

    # === Public helper methods (no underscore) ===
    def scout_visible(self, viewcone) -> bool:
        for row in viewcone:
            for tile in row:
                if tile & 0b00000100:  # Bit 2 → scout
                    return True
        return False

    def guard_visible(self, viewcone) -> bool:
        for row in viewcone:
            for tile in row:
                if tile & 0b00001000:  # Bit 3 → guard
                    return True
        return False

    def near_tile(self, viewcone, tile_types: set[int]) -> bool:
        for row in viewcone:
            for tile in row:
                if (tile & 0b11) in tile_types:
                    return True
        return False

    def detect_and_penalise_guard_competition(self, obs, agent):
        # Only run once, when the scout is terminated
        if obs["scout"] == 1 and self.env.terminations.get(agent, False):
            scout_loc = tuple(obs["location"])
            capturers = []

            for other in self.env.agents:
                if other == agent:
                    continue
                other_obs = self.env.observe(other)
                if other_obs["scout"] == 0:
                    other_loc = tuple(other_obs["location"])
                    dist = abs(scout_loc[0] - other_loc[0]) + abs(scout_loc[1] - other_loc[1])
                    if dist == 1:
                        capturers.append(other)

            # Apply penalty to all non-capturing guards
            for other in self.env.agents:
                other_obs = self.env.observe(other)
                if other_obs["scout"] == 0 and other not in capturers:
                    self.rewards[other] -= 5.0
