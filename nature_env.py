import copy
from collections import defaultdict
import json
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, List

from env_components.world import World
from collections import Counter

class Nature(MultiAgentEnv):
    def __init__(self, config, eval=False, grid_size=None, evolution=False):
        super().__init__()

        assert config is not None, "No config provided to world"
        self.config = config

        # max steps
        self.eval = eval
        if self.eval:
            self.max_steps = self.config["train"]["max_steps"] * 10
        else:
            self.max_steps = self.config["train"]["max_steps"] 

        
        self.world = World(config, eval, grid_size, evolution)

        # See: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html#rllib-multi-agent-environments-doc for a description of the required parameters.

        # All possible agents that might exist (required by rllib) - IDs
        self.possible_agents: List[AgentID] = self.world.get_all_poss_agents()

        # Agents that start out on the world - IDs
        self.agents: List[AgentID] = self.world.init_agents()

        # Non-learning agents (grass); not included in 'possible_agents' or 'agents'
        self.plants: List[AgentID] = self.world.init_plants()

        self.observation_spaces = self.world.setup_obs_spaces()
        self.action_spaces = self.world.setup_act_spaces()
        self.dead_animals_id_previous_episode = []

# ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        print("Resetting environment")
        # check: anything to do for random seed?
        super().reset(seed=seed)

        self.current_step = 0        

        # Initialize grid_world_state
        init_tuple = self.world.reset()
        self.possible_agents = init_tuple[0]
        self.agents = init_tuple[1]
        self.plants = init_tuple[2]
        self.observation_spaces = init_tuple[3]
        self.action_spaces = init_tuple[4]

        self.init_nr_unique_species = len(set(['_'.join(agent_id.split("_")[:2]) for agent_id in self.agents]))        

        # Generate observations
        observations = {idx: self.world.agents[idx].observe()[0] for idx in self.agents}

        infos = {idx: {} for idx in observations}
        print(f"Created {len(self.agents)} agents and {len(self.plants)} plants")
        return observations, infos
# ------------------------------------------------------------

    def step(self, action_dict):
        ''' Executes a single step in the environment '''
        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        assert np.min(self.world.state) == 0, "Err in act"

        # STEP 0: remove dead animals from previous episodes animals
        all_agents = copy.deepcopy(self.agents)
        for idx in all_agents:
            if self.world.agents[idx].isDead():
                self.world.animal_died(idx)


        # print(f"Step: {self.current_step}")
        unique_species = len(list(set(['_'.join(agent_id.split("_")[:2]) for agent_id in self.agents])))

        # step 0: check for truncation: See https://discuss.ray.io/t/multi-agent-truncateds-vs-terminateds/12088/5
        if not self.eval:
            if self.current_step >= self.max_steps or len(self.agents) > 150 or len(self.agents) == 0 or unique_species < self.init_nr_unique_species:
                for idx in self.agents:
                    observations[idx], rewards[idx], _, _ = self.world.agents[idx].observe()

                    truncateds[idx] = True
                    terminateds[idx] = False

                truncateds["__all__"] = True
                terminateds["__all__"] = False

                return observations, rewards, terminateds, truncateds, infos

        # Step 1: Process actions.
        agents_copy = copy.deepcopy(self.agents)
        for idx in agents_copy:
            if idx in action_dict:
                action = action_dict[idx]
                self.world.agents[idx].act(action)

        # Step 2: Eat or be eaten
        for idx in self.agents:
            self.world.agents[idx].eat_or_be_eaten()
        for idx in self.plants:
            self.world.plants[idx].being_eaten()

        # Step 3: Process energy depletion due to time steps
        # for idx, action in action_dict.items():
        for idx in self.agents:
            self.world.agents[idx].resource_consumption()
        for idx in self.plants:
            self.world.plants[idx].resource_consumption()

        # Step 4: Generate observations for all agents AFTER all engagements in the step
        for idx in self.agents:
            observations[idx], rewards[idx], terminateds[idx], truncateds[idx] = self.world.agents[idx].observe()

        current_plants = copy.deepcopy(self.plants)
        for idx in current_plants:
            if self.world.plants[idx].isDead():
                self.world.plant_died(idx)

        # Step 5: Create more plants
        self.world.make_new_plants()

        # Step 6: Wake up sleeping agents
        for idx in self.agents:
            self.world.agents[idx].check_if_done_sleeping()

        self.agents.sort()
        self.current_step += 1
        
        terminateds["__all__"] = all(v for v in terminateds.values())
        truncateds["__all__"] = all(v for v in truncateds.values())

        return observations, rewards, terminateds, truncateds, infos
    
# ------------------------------------------------------------

    def get_observation_space(self, agent_id):
        ''' '''
        policy_id = "_".join(agent_id.split("_")[:2])
        return self.observation_spaces[policy_id]
    
# ------------------------------------------------------------
    def get_action_space(self, agent_id):
        policy_id = "_".join(agent_id.split("_")[:2])
        return self.action_spaces[policy_id]
    

if __name__ == "__main__":
    import argparse
    import os

    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="basic")
    curdir = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()

    with open(f"{curdir}/config/{args.config}.yaml", "r") as file:
        config = yaml.safe_load(file)
        config["world"]["obs_channels"] = 4

    env = Nature(config)
    observations, infos = env.reset()
    observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
    summary_dict = defaultdict(int)
    for i in range(2000):
        summary_dict[i] = []
        for agent_id in env.agents:
            summary_dict[i].append(env.world.agents[agent_id].get_summary())


        print(f"Step {i}, {Counter(['_'.join(k.split('_')[:2]) for k in env.world.agents.keys()])}, {len(env.world.plants)}")
        actions = {}
        for idx in env.agents:
            actions[idx] = np.random.randint(0, 7)

        flat_index = np.argmin(env.world.state)
        ll = np.unravel_index(flat_index, env.world.state.shape)[0]

        observations, rewards, terminateds, truncateds, infos  = env.step(actions)

    with open(f'{os.path.dirname(os.path.abspath(__file__))}/output.json', "w") as file:
        json.dump(summary_dict, file)    
