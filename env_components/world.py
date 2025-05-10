import argparse
import os

import gymnasium
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from .agents import Animal, Plant
from .base import MetaInformation
from constants import Dim


class World(MetaInformation):
    def __init__(self, config, eval=False, grid_size=None, evolution=False) -> None:
        super().__init__(config, eval, grid_size)
        assert self.config["world"]["max_obs"] % 2 == 1, (
            "self.config['world']['max_obs'] should be odd for spacing reasons in self.observation_spaces"
        )
        
        #3D Numpy array that represnets the state of the world D1: number of observation channels with each channel corresponding to a specific entity D2 and D3: 2 dimensional size of the world
        self.evolution = evolution
        self.state = np.zeros(
            (self.obs_channels, self.grid_size, self.grid_size),
            dtype=np.int64,
        )

        # 2D array representing spots where there are water
        self.water = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)

        # +1 for water
        self.observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1000,
            shape=(self.obs_channels + 1, self.grid_size, self.grid_size),
            dtype=np.int64,
        )

        self.cum_reward = {}            # Note that this is never used
        self.agents_ids = []          # Keys to be used in self.agents
        self.plants_ids = []          # Keys to be used in self.plants
        self.agents = {}                # Represents all animals in the env
        self.plants = {}                # Represents all plants in the env
        # Manages relationships between species, channels, and levels
        self.map_spec_chan = {"p": 0}   # Keys: species identifiers (str) | Values: channel index (int)
        self.map_chan_lev = {}          # Keys: channel indices (int) | Values: integers representing level (int)

        channel = 0
        self.map_spec_chan["p"] = channel       # Todo: is this redundant
        self.map_chan_lev[channel] = 0
        self.map_chan_spe = {0: "p"}

        for lev in range(len(self.species_per_level)):
            for species_id in range(self.species_per_level[lev]):
                channel += 1
                s = f"l{lev + 1}_s{species_id}"
                self.map_spec_chan[s] = channel
                self.map_chan_lev[channel] = lev + 1
                self.map_chan_spe[channel] = s

        self.next_animal_id_per_species = {}
        self.init_next_agent_id()

# ------------------------------------------------------------------

    def reset(self):
        """Resets the environment for a new episode"""
        self.cum_reward = {}            # Note that this is never used
        self.agents_ids = []          # Keys to be used in self.agents
        self.plants_ids = []          # Keys to be used in self.plants
        self.agents = {}                # Represents all animals in the env
        self.plants = {}  
        self.state = np.zeros_like(self.state)
        self.water = self.generate_water(self.grid_size, self.percentage_water)
        self.init_next_agent_id()
        all_pos = self.get_all_poss_agents()
        ag_ids = self.init_agents()
        pl_ids = self.init_plants()
        obs_sp = self.setup_obs_spaces()
        act_sp = self.setup_act_spaces()
        return all_pos, ag_ids, pl_ids, obs_sp, act_sp

# ------------------------------------------------------------------

    def generate_water(self, grid_size, percent_water=8):
        """Generates water"""
        target_nr = int((percent_water / 100.0) * grid_size**2)
        noise = np.random.rand(grid_size, grid_size)
        sm = gaussian_filter(noise, sigma=grid_size*self.water_dispersion)
        flat = sm.flatten()
        threshold = np.partition(flat, target_nr)[target_nr]
        water_mask = sm <= threshold
        return water_mask.astype(np.int64)

# ------------------------------------------------------------------

    def get_all_poss_agents(self):
        """returns ids for all possible agents"""
        all_possible_agents = []
        for lev in range(len(self.species_per_level)):
            for species_id in range(self.species_per_level[lev]):
                for animal_id in range(self.max_animals_per_species):
                    all_possible_agents.append(
                        "l{}_s{}_id{}".format(lev + 1, species_id, animal_id)
                    )

        return all_possible_agents

# ------------------------------------------------------------------

    def init_agents(self):
        """makes id for all agents that start out on the world based on the distribution described in the init file"""
        animals = []
        probs = []
        for lev in range(0, len(self.species_per_level)):
            for spe in range(self.species_per_level[lev]):
                animals.append(f"l{lev + 1}_s{spe}")
                probs.append(self.init_dist[lev] / self.species_per_level[lev])

        alive_animals = np.random.choice(
            animals,
            int(self.grid_size**2 * self.init_amount_animals),
            p=probs / np.array(probs).sum(),
        )
        self._startingAnimal = sorted([str(i) for i in alive_animals])
        
        for idx in self._startingAnimal:
            animal = Animal(world=self, species=idx)
            self.agents[animal.getid()] = animal
            self.agents_ids.append(animal.getid())

        return self.agents_ids

# ------------------------------------------------------------------

    def init_plants(self):
        """makes id for all plants that start out on the world based on the distribution described in the init file"""
        
        nr_plants = int(
            (self.grid_size**2 * self.init_amount_animals) * self.frac_plants_to_animals
        )
        self._startingPlant = ["p" for _ in range(nr_plants)]

        for idx in self._startingPlant:
            plant = Plant(world=self, species=idx)
            self.plants[plant.getid()] = plant

        self.plants_ids = [i for i in self.plants.keys()]
        return self.plants_ids

# ------------------------------------------------------------------

    def init_next_agent_id(self):
        # This keeps for every species the nr of previously existed animals. This is useful for spawning a new animal and giving it a unique id.
        self.next_animal_id_per_species = {}
        for lev in range(len(self.species_per_level)):
            for species_id in range(self.species_per_level[lev]):
                self.next_animal_id_per_species[f"l{lev + 1}_s{species_id}"] = 0

        self.next_plant_id = {"p": 0}

# ------------------------------------------------------------------

    def setup_obs_spaces(self):
        ''' Creates and returns a dictionary of observation spaces for each species
        
        obs_spaces
            Keys: species string
            Values: obs_space object (defines what exactly is observable to a specific species of animal)
        '''

        obs_spaces = {}
        for lev in range(len(self.species_per_level)):
            for species_id in range(self.species_per_level[lev]):
                obs_spaces[f"l{lev + 1}_s{species_id}"] = (self._max_obs_space_any_animal())

        return obs_spaces
    
# ------------------------------------------------------------------

    def _max_obs_space_any_animal(self):
        x = self.config["world"]["max_obs"]
        # +1 for watre
        self.obs_space_world = gymnasium.spaces.Box(
            low=-3, high=1000, shape=(self.obs_channels + 1, x, x), dtype=np.int64
        )
        self.obs_space_health = gymnasium.spaces.Box(
            low=-1, high=1, shape=(int(Dim.HEALTH_SPACE),), dtype=np.float64
        )

        self.obs_space = gymnasium.spaces.Tuple(
            (self.obs_space_world, self.obs_space_health)
        )

        return self.obs_space
    
# ------------------------------------------------------------------

    def setup_act_spaces(self):
        ''' Creates and returns dictionary of action spaces for each species
        
        act_spaces
            Keys: species,
            Values: action space object (see get_act_space_species)

        '''
        act_spaces = {}
        for lev in range(len(self.species_per_level)):
            for species_id in range(self.species_per_level[lev]):
                act_spaces[f"l{lev + 1}_s{species_id}"] = self._get_act_space_species()

        return act_spaces
    
# ------------------------------------------------------------------

    def get_all_levels(self):
        ''' Returns list of unique levels'''
        all_possible = self.get_all_poss_agents()
        return list(
            set([int(idx.split("_")[0].replace("l", "")) for idx in all_possible])
        )

# ------------------------------------------------------------------

    def get_all_species(self):
        ''' Returns list of unique species'''
        all_possible = self.get_all_poss_agents()
        return list(set(["_".join(idx.split("_")[:2]) for idx in all_possible]))

# ------------------------------------------------------------------

    def _get_act_space_species(self):
        # TODO: for now we are going to assume we have self.max_step_size max steps per animal. Later we should update this so different animals (even within a species) can have different amounts of max_step_sizes (maybe also make dependent energy!). One extra for nothing, another for sleep
        # right now: up, left, down, right, nothing. add sleep etc.
        return gymnasium.spaces.Discrete(int(Dim.ACTION_SPACE))

    def make_new_plants(self):
        # self.make_plants_via_reproduction()
        self.make_plants_via_random_locations()

    def make_plants_via_reproduction(self):
        existing_plants = list(self.plants.values())
        for plant in existing_plants:
            if len(self.plants.values()) >= self.max_frac_covered_plants * self.grid_size**2:
                return
            
            plant.reproduce()

    def make_plants_via_random_locations(self):
        """Randomly places plants on the grid with a given probability"""

        x, y = np.where((self.water == 0) & (self.state[0] == 0))
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        for x_val, y_val in zip(x[idx], y[idx]):
            current_nr_plants = len(self.plants.values())
            max_coverage = self.max_frac_covered_plants * self.grid_size**2

            if current_nr_plants >= max_coverage:
                return

            new_plant = Plant(
                world=self,
                species="p",
                position=np.array([x_val, y_val])
            )
            self.plants[new_plant.getid()] = new_plant
            self.plants_ids.append(new_plant.getid())
        
   

# ------------------------------------------------------------------

    def animal_died(self, idx: str = ""):
        """idx: unique id of animal that died"""
        # do minus 1 for position
        self.state[self.agents[idx].obs_idx, *self.agents[idx].position] -= 1

        # remove from lists of agents
        self.agents_ids.remove(idx)

        # remove from list of objects.
        del self.agents[idx]
# ------------------------------------------------------------------

    def plant_died(self, idx: str = ""):
        """idx: unique id of plant that died"""
        self.state[self.plants[idx].obs_idx, *self.plants[idx].position] -= 1

        # remove from lists of plants
        self.plants_ids.remove(idx)

        # remove from list of objects.
        del self.plants[idx]

    def zero_observation(self, idx, obs_style="square"):
        self.level = int(idx.split("_")[0].replace("l", ""))
        self.species = int(idx.split("_")[1].replace("s", ""))

        # TODO: finish this implementation here. For this we need more stuff in the config


# ----------------------------------------------------------------------------------------------------
# Testcode to make sure things are working as intended
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="basic")
    curdir = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()

    with open(f"{curdir}/../config/{args.config}.yaml", "r") as file:
        config = yaml.safe_load(file)
        config["world"]["obs_channels"] = 4

    test = World(config)
    print(test.init_plants())
