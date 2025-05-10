import gymnasium
import numpy as np
from .base import MetaInformation
from constants import Actions, Dim
import copy

class Animal:
    def __init__(
        self,
        world: "World" = None,
        species: str = "",
        obs_range: int = -1,
        position: np.ndarray = None,
        obs_style: str = "square",
        **kwargs,
    ):
        assert world is not None, "No reference to world provided"

        assert not (id == "" and species == ""), "Neither id nor species specified"

        # Animal Characteristics that remain constant throughout the life of the animal
        self.world = world

        self.species = species

        self.id = self.assign_new_id()

        self.level = int(self.id.split("_")[0].replace("l", ""))
        self.obs_idx = self.world.map_spec_chan[self.species]

        # Animal Characteristics that change throughout the life of the animal
        self.position = copy.deepcopy(position) if position is not None else self.get_rand_position()
        self.isAlive = True
        self.thirst = 0.5  # replenished by eating
        self.hunger = 0.5  # replenished by drinking
        self.energy = 0.5  # replenished by sleeping
        # Other:
        self.cumulative_rewards = 0
        self.age = 0
        self.descendants = 0
        self.animals_eaten = 0

        if obs_style == "square":
            self.observe_world = self.square_window_observation
        
        self.obs_range = obs_range if obs_range != -1 else self.get_obs_range()

        if 'inheritance' in kwargs:            
            self.energy_base_fact = kwargs['energy_base_fact'] 
            self.energy_step_fact = kwargs['energy_step_fact']
            self.hunger_step_fact = kwargs['hunger_step_fact']
            self.thirst_step_fact = kwargs['thirst_step_fact']
            self.energy_repr_fact = kwargs['energy_repr_fact']
            self.generation = kwargs['generation'] + 1
        else:
            self.generation = 1
            self.energy_base_fact = 1
            self.energy_step_fact = 1
            self.hunger_step_fact = 1
            self.thirst_step_fact = 1
            self.energy_repr_fact = 1
        
        self.base_energy_usage = self.energy_base_fact * self.world.baseline_energy_usage
        self.hunger_per_step = self.hunger_step_fact * self.world.hunger_per_step
        self.thirst_per_step = self.thirst_step_fact * self.world.thirst_per_step
        self.energy_cost_reproduction = self.energy_repr_fact * self.world.energy_cost_reproduction
        self.reward_reproduction = self.world.reproduction_reward * self.level + 1
        self.steps_until_reproduction = self.world.min_steps_between_reproduction * self.level

        # put on the world:
        self.world.state[self.obs_idx, *self.position] += 1
        self.this_step = self._init_this_step()

        self.is_asleep = False
        self.sleep_cycles = 0

        assert self.world.state[self.obs_idx, *self.position] > 0, "Animal is not on the map"

    def _init_this_step(self):
        return {
            'fell_asleep': False,
            'tried_sleeping': False,
            'sufficient_energy': False,
            'ate_this_step': False,
            'drank_this_step': False,
            'sufficient_water': False,
            'reproduced_this_step': False,
            'tried_reproducing': False,
            'animal_level_eaten': 0,
        }

    def get_summary(self, spec_only=True):

        if spec_only:
            return {
                "species": self.species,
                "id": self.id,
            }
        else:
            return {
                "species": self.species,
                "id": self.id,
                "position": self.position.tolist(),
                "thirst": self.thirst,
                "hunger": self.hunger,
                "energy": self.energy,
                "isAlive": self.isAlive,
                "age": self.age,
                "descendants": self.descendants,
                "cumulative_rewards": self.cumulative_rewards,
                "is_asleep": self.is_asleep,
                'animals_eaten': self.animals_eaten,
                **self.get_inheritance(False),
            }

    def get_inheritance(self, inherit=False):
        """returns the parameters that should be inherited of the animal. This is used to create new animals with the same inheritance parameters."""
        vals = {
                "energy_base_fact": self.energy_base_fact,
                "energy_step_fact": self.energy_step_fact,
                "hunger_step_fact": self.hunger_step_fact,
                "thirst_step_fact": self.thirst_step_fact,
                "energy_repr_fact": self.energy_repr_fact,
                "generation": self.generation,
            }
        
        if not inherit:
            return vals
        else:
            return {k: np.random.normal(loc=v, scale=self.world.evolution_std) if k != "generation" else v for k, v in vals.items()}

    def getid(self):
        return self.id

    def assign_new_id(self):
        """This assigns eact animal a unique id so we can track them through their existance."""
        new_id = "{}_id{}".format(
            self.species, self.world.next_animal_id_per_species[self.species]
        )
        self.world.next_animal_id_per_species[self.species] += 1
        return new_id
    
    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def reward_now(self):
        """returns current reward. Balances thirst, hunger, energy, etc."""
        if self.isDead():
            return self.world.reward_death

        if self.is_asleep and not self.this_step['fell_asleep']:
            return 0
        elif self.this_step['fell_asleep']:
            return self.energy
        
        penalty = 0
        eaten_animal = 0
        if self.this_step['tried_sleeping'] and self.this_step['sufficient_energy']:
            penalty += self.world.penalty_disallowed_action

        if self.this_step['drank_this_step'] and self.this_step['sufficient_water']:
            penalty += self.world.penalty_disallowed_action

        if self.this_step['ate_this_step']:
            eaten_animal = self.world.reward_eaten 

        if self.energy < self.world.low_energy_threshold and not self.is_asleep:
            penalty += self.world.penalty_not_sleeping_on_low_energy
        
        if not self.this_step['reproduced_this_step'] and self.this_step['tried_reproducing']:
            penalty += self.world.penalty_trying_to_reproduce_too_early

        reproduction = int(self.this_step['reproduced_this_step']) * self.reward_reproduction + int(self.this_step['reproduced_this_step']) * (5 - self.level)

        thirst = np.min(0.5 - self.thirst, 0)
        hunger = np.min(0.5 - self.hunger, 0)
        
        return thirst + hunger + eaten_animal + reproduction - penalty

    # def state(self):
    #     """returns state of animal - thirst, hunger, energy"""
    #     return np.array([self.thirst, self.hunger, self.energy, self.is_asleep])

    def isDead(self):
        """returns whether animal is alive or not"""
        if self.energy <= 0:
            self.isAlive = False

        if self.hasDrowned():
            self.isAlive = False

        return not self.isAlive

    def hasDrowned(self):
        '''Checks if a given animal has drowned'''
        x_dim = self.world.state.shape[1]
        y_dim = self.world.state.shape[2]

        neighbors = [
            ((self.position[0] + dx) % x_dim, (self.position[1] + dy) % y_dim)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        return all([self.world.water[x, y] for x, y in neighbors])


    def get_action_state(self):
        return self.action_space

    def get_obs_range(self):
        # TODO: This should be finished and set up. Based on obs range of animal in the default case.
        return 5

    def get_rand_position(self):    
        x, y = np.where(self.world.water == 0)
        idx = np.random.choice(np.arange(len(x)))
        return np.array([x[idx], y[idx]])

    def resource_consumption(self):
        """Energy consumption of being alive"""
        # magic numbers
        if not self.is_asleep:
            self.energy -= self.base_energy_usage
            self.hunger += self.hunger_per_step + (np.max(self.level - 2, 0) * 0.02)
            self.thirst += self.thirst_per_step
        if self.energy <= 0 or self.hunger >= 1 or self.thirst >= 1:
            self.isAlive = False

    def observe(self):
        state = self.observe_world()
        state = self.make_state_fit(state)
        health = self.animal_state()

        reward = self.reward_now()
        self.cumulative_rewards += reward

        # Observation, reward, termination, truncation
        return (state, health), reward, self.isDead(), False

    def act(self, action):
        
        self.this_step = self._init_this_step()
        self.age += 1

        assert self.world.state[self.obs_idx, *self.position] > 0, "Err in act"

        if self.is_asleep:
            self.sleep_cycles -= 1
            return 

        self.steps_until_reproduction -= 1

        if self.world.eval == True and np.random.random() < 0.05:
            action = Actions.REPRODUCE

        if action == Actions.NOTHING:
            self._do_nothing()
        elif action == Actions.REPRODUCE: 
            self._reproduce()
        elif action == Actions.SLEEP:
            self._sleep()
        else:
            self._move(action) 


    def _do_nothing(self):
        """Animal does nothing."""
        return
    
    def _sleep(self):
        self.this_step['tried_sleeping'] = True

        if self.energy > self.world.low_energy_threshold:
            self.this_step['sufficient_energy'] = True
            return

        self.this_step['fell_asleep'] = True
        self.is_asleep = True
        self.sleep_cycles = np.random.randint(
            self.world.sleep_cycles_range[0], self.world.sleep_cycles_range[1] + 1
        )
        self.energy = 1

    def _move(self, action):       
        old_pos = copy.deepcopy(self.position)
        direction = self.world.action_to_move[action]
        new_pos = (self.position + direction) % self.world.grid_size
        self.position = new_pos

        self.world.state[self.obs_idx, *old_pos] -= 1
        self.world.state[self.obs_idx, *new_pos] += 1

    def _reproduce(self):
        """
        Animal is reproducing in this step
        """
        self.this_step['tried_reproducing'] = True

        if self.energy < self.world.energy_threshold_reproduction or self.steps_until_reproduction > 0:
            return 
            
        x_idx = np.array(
            [
                (self.position[0] + i) % self.world.state.shape[1]
                for i in range(-1, 2)
            ]
        )
            
        y_idx = np.array(
            [
                (self.position[1] + i) % self.world.state.shape[2]
                for i in range(-1, 2)
            ]
        )
        
        X, Y = np.meshgrid(x_idx, y_idx, indexing="ij")
        around = self.world.state[self.obs_idx, X, Y]

        if self.world.state[self.obs_idx, *self.position] > 2 or np.sum(around) > 4:
            return
        
        self.this_step['reproduced_this_step'] = True
        self.steps_until_reproduction = self.world.min_steps_between_reproduction * self.level

        # Spawn new agent of the same species
        animal = Animal(
            world=self.world,
            species=self.species,
            obs_range=self.obs_range,
            position=self.position,
            inheritance=True,
            **self.get_inheritance(self.world.inherit_params or bool(self.world.evolution)),
        )

        self.descendants += 1

        # Reduce parent's energy by the reproduction threshold
        self.energy -= self.energy_cost_reproduction

        self.world.agents[animal.getid()] = animal
        self.world.agents_ids.append(animal.getid())
    
    def check_if_done_sleeping(self):
        if self.sleep_cycles <= 0 and not self.this_step['fell_asleep'] and self.isAlive:
            self.is_asleep = False
            self.sleep_cycles = 0

    def square_window_observation(self):
        x_dim = self.world.state.shape[1]
        y_dim = self.world.state.shape[2]
        x_idx = np.array(
            [
                (self.position[0] + i) % x_dim
                for i in range(-int(self.obs_range), int(self.obs_range) + 1)
            ]
        )
        y_idx = np.array(
            [
                (self.position[1] + i) % y_dim
                for i in range(-int(self.obs_range), int(self.obs_range) + 1)
            ]
        )
        X, Y = np.meshgrid(x_idx, y_idx, indexing="ij")
        
        return np.concat([self.world.state[:, X, Y], (self.world.water[X, Y])[None, :, :]])
    
    def make_state_fit(self, state):
        """Note: This is necessary to fit a given agents observation space with the species' observation space. observation might return a state that is smaller than x, 31, 31 (which is the default) so this function padds the state with -1s."""
        t = self.world._max_obs_space_any_animal()[0]._shape
        padded = np.full(t, -1, dtype=state.dtype)
        s = state.shape

        # padd with -1 around. 
        m_0 = int((t[1] - s[1])/2)
        m_1 = int((t[2] - s[2])/2)
        s_0 = m_0
        e_0 = m_0 + s[1]
        s_1 = m_1
        e_1 = m_1 + s[2]

        padded[:, s_0:e_0, s_1:e_1] = state
        return padded
    
    def animal_state(self):
        return np.array([self.hunger, self.thirst, float(self.energy > self.world.low_energy_threshold), float(self.is_asleep), float(self.steps_until_reproduction > 0), float(self.energy > self.world.energy_threshold_reproduction)], dtype=np.float64)

    def eat_or_be_eaten(self):
        overlap = self.world.state[:, *self.position]
        # first: check if animal was eaten by predator
        for animal_idx in range(self.obs_idx + 1, len(overlap)):
            other_present = overlap[animal_idx] > 0
            predator = (
                self.level
                < self.world.map_chan_lev[animal_idx]
                <= self.level + self.world.levels_down_eat
            )
            if other_present and predator:
                self.isAlive = False
                break

        if self.isDead() or self.is_asleep:
            return 

        # second: check if animal ate prey
        for animal_idx in range(0, self.obs_idx):
            other_present = overlap[animal_idx] > 0
            prey = (
                self.level - self.world.levels_down_eat
                <= self.world.map_chan_lev[animal_idx]
                < self.level
            )
            if other_present and prey:
                # reward: animal captured prey. Reset hunger to 0 or 0.5
                if self.world.state[self.obs_idx, *self.position] > 1:
                    self.hunger = 0.5
                else:
                    self.hunger = 0

                self.this_step['animal_level_eaten'] = self.world.map_chan_lev[animal_idx]
                self.this_step['ate_this_step'] = True
                self.animals_eaten += 1
                break

        if self.world.water[self.position[0], self.position[1]]:
            # Animal is drinking
            self.this_step['drank_this_step'] = True
            if self.thirst < 0.5:
                self.this_step['sufficient_water'] = True
            else:
                self.thirst = 0



class Plant(MetaInformation):
    def __init__(
        self,
        world: "World" = None,
        species: str = "",
        position: np.ndarray = np.array([0, 0]),
    ) -> None:
        """Plant should have some energy that gets reduced if it's being eaten by an animal"""
        self.world = world
        self.position = position
        self.level = 0
        self.obs_idx = self.level
        self.isAlive = True
        self.energy = 1
        self.age = 0
        self.descendants = 0

        self.id = self.assign_new_id(species)
        if np.array_equal(self.position, np.array([0, 0])):
            self.position = self.get_rand_position()

        self.world.state[self.obs_idx, *self.position] += 1


    def assign_new_id(self, sp):
        new_nr = self.world.next_plant_id[sp]
        self.world.next_plant_id[sp] += 1
        return f"{sp}_id{new_nr}"
    
    def get_rand_position(self):
        x, y = np.where(self.world.water == 0)
        idx = np.random.choice(np.arange(len(x)))
        return np.array([x[idx], y[idx]])

    def getid(self):
        return self.id

    def isDead(self):
        """returns whether plant is dead (has been eaten)."""
        return not self.isAlive

    def being_eaten(self):
        self.age += 1
        overlap = self.world.state[:, *self.position]
        # did plant get eaten by animal?
        last_animal_eating_plants = np.sum(self.world.tlevels[: self.world.levels_down_eat]) + 1
        for animal_idx in range(1, last_animal_eating_plants):
            if overlap[animal_idx] > 0:
                self.isAlive = False
                return

    def resource_consumption(self):
        """Energy consumption from being alive"""

        if not self.isDead():  
            # for plants it is fine if this is hardcoded          
            self.energy -= 0.01

            if self.energy <= 0:
                self.isAlive = False

    def reproduce(self):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x = (self.position[0] + dx) % self.world.grid_size
            new_y = (self.position[1] + dy) % self.world.grid_size
            
            if self.world.water[new_x, new_y] == 0 and self.world.state[0, new_x, new_y] == 0 and np.random.random() < self.world.prob_plant_creates_plant:
                self.descendants += 1
                
                new_plant = Plant(
                    world=self.world,
                    species="p",
                    position=np.array([new_x, new_y])
                )
                self.world.plants[new_plant.getid()] = new_plant
                self.world.plants_ids.append(new_plant.getid()) 

    def get_summary(self, spec_only=True):
        if spec_only:
            return {
                "species": 'p',
                "id": self.id,
            }
        else:
            return {
                "species": 'p',
                "id": self.id,
                "position": self.position.tolist(),
                "age": self.age,
                "descendants": self.descendants,
            }



if __name__ == "__main__":
    a = (2, 3)
    b = (4, 2)
    print(np.add(a, b))
