from typing import Dict, Tuple

import numpy as np
from constants import Dim 


class MetaInformation:
    def __init__(
        self,
        config=None,
        eval=False,
        grid_size=None
    ) -> None:
        self.config = config

        self.eval = eval

        if eval and grid_size is not None:
            self.grid_size = int(grid_size)
        else:
            self.grid_size = self.config["world"]["grid_size"]
        
        self.percentage_water = int(self.config["world"]["percentage_water"])
        try:
            self.water_dispersion = float(self.config["world"]["water_dispersion"])
        except KeyError:
            self.water_dispersion = 0.03
        
        self.max_frac_covered_plants = float(
            self.config["world"]["max_frac_covered_plants"]
        )
        self.prob_random_plant_on_grid = float(
            self.config["world"]["prob_random_plant_on_grid"]
        )
        self.prob_plant_creates_plant = float(
            self.config["world"]["prob_plant_creates_plant"]
        )

        if eval:
            self.inherit_params = bool(
                self.config["eval"]["inherit_params"]
            )
            self.evolution_std = float(
                self.config["eval"]["evolution_std"]
            )
        else:
            self.inherit_params = bool(
                self.config["train"]["inherit_params"]
            )
            self.evolution_std = float(
                self.config["train"]["evolution_std"]
            )

        # This should be nr_animals. This allows us to model capturing, fighting, and no reaction between animals
        self.nr_species = np.sum(
            [sp for sp in self.config["trophic_levels"]["species_per_level"]]
        )

        self.levels_down_eat = self.config["trophic_levels"]["levels_down_eat"]
        self.tlevels = self.config["trophic_levels"]["species_per_level"]

        self.baseline_energy_usage = float(
            self.config["trophic_levels"]["baseline_energy_consumption"]
        )
        self.reward_death = float(self.config["trophic_levels"]["reward_death"])
        self.reward_eaten = float(self.config["trophic_levels"]["reward_eaten"])

        _penalty = float(
            self.config["trophic_levels"]["penalty_on_actions"]
        )

        self.penalty_disallowed_action = _penalty
        self.penalty_not_sleeping_on_low_energy = _penalty
        self.penalty_trying_to_reproduce_too_early = _penalty

        self.low_energy_threshold = float(
                self.config["trophic_levels"]["low_energy_threshold"]
            )
        
        self.min_steps_between_reproduction = int(
            self.config["trophic_levels"]["min_steps_between_reproduction"]
        )
        self.hunger_per_step = float(self.config["trophic_levels"]["hunger_per_step"])
        self.thirst_per_step = float(self.config["trophic_levels"]["thirst_per_step"])
        self.energy_threshold_reproduction = float(
            self.config["trophic_levels"]["energy_threshold_reproduction"]
        )
        self.energy_cost_reproduction = float(
            self.config["trophic_levels"]["energy_cost_reproduction"]
        )
        self.sleep_cycles_range = list(
            self.config["trophic_levels"]["sleep_cycles_range"]
        )

        # plus one for plants
        self.obs_channels = self.nr_species + 1

        # furtherst a single step is by any agent
        self.max_step_size = 1

        # Map each action to a movement
        self.action_to_move: Dict[int, Tuple[int, int]] = {
            0: np.array([0, 0]),  # no op
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]),
            5: np.array([0, 0]),  # reproduction
            6: np.array([0, 0]),  # sleeping
        }

        assert len(self.action_to_move) == Dim.ACTION_SPACE, f"Number of actions in action_to_move does not match length of action space. {len(self.action_to_move)} vs {Dim.ACTION_SPACE}"

        # total number of animals on the board initially
        self.init_amount_animals = self.config["world"]["init_amount_animals"]

        # how many species per level
        self.trophic_levels = self.config["trophic_levels"]["levels"]
        self.species_per_level = self.config["trophic_levels"]["species_per_level"]
        self.init_dist = self.config["trophic_levels"]["init_dist"]
        self.max_animals_per_species = self.config["trophic_levels"][
            "max_animals_per_species"
        ]
        self.frac_plants_to_animals = self.config["trophic_levels"][
            "frac_plants_to_animals"
        ]
        self.reproduction_reward = self.config["trophic_levels"]["reproduction_reward"]

        # Check that config file makes sense
        assert len(self.species_per_level) == self.trophic_levels, (
            "Error in config file. len(self.species_per_level) != self.trophic_levels"
        )
