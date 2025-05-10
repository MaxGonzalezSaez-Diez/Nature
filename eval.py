import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import ray
import torch
import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import utils as util
from nature_env import Nature

def dump_summary(summary, results_file):
    with open(results_file, "a") as file: 
        file.write(json.dumps(summary) + "\n")


if __name__ == "__main__":

    print("Starting Eval run")

    def env_creator(config):
        return Nature(config)

    register_env("Nature", env_creator)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, default=None)
    parser.add_argument("--grid", "-g", required=False, default=20)
    parser.add_argument("--out", "-o", required=False, default=25)
    parser.add_argument("--all_info", "-a", action="store_true", default=False)
    parser.add_argument("--evolution", "-e", action="store_true", default=False)
    args = parser.parse_args()

    with open(f"{args.path}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ckpt = args.path + "/*/checkpoint_*"
    experiment_name = os.path.basename(args.path)
    ckptmodel = glob.glob(ckpt)[0]
    print("Making directories")

    current_dir = os.path.dirname(Path(__file__).resolve())
    os.makedirs(f"{current_dir}/results/{experiment_name}", exist_ok=True)
    results_file = f"{current_dir}/results/{experiment_name}/results_{np.random.randint(0, 100000)}.json"

    print(f"Loading checkpoint from: {os.path.abspath(ckptmodel)}")
    algorithm = Algorithm.from_checkpoint(os.path.abspath(ckptmodel))

    
    env = Nature(config, eval=True, grid_size=(args.grid), evolution=args.evolution)
    obs, _ = env.reset()
    done = False
    eval_iter = 0
    water = np.copy(env.world.water).tolist()
    with open(results_file, "a") as file: 
        file.write(json.dumps(water) + "\n")

    part_summary = []
    limited_info = not args.all_info

    while not done:
        # if eval_iter % 100 == 0:
        print(f"Iter: {eval_iter}")
            
        actions = {}
        agent_summary = []
        plant_summary = []

        # for agent_id, agent_obs in obs.items():
        for agent_id in env.world.agents_ids:
            policy_id = algorithm.config["policy_mapping_fn"](agent_id)
            module = algorithm.get_module(policy_id)
            (state, health) = obs[agent_id]
            if isinstance(state, np.ndarray):
                single_obs = {
                    "obs": (
                        # I added unsqueeze here for the batch dimension
                        torch.from_numpy(state).float().unsqueeze(0),
                        torch.from_numpy(health).float().unsqueeze(0),
                    )
                }
            action_output = module.forward_inference(single_obs, explore=False)
            if Columns.ACTION_DIST_INPUTS in action_output:
                action = torch.argmax(
                    action_output["action_dist_inputs"], dim=-1
                ).item()
                actions[agent_id] = action

            species = util.strip_idNr(agent_id)
            single_summary = env.world.agents[agent_id].get_summary(limited_info)
            single_summary['action'] = action
            agent_summary.append(single_summary)

        for plant_id in env.world.plants_ids:
            plant_summary.append(env.world.plants[plant_id].get_summary(limited_info))

        obs, rewards, dones, _, _ = env.step(actions)
        done = dones["__all__"]
        part_summary.append(
            {
                "iter": eval_iter,
                "agents": agent_summary,
                "plants": plant_summary,
            }
        )

        eval_iter += 1
        if eval_iter % args.out == 0:
            print(f"Saving results to {results_file}")
            water = np.copy(env.world.water)
            dump_summary(part_summary, results_file)
            part_summary = []


    dump_summary(part_summary, results_file)