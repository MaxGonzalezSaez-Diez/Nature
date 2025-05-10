import argparse
import os
import sys
import time

import ray
import yaml
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# from predpreygrass.rllib.nature.utils import policy_mapping_fn
from ray.rllib.models import ModelCatalog
from ray.tune import RunConfig
from ray.tune.registry import register_env

from callbacks.base import (
    CustomCallback
)
from models.default import NatureCNN
from nature_env import Nature

ModelCatalog.register_custom_model("NatureCNN", NatureCNN)


def env_creator(config):
    return Nature(config)


register_env("Nature", env_creator)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="basic")
    parser.add_argument("--ckpt", type=str, required=False, default="")
    curdir = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()

    with open(f"{curdir}/config/{args.config}.yaml", "r") as file:
        config = yaml.safe_load(file)

    config["run_name"] = config["names"]["raw_run_name"].format(
        **config, config_name=args.config, time=str(time.time()).replace(".", "")
    )

    checkpoint_dir = config["names"]["dir"]
    if checkpoint_dir is None or checkpoint_dir == "./ckpt":
        os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}/ckpt", exist_ok=True)
        checkpoint_dir = f"{os.path.dirname(os.path.realpath(__file__))}/ckpt"
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)

    config["full_checkpoint_dir"] = "{}/{}".format(checkpoint_dir, config["run_name"])
    os.makedirs(config["full_checkpoint_dir"], exist_ok=True)

    with open(os.path.join(config["full_checkpoint_dir"], "config.yaml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Starting run {config['run_name']}")

    ray.shutdown()

    ray.init(
        local_mode=False,
        configure_logging=False,
    )


    wandb_callback = WandbLoggerCallback(
        project=config["names"]["project"],
        name=config["run_name"],
        resume=False,
        reinit=True,
        log_config=True,
        excludes=['perf', 'env_runners', 'learners']
    )


    sample_env = env_creator(config)  

    trainable_policies = []
    for i in range(len(config["trophic_levels"]["species_per_level"])):
        for j in range(config["trophic_levels"]["species_per_level"][i]):
            trainable_policies.append(f"l{i + 1}_s{j}")

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "_".join(agent_id.split("_")[:2])

    # Try restoring from an existing experiment if available
    if len(args.ckpt) > 0 and os.path.isfile(args.ckpt):
        restored_tuner = tune.Tuner.restore(
            path=args.ckpt,  
            resume_errored=True, 
            trainable=PPOConfig().algo_class, 
        )

        # Continue training
        results = restored_tuner.fit()
        ray.shutdown()
        sys.exit()

    # Create a fresh PPO configuration if no checkpoint is found
    ppo = (
        PPOConfig()
        .environment(env="Nature", env_config=config)
        .framework("torch")
        .multi_agent(
            policies_to_train=trainable_policies,
            policy_mapping_fn=policy_mapping_fn,
            policies={
                k: (
                    None,
                    sample_env.observation_spaces[k],
                    sample_env.action_spaces[k],
                    {},
                )
                for k in trainable_policies
            },
        )
        .reporting(
            keep_per_episode_custom_metrics=True
        )
        .training(
            train_batch_size=config["train"]["train_batch_size"],
            gamma=config["train"]["gamma"],
            lr=config["train"]["lr"],
            num_sgd_iter=config['train']['num_sgd_iter'], 
            minibatch_size=config['train']['sgd_minibatch_size'], 
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=NatureCNN,
                model_config={
                    "conv_filters": [
                        [8, [3, 3], 1],
                        [16, [3, 3], 1],
                    ],
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
            )
        )
        .env_runners(
            num_env_runners=config["ss"]["nr_env_runners"],
            num_envs_per_env_runner=config["ss"]["nr_envs_per_env_runner"],
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_cpus_for_main_process=config["ss"]["nr_cpus_for_main_process"],
        num_gpus=config['ss']['num_gpus'],
        num_cpus_per_worker=config['ss']['num_cpus_per_worker'],
        )
        .callbacks(CustomCallback)
    )

    algo = ppo.build()

    # Start a new experiment if no checkpoint is found
    tuner = tune.Tuner(
        ppo.algo_class,
        param_space=ppo,
        run_config=RunConfig(
            storage_path=os.path.abspath(checkpoint_dir),
            name=config["run_name"],
            stop={"training_iteration": config["train"]["training_iterations"]},
            callbacks=[wandb_callback],
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=config["train"]["ckpt_to_keep"],
                checkpoint_frequency=config["train"]["checkpoint_frequency"],
                checkpoint_at_end=True,
            ),
        ),
    )
    # Run the Tuner and capture the results.
    results = tuner.fit()
    ray.shutdown()
