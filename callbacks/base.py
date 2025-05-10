from collections import defaultdict
from ray.rllib.algorithms.callbacks import RLlibCallback
import debugpy
import utils as util


class CustomCallback(RLlibCallback):
    def on_episode_step(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        actions_taken = defaultdict(int)
        all_actions = [action[0].item() for _, action in episode.get_actions().items()]
        for action in all_actions:
            actions_taken[action] += 1

        reward_species = defaultdict(int)
        species_count = defaultdict(int)
        for agent_id, rewards_list in episode.get_rewards().items():
            reward = rewards_list[-1]
            species = util.strip_idNr(agent_id)
            reward_species[species] += reward
            species_count[species] += 1

        metrics_logger.log_dict(
            actions_taken, key="actions_taken", reduce="sum", clear_on_reduce=False
        )
        metrics_logger.log_dict(
            reward_species, key="reward_species", reduce="sum", clear_on_reduce=False
        )
        metrics_logger.log_dict(
            species_count, key="species_count", reduce="sum", clear_on_reduce=False
        )

    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger,
        result,
        **kwargs,
    ) -> None:
        try:
            debugpy.listen(("0.0.0.0", 5678))
            debugpy.wait_for_client()
            debugpy.breakpoint()
        except RuntimeError:
            pass

        reward_species = result["env_runners"]["reward_species"]
        species_count = result["env_runners"]["species_count"]
        rewards_per_species = {
            k: v / species_count[k] for k, v in reward_species.items()
        }

        count_actions = defaultdict(int)
        total = 0
        for action, nr_taken in result["env_runners"]["actions_taken"].items():
            count_actions[action] += nr_taken
            total += nr_taken

        count_actions = {k: v / total for k, v in count_actions.items()}

        metrics_logger.log_dict(
            rewards_per_species, key="species_mean_reward", reduce="mean", window=10
        )
        metrics_logger.log_dict(
            count_actions, key="frac_action_taken", reduce="mean", window=1
        )

        algorithm.config.train_batch_size = algorithm.config.train_batch_size * 2
        algorithm.workers.sync_env_runner_states()
        algorithm.workers.sync_weights()
