from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis import (
    TARGET_NETWORK_ACTION_DIST_INPUTS,
    TargetNetworkAPI,
    ValueFunctionAPI,
)
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.misc import (
    same_padding,
    valid_padding,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
import time


class NatureCNN(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """This is adapted from: https://github.com/ray-project/ray/blob/master/rllib/examples/rl_modules/classes/tiny_atari_cnn_rlm.py"""

    @override(TorchRLModule)
    def setup(self):
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.model_config`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).
        """
        # Get the CNN stack config from our RLModuleConfig's (self.config)
        # `model_config` property:
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        grid_shape = self.observation_space.spaces[0].shape
        health_shape = self.observation_space.spaces[1].shape
        in_depth, width, height = grid_shape
        og_in_depth = in_depth

        conv_filters = self.model_config.get("conv_filters")
        # Default CNN stack with 3 layers:
        if conv_filters is None:
            # num filters, kernel wxh, stride wxh, padding type
            conv_filters = [
                [8, [3, 3], 1, "same"],
                [16, [3, 3], 1, "same"],
            ]

        # Build the CNN layers.
        layers = []
        in_size = [height, width]

        for filter in conv_filters:
            if len(filter) == 4:
                out_depth, kernel_size, strides, padding = filter
            else:
                # let's have this as default just in case
                out_depth, kernel_size, strides = filter
                padding = "same"

            # Pad like in tensorflow's SAME mode.
            if padding == "same":
                padding_size, out_size = same_padding(in_size, kernel_size, strides)
                layers.append(nn.ZeroPad2d(padding_size))
            else:
                out_size = valid_padding(in_size, kernel_size, strides)

            layer = nn.Conv2d(in_depth, out_depth, kernel_size, strides, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())

            in_size = out_size
            in_depth = out_depth

        self._base_cnn_stack = nn.Sequential(*layers).to(self.device)

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, og_in_depth, in_size[0], in_size[1], device=self.device
            )
            cnn_out = self._base_cnn_stack(dummy_input)
            cnn_flat_size = int(
                torch.prod(torch.tensor(cnn_out.shape[1:], device=self.device))
            )

        self._vector_net = nn.Sequential(
            nn.Linear(health_shape[0], 32),
            nn.ReLU(),
        ).to(self.device)

        combined_shape = cnn_flat_size + 32

        fc_dims = self.model_config.get("fcnet_hiddens", [256, 256])

        policy_layers = []
        last = combined_shape
        for hidden_dim in fc_dims:
            policy_layers.append(nn.Linear(last, hidden_dim))
            policy_layers.append(nn.ReLU())
            last = hidden_dim
        policy_layers.append(nn.Linear(last, self.action_space.n))

        self._policy_head = nn.Sequential(*policy_layers).to(self.device)
        self._value_head = nn.Sequential(
            nn.Linear(combined_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        _, logits = self._compute_embeddings_and_logits(batch)
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        embeddings, logits = self._compute_embeddings_and_logits(batch)
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a TargetNetworkAPI RLModule, so it can be used
    # by the APPO algorithm.
    @override(TargetNetworkAPI)
    def make_target_networks(self) -> None:
        self._target_base_cnn_stack = make_target_network(self._base_cnn_stack)
        self._target_vector_net = make_target_network(self._vector_net)
        self._target_policy_head = make_target_network(self._policy_head)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self):
        return [
            (self._base_cnn_stack, self._target_base_cnn_stack),
            (self._vector_net, self._target_vector_net),
            (self._policy_head, self._target_policy_head),
        ]

    @override(TargetNetworkAPI)
    def forward_target(self, batch, **kw):
        grid_obs = batch[Columns.OBS][0].float().to(self.device)
        health_obs = batch[Columns.OBS][1].to(self.device)

        cnn_emb = self._target_base_cnn_stack(grid_obs)
        cnn_emb = torch.flatten(cnn_emb, start_dim=1)
        health_embed = self._target_vector_net(health_obs)

        combined = torch.cat([cnn_emb, health_embed], dim=-1)
        logits = self._target_policy_head(combined)

        return {TARGET_NETWORK_ACTION_DIST_INPUTS: logits}

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Features not provided -> We need to compute them first.
        if embeddings is None:
            cnn_emb, health_emb = self._get_grid_and_health_emb(batch)
            embeddings = torch.cat([cnn_emb, health_emb], dim=-1)

        return self._value_head(embeddings).squeeze(-1)

    def _compute_embeddings_and_logits(self, batch):
        cnn_emb, health_emb = self._get_grid_and_health_emb(batch)
        comb_emb = torch.cat([cnn_emb, health_emb], dim=-1)
        logits = self._policy_head(comb_emb)

        return (
            comb_emb,
            logits,
        )

    def _get_grid_and_health_emb(self, batch):
        """Helper function to go from batch to embs"""
        grid_obs = batch[Columns.OBS][0].float().to(self.device)
        health_obs = batch[Columns.OBS][1].to(self.device)

        cnn_emb = self._base_cnn_stack(grid_obs)
        cnn_emb = torch.flatten(cnn_emb, start_dim=1)

        health_emb = self._vector_net(health_obs)

        return cnn_emb, health_emb
