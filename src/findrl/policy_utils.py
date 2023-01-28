#   https://github.com/Akhilesh-Gogikar/MultiStockRLTrading
#   https://pytorch-geometric.readthedocs.io/en/latest/
#   https://github.com/leftthomas/CapsuleLayer
#   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
#   pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
#   https://medium.com/@akhileshgogikar/applied-reinforcement-learning-3e73ca771bac

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from stable_baselines3.common.policies import ActorCriticPolicy
from torch_geometric.nn import GATv2Conv


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.time_step = time_step
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_block = AttentionBlock(time_step, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim

    def forward(self, seq):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        seq_vector, _ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        attention_vec = attention_vec.view(-1, 1, self.dim)  # prepare for concat
        return attention_vec


class CapsGATattentionGRU(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, use_gru=True):
        super(CapsGATattentionGRU, self).__init__()

        outer_edge = np.ones(shape=(2, input_dim ** 2))

        count = 0
        for i in range(input_dim):
            for j in range(input_dim):
                outer_edge[0][count] = i
                outer_edge[1][count] = j
                count += 1

        # basic parameters
        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.outer_edge = outer_edge
        self.batch = 1
        self.inner_edge = torch.tensor(outer_edge, dtype=torch.int64).to('cuda:0')
        self.use_gru = use_gru
        # hidden layers
        if self.use_gru:
            self.temporal_encoder = nn.GRU(input_dim * hidden_dim, input_dim * hidden_dim, num_layers=2,
                                           bidirectional=False)

        self.TBNLayer = torch.nn.BatchNorm1d(time_step, track_running_stats=False)
        self.encoder_list = SequenceEncoder(hidden_dim, time_step, input_dim)

        self.inner_gat0 = GATv2Conv(hidden_dim, hidden_dim)
        self.inner_gat1 = GATv2Conv(hidden_dim, hidden_dim)
        self.attention = AttentionBlock(12, hidden_dim)
        self.caps_module = CapsuleLinear(out_capsules=self.input_dim, in_length=2 * hidden_dim, out_length=hidden_dim,
                                         in_capsules=None, routing_type='dynamic', num_iterations=3)
        self.fusion = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs):

        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0)

        if self.use_gru:
            embedding, _ = self.temporal_encoder(inputs.view(-1, self.time_step, self.input_dim * self.dim))

        att_vector, _ = self.attention(torch.tanh(embedding))  # (100,dim)

        batch = att_vector.shape[0]

        att_vector = torch.tanh(att_vector.view(-1, self.input_dim, self.dim))

        x = att_vector.view(-1, self.dim)

        if self.batch != batch:
            outer_edge = self.outer_edge
            for i in range(1, self.batch):
                outer_edge2 = outer_edge + self.input_dim
                outer_edge = np.concatenate((outer_edge, outer_edge2), axis=1)

            self.inner_edge = torch.tensor(outer_edge, dtype=torch.int64).to('cuda:0')
            self.batch = batch

        # inner graph interaction
        # print(att_vector.shape)
        inner_graph_embedding = torch.tanh(self.inner_gat0(x, self.inner_edge))
        inner_graph_embedding0 = torch.tanh(self.inner_gat1(inner_graph_embedding, self.inner_edge.view(2, -1)))
        inner_graph_embedding = torch.add(inner_graph_embedding, inner_graph_embedding0)

        inner_graph_embedding = inner_graph_embedding.view(-1, self.input_dim, self.dim)

        # fusion
        fusion_vec = torch.cat((att_vector, inner_graph_embedding), dim=-1)

        caps_out, _ = self.caps_module(fusion_vec)

        out_vec = torch.tanh(self.fusion(torch.tanh(caps_out)))

        return out_vec


class Transpose(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            # timesteps: int = 12,
            timesteps: int = 4,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(last_layer_dim_pi * timesteps * feature_dim, last_layer_dim_pi * feature_dim), nn.ReLU(),
            nn.Linear(last_layer_dim_pi * feature_dim, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(last_layer_dim_pi * timesteps * feature_dim, last_layer_dim_pi * feature_dim), nn.ReLU(),
            nn.Linear(last_layer_dim_pi * feature_dim, last_layer_dim_pi), nn.Tanh()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=self.action_space.shape[0],
            last_layer_dim_vf=self.action_space.shape[0],
            # timesteps=self.observation_space.shape[0],
            feature_dim=self.observation_space.shape[0]
            # timesteps=self.observation_space.shape[1],
            # feature_dim=self.observation_space.shape[2]
        )


class GATNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            # timesteps: int = 12,
            timesteps: int = 4,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(GATNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            CapsGATattentionGRU(last_layer_dim_pi, timesteps, feature_dim),
            nn.Linear(last_layer_dim_pi, 1), Transpose(), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            CapsGATattentionGRU(last_layer_dim_vf, timesteps, feature_dim),
            nn.Linear(last_layer_dim_pi, 1), Transpose(), nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class GATActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(GATActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GATNetwork(
            last_layer_dim_pi=self.action_space.shape[0],
            last_layer_dim_vf=self.action_space.shape[0],
            timesteps=self.observation_space.shape[0],
            feature_dim=self.observation_space.shape[0]
            # timesteps=self.observation_space.shape[1],
            # feature_dim=self.observation_space.shape[2]
        )
