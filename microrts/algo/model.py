import torch
import torch.nn as nn
import torch.nn.functional as F
from microrts.algo.config import model_path
import os
import numpy as np
from torch import Tensor
from microrts.rts_wrapper.envs.datatypes import *
from torch.distributions import Categorical
from microrts.rts_wrapper.envs.utils import encoded_utt_feature_size


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Pic2Vector(nn.Module):
    """
    convent matrix to Vectors
    each pixel is like embedding to it's channel's dimension
    """
    def forward(self, input):
        batch_size, channel_size = input.size(0), input.size(1)
        return input.view(batch_size, channel_size, -1).transpose(-2, -1)  # transpose to keep d_model last


class NNBase(nn.Module):
    def __init__(self, out_size):
        super(NNBase, self).__init__()

    def common_func1(self):
        pass

    def common_func2(self):
        pass


class ActorCritic(nn.Module):

    def __init__(self, map_height, map_width, input_channel=21, unit_feature_size=18):
        super(ActorCritic, self).__init__()
        self.shared_out_size = 256

        self.activated_agents = [
            UNIT_TYPE_NAME_BASE,
            # UNIT_TYPE_NAME_BARRACKS,
            UNIT_TYPE_NAME_WORKER,
            # UNIT_TYPE_NAME_HEAVY,
            # UNIT_TYPE_NAME_LIGHT,
            # UNIT_TYPE_NAME_RANGED,
        ]

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU(),
            nn.Conv2d(128, 64, 2), nn.ReLU(),
            nn.AdaptiveMaxPool2d((map_height, map_width)),  # n * 64 * map_height * map_width
        )
        self.self_attn = nn.Sequential(
            Pic2Vector(),
            nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=1), nn.ReLU()
        )
        self.shared_linear = nn.Sequential(
            Flatten(),
            nn.Linear(64 * map_height * map_width, 512), nn.ReLU(),
            nn.Linear(512, self.shared_out_size), nn.ReLU(),
        )

        self.critic_mlps = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.critic_out = nn.Linear(32, 1)

        self.actor_mlps = nn.Sequential(
            nn.Linear(self.shared_out_size + unit_feature_size + encoded_utt_feature_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        self.actor_out = nn.ModuleDict({
            UNIT_TYPE_NAME_WORKER: nn.Sequential(
                # nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),

                nn.Linear(32, WorkerAction.__members__.items().__len__()),  # logits
                nn.Softmax(dim=1)
            ),
            UNIT_TYPE_NAME_BASE: nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, BaseAction.__members__.items().__len__()),
                nn.Softmax(dim=1),
            )
        })

    def forward(self, spatial_feature: Tensor, unit_feature: Tensor, actor_type='Worker'):
        value = self.critic_forward(spatial_feature)
        pi = self.actor_forward(actor_type, spatial_feature, unit_feature)
        return value, pi

    def _shared_forward(self, spatial_feature):
        x = self.shared_conv(spatial_feature)
        x = self.self_attn(x)
        # print(x.shape)
        x = self.shared_linear(x)
        return x

    def evaluate(self, spatial_feature: Tensor):
        self.eval()
        x = self._shared_forward(spatial_feature)
        x = self.critic_mlps(x)
        x = self.critic_out(x)
        return x

    def critic_forward(self, spatial_feature: Tensor):
        x = self._shared_forward(spatial_feature)
        x = self.critic_mlps(x)
        x = self.critic_out(x)
        return x

    def actor_forward(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        # b_sz = spatial_feature.size(0)
        #
        # if training:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # else:
        #     device = torch.device("cpu")
        #
        # encoded_utt = encoded_utt.repeat(b_sz, 1).to(device)

        x = self._shared_forward(spatial_feature)

        x = torch.cat([x, unit_feature], dim=1)
        x = self.actor_mlps(x)
        probs = self.actor_out[actor_type](x)
        return probs

    def deterministic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.actor_forward(actor_type, spatial_feature, unit_feature)
        # print(prob)
        return list(AGENT_ACTIONS_MAP[actor_type])[torch.argmax(probs).item()]

    def stochastic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs = self.actor_forward(actor_type, spatial_feature, unit_feature)
        m = Categorical(probs)
        idx = m.sample().item()
        action = list(AGENT_ACTIONS_MAP[actor_type])[idx]
        return action



def test_network():
    import json
    unit_entity_str = '{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1}'
    pgs_wrapper_str = '{"reward":140.0,"done":false,"validActions":[{"unit":{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":0, "parameter":10}]},{"unit":{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},"unitActions":[{"type":1, "parameter":0} ,{"type":1, "parameter":1} ,{"type":1, "parameter":2} ,{"type":1, "parameter":3} ,{"type":0, "parameter":10}]}],"gs":{"time":164,"pgs":{"width":6,"height":6,"terrain":"000000000000000000000000000000000000","players":[{"ID":0, "resources":2},{"ID":1, "resources":5}],"units":[{"type":"Resource", "ID":0, "player":-1, "x":0, "y":0, "resources":230, "hitpoints":1},{"type":"Base", "ID":19, "player":1, "x":5, "y":5, "resources":0, "hitpoints":10},{"type":"Base", "ID":20, "player":0, "x":2, "y":2, "resources":0, "hitpoints":10},{"type":"Worker", "ID":22, "player":0, "x":0, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":23, "player":0, "x":5, "y":2, "resources":0, "hitpoints":1},{"type":"Worker", "ID":24, "player":0, "x":3, "y":4, "resources":0, "hitpoints":1},{"type":"Worker", "ID":25, "player":0, "x":0, "y":1, "resources":0, "hitpoints":1},{"type":"Worker", "ID":26, "player":0, "x":2, "y":3, "resources":0, "hitpoints":1}]},"actions":[{"ID":20, "time":153, "action":{"type":4, "parameter":1, "unitType":"Worker"}},{"ID":26, "time":158, "action":{"type":1, "parameter":3}},{"ID":19, "time":160, "action":{"type":0, "parameter":10}},{"ID":25, "time":162, "action":{"type":2, "parameter":0}},{"ID":23, "time":163, "action":{"type":1, "parameter":2}}]}}'
    unit = from_dict(data_class=Unit, data=json.loads(unit_entity_str))
    gs_wrapper = from_dict(data_class=GsWrapper, data=json.loads(pgs_wrapper_str))

    scalar_feature_actor = np.array([p.resources for p in gs_wrapper.gs.pgs.players])
    rsrc1, rsrc2 = scalar_feature_actor
    scalar_feature_critic = np.array([0.5 if rsrc1 == rsrc2 else (rsrc1 - rsrc2) / (rsrc1 + rsrc2)])
    unit_feature = unit_feature_encoder(unit, gs_wrapper.gs.pgs.height, gs_wrapper.gs.pgs.width)

    # print(scalar_feature)

    cnnbase = CNNBase(6, 6, 19)
    input_data = torch.randn(1, 19, 6, 6)

    critic = Critic(cnnbase)

    base_out = cnnbase(input_data, torch.from_numpy(scalar_feature_critic).unsqueeze(0).float())

    actor = Actor(base_out.size(1))
    print(actor("Worker", base_out, unit_feature, scalar_feature_actor))
    # print(actor)
    # print(actor(base_out,))
    # actor = Actor()np.ndarray
    # print(actor)


if __name__ == '__main__':
    print(nn.Transformer)
