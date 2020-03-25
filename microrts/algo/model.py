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

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        return m

class Pic2Vector(nn.Module):
    """
    convent matrix to Vectors
    each pixel is like embedding to it's channel's dimension
    """
    def forward(self, input):
        batch_size, channel_size = input.size(0), input.size(1)
        return input.reshape(batch_size, channel_size, -1).transpose(-2, -1)  # transpose to keep d_model last

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class NNBase(nn.Module):
    def __init__(self, out_size):
        super(NNBase, self).__init__()

    def common_func1(self):
        pass

    def common_func2(self):
        pass


class ActorCritic(nn.Module):

    def __init__(self, 
        map_size, 
        input_channel=43, #* 16,
        unit_feature_size=20,
        recurrent=False,
        ):
        """[summary]
        
        Arguments:
            map_size {tuple} -- (map_height, map_width)
        
        Keyword Arguments:
            input_channel {int} -- [description] (default: {21})
            unit_feature_size {int} -- [description] (default: {18})
        """
        super(ActorCritic, self).__init__()
        
        self.recurrent = recurrent
        map_height, map_width = map_size
        self.shared_out_size = 64


        self.activated_agents = [
            UNIT_TYPE_NAME_BASE,
            UNIT_TYPE_NAME_BARRACKS,
            UNIT_TYPE_NAME_WORKER,
            UNIT_TYPE_NAME_HEAVY,
            UNIT_TYPE_NAME_LIGHT,
            # UNIT_TYPE_NAME_RANGED,
        ]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.shared_conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=2)), nn.ReLU(),
            # init_(nn.Conv2d(64, 32, 1)), nn.ReLU(),
            init_(nn.Conv2d(32, 16, 2)), nn.ReLU(),
            # init_(nn.Conv2d(64, 32, 2)), nn.ReLU(),
            # nn.Conv2d(64, 32, 2), nn.ReLU(),
            # nn.AdaptiveMaxPool2d((map_height, map_width)),  # n * 64 * map_height * map_width
        )

        # self.self_attn = nn.Sequential(
        #     Pic2Vector(),
        #     # nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, dropout=0, activation="relu"),
        # )
        # self.self_attn = nn.MultiheadAttention(embed_dim=16,num_heads=8)
        self.self_attn = MultiHeadAttention(n_head=4,d_model=16,d_k=16,d_v=16,dropout=0)
        # self.self_attn.share_memory()


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.shared_linear = nn.Sequential(
            Flatten(),
            init_(nn.Linear(16 * (map_height-2) * (map_width-2), 128)), nn.ReLU(),
            # init_(nn.Linear(256, 256)), nn.ReLU(),
            init_(nn.Linear(128, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            # init_(nn.Linear(128, 128)), nn.ReLU(),
            # init_(nn.Linear(128, self.shared_out_size)), nn.ReLU(),
        )


        self.critic_mlps = nn.Sequential(
            init_(nn.Linear(self.shared_out_size, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            # init_(nn.Linear(128, 128)), nn.ReLU(),
            # init_(nn.Linear(256, 256)), nn.ReLU(),
            # init_(nn.Linear(256, 256)), nn.ReLU(),

        )
        self.critic_out = init_(nn.Linear(64, 1))

        #-------------------------------------------#


        self.actor_mlps = nn.Sequential(
            init_(nn.Linear(self.shared_out_size + unit_feature_size + encoded_utt_feature_size, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            init_(nn.Linear(64, 64)), nn.ReLU(),
            # init_(nn.Linear(128, 128)), nn.ReLU(),
            # init_(nn.Linear(128, 128)), nn.ReLU(),
            # init_(nn.Linear(256, 256)), nn.ReLU(),
        )
        if recurrent:
            self.gru = nn.GRU(128, 128)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        self.layer_norm = nn.LayerNorm(normalized_shape=(128),elementwise_affine=False)

        self.actor_out = nn.ModuleDict({
            UNIT_TYPE_NAME_WORKER: nn.Sequential(
                init_(nn.Linear(64, 64)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),

                init_(nn.Linear(64, WorkerAction.__members__.items().__len__())),
                nn.Softmax(dim=1)
            ),
            UNIT_TYPE_NAME_BASE: nn.Sequential(
                # nn.Linear(256, 128), nn.ReLU(),
                # nn.Linear(128, 64), nn.ReLU(),
                # nn.Linear(64, 64), nn.ReLU(),
                # nn.Linear(64, 64), nn.ReLU(),
                # nn.Linear(64, 32), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),

                init_(nn.Linear(64, BaseAction.__members__.items().__len__())),
                nn.Softmax(dim=1),
            ),
            UNIT_TYPE_NAME_LIGHT: nn.Sequential(
                # init_(nn.Linear(256, 256)), nn.ReLU(),
                # init_(nn.Linear(256, 256)), nn.ReLU(),
                # init_(nn.Linear(256, 256)), nn.ReLU(),
                # init_(nn.Linear(256, 256)), nn.ReLU(),
                # init_(nn.Linear(128, 128)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),

                init_(nn.Linear(64, LightAction.__members__.items().__len__())),
                nn.Softmax(dim=1),
            ),
            UNIT_TYPE_NAME_BARRACKS: nn.Sequential(
                # init_(nn.Linear(128, 128)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),
                init_(nn.Linear(64, BarracksAction.__members__.items().__len__())),
                nn.Softmax(dim=1),
            ),
            UNIT_TYPE_NAME_HEAVY: nn.Sequential(
                # nn.Linear(256, 128), nn.ReLU(),
                # nn.Linear(128, 64), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),
                init_(nn.Linear(64, 64)), nn.ReLU(),
                # nn.Linear(64, 64), nn.ReLU(),
                # nn.Linear(64, 32), nn.ReLU(),
                init_(nn.Linear(64, HeavyAction.__members__.items().__len__())),
                nn.Softmax(dim=1),
            )
        })

    def forward(self, spatial_feature: Tensor, unit_feature: Tensor, actor_type, hxs = None):
        value = self.critic_forward(spatial_feature)
        pi, hxs_n = self.actor_forward(actor_type, spatial_feature, unit_feature, hxs)
        return value, pi, hxs_n

    def _shared_forward(self, spatial_feature):
        x = self.shared_conv(spatial_feature)
        # print(x.shape)
        # input()
        x = Pic2Vector()(x)
        # print(x.size())
        # input()
        x, _ = self.self_attn(x,x,x)
        x = self.shared_linear(x)
        # x = self.layer_norm(x)

        # print(x.shape)
        # input()
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

    def actor_forward(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor, hxses:Tensor=None):
        hxs = None
        x = self._shared_forward(spatial_feature)
        # x = self.shared_to_actor(x)
        # x = x.detach()
        if self.recurrent:
            x, hxs =self.gru(x.unsqueeze(0), hxses)
            x = x.squeeze(0)
            # x = self.layer_norm(x)

        x = torch.cat([x, unit_feature], dim=1)
        x = self.actor_mlps(x)
        # x, hxs =self.gru(x.unsqueeze(0), torch.randn(1,x.size(0),128))

        # print(x)
        # # self.gru(x, torch.Tensor())
        # print(x.shape)
        # input()

        probs = self.actor_out[actor_type](x)
        # print(probs)
        return probs, hxs
    
    def evaluate_actions(self, actor_type: str, states: Tensor, unit_feature: Tensor, actions):
        x = self._shared_forward(states)
        value = self.critic_out(self.critic_mlps(x))
        x = torch.cat([x, unit_feature], dim=1)
        x = self.actor_mlps(x) 
        dist = Categorical(logits=x)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy


    def deterministic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs, hxs = self.actor_forward(actor_type, spatial_feature, unit_feature)
        # print(prob)
        return list(AGENT_ACTIONS_MAP[actor_type])[torch.argmax(probs).item()]

    def stochastic_action_sampler(self, actor_type: str, spatial_feature: Tensor, unit_feature: Tensor):
        if actor_type not in self.activated_agents:
            return AGENT_ACTIONS_MAP[actor_type].DO_NONE

        probs, hsx = self.actor_forward(actor_type, spatial_feature, unit_feature)
        # print(probs)
        m = Categorical(probs)
        idx = m.sample().item()
        action = list(AGENT_ACTIONS_MAP[actor_type])[idx]
        return action





if __name__ == '__main__':
    print(nn.Transformer)
