import dill
from microrts.rts_wrapper.envs.datatypes import Records
from microrts.rts_wrapper.envs.utils import state_encoder, unit_feature_encoder


import torch
from microrts.sl.play_buffer import PlayBuffer
from microrts.algo.model import ActorCritic
from microrts.algo.eval import evaluate_game
from .utils import load



def get_data(saving_dir) -> PlayBuffer:
    storage = PlayBuffer()
    rcds = load(saving_dir)
    print("total records:{}".format(rcds.records.__len__()))
    for r in rcds.records:
        gs = r.gs
        actions = r.actions
        curr_player = r.player
        # shared_states = state_encoder(gs, curr_player)
        for a in actions:
            storage.push(gs,curr_player, a.unit, a.unitAction)
    return storage



if __name__ == '__main__':

    input()
    # print(actions)
    # print(_actions)
    # actions = torch.from_numpy(actions).float()
    # print(prob - actions)
    # print(states.shape, unit_types.shape, units.shape, actions.shape)