from .model import ActorCritic
import torch

def load_model(path, map_size):
    model = ActorCritic(map_size)
    model.load_state_dict(torch.load(path))
    return model





