from .model import ActorCritic
import torch

def load_model(path, height, width) -> ActorCritic:
    model = ActorCritic(height, width)
    model.load_state_dict(torch.load(path))
    return model


