import torch
import torch.nn as nn
import torch.optim as optim



class A2C:
    def __init__(self, ac_model, lr, optimizer):
        self.actor_critic = ac_model
        self.lr = lr
        pass


    def update(self):
        pass

