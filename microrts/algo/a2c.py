import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer
import scipy.signal
from torch.functional import F
import numpy as np
def discount_cumsum_(rewards, gamma, durations):
    """
    semi-MDP discounted cumsum
    input: 
        vector rewards, 
        [x0, 
         x1, 
         x2]
        
        vector duration
        [d0,
         d1,
         d2,
        ]

        scalar gamma
        g
    output:
        [x0 * (g ** d0) + x1 * (g ** (d0 + d1)) + x2 * (g ** (d0 + d1 + d2)),
         x1 * (g ** d1) + x2 * (g ** (d1 + d2))
         x2 * (g ** d2)
        ]
    """
    lrews = rewards.numpy()[::-1]
    ld = durations.numpy()[::-1]
    for i in range(len(lrews)):
        discount = gamma ** ld[i]
        lrews[i] = discount * (sum(lrews[i-1:i]) + lrews[i])

    # lrew = torch.squeeze(rewards).numpy()[::-1]
    # for i in range(lrew.shape[0]):
    #     print(i)
    #     lrew[i] = discount[i] * sum(lrew[i-1:i]) +lrew[i]
    return torch.from_numpy(lrews[::-1])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    x = x.numpy()
    # print(x.shape, type(x))
    x= scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return torch.Tensor(x.copy())


class A2C:
    def __init__(self,
        ac_model,
        lr=None,
        weight_decay=None,
        eps=1e-5,
        log_interval=100,
        gamma=0.99,
        entropy_coef=.01,
        value_loss_coef=1,
        debug=False,
        ):
        self.actor_critic = ac_model
        # self.optimizer = optim.RMSprop(ac_model.parameters(), lr, weight_decay=weight_decay)
        self.optimizer = optim.Adam(ac_model.parameters(), lr, weight_decay=weight_decay)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.eps = eps
        self.log_interval = log_interval
    

    def update(self, rollouts: ReplayBuffer, iter_idx, device="cpu", callback=None):
        def compute_pi_loss():

            pass

        def compute_val_loss(self):
            pass

        if rollouts.__len__() <= 0:
            return 
        sps_dict = rollouts.sample(batch_size='all')
        nn = self.actor_critic
        optimizer = self.optimizer

        total_loss = 0
        total_rewards = 0

        num_sps = 0
        # for key in sps_dict:
        #     if sps_dict[key]:
        #         num_sps += len(sps_dict[key].actions)
        # #         print(num_sps)
        
        # print(num_sps)
        # input()

        for key in sps_dict:
            if key not in nn.activated_agents:
                continue

            if sps_dict[key]:
                states, units, actions, next_states, rewards, hxses, done_masks, durations, ctfs, irews = sps_dict[key].to(device)
                # rets = discount_cumsum(rewards, self.gamma)
                # rets = discount_cumsum_(rewards, self.gamma, durations)
                if self.actor_critic.recurrent:
                    value, probs, _ = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units,hxs=hxses.unsqueeze(0))
                else:
                    value, probs, _ = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units)
                # print(value)
                m = torch.distributions.Categorical(probs=probs)
                # print(probs.is_leaf)
                # input()
                # entropy = - (probs * torch.log(probs)).mean(dim=1)
                # entropy = - (probs * torch.log(probs)).sum()

                value_next = nn.critic_forward(next_states).detach()
                # probs_next, _ = nn.actor_forward(actor_type=key,spatial_feature=states,unit_feature=units)
                # m = torch.distributions.Categorical(probs=probs_next)

                # rewards = rewards + m.entropy().unsqueeze(0)
                pi_sa = probs.gather(1, actions)
                # pi_sa.retain_grad()
                # rewards = (rewards - rewards.mean())/rewards.std()
                targets = rewards + (self.gamma ** durations) * ( value_next * done_masks)
                # targets = rets
                # for bat in states:
                # ratio = []
                # for i in range(len(rewards)):
                #     if rewards[i] == 0:
                #         ratio.append([1])
                #     else:
                #         ratio.append([irews[i]/rewards[i]])
                # ratio = torch.Tensor(ratio)
                # print(ratio)
                advantages = targets - ctfs

                # advantages = rets - value
                # advantages = rewards[:-1] + self.gamma ** durations * value[1:] - value.detach()
                # adv = (advantages - advantages.mean()) / advantages.std()
                # print(adv)
                adv = advantages.detach()
                # print(adv)
                # print(m.entropy())
                # input()
                entropy_loss = -m.entropy().mean()
                policy_loss = -(torch.log(pi_sa) * adv).mean()
                # print(len(rewards))
                # input()
                value_loss = F.mse_loss(value, targets)
                # value_loss = value_criteria(targets, rets * done_masks)

                all_loss = policy_loss + value_loss * self.value_loss_coef + self.entropy_coef * entropy_loss
                # print(len(actions))
                total_loss += all_loss
                total_rewards += rewards.mean()
                # all_loss = value_loss * self.value_loss_coef + policy_loss - dist_entropy * self.entropy_coef


        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), .5)
        optimizer.step()

        results = {
            # "p_loss": policy_loss.mean(),
            # "v_loss": value_loss.mean(),
            "rewards": total_rewards.mean(),
            # "entropy_loss": entropy_loss,
            "all_losses":all_loss,
        }
        if iter_idx % self.log_interval == 0:
            if callback:
                callback(iter_idx, results)
                
        rollouts.refresh()

