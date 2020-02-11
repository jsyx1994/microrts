import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer



class A2C:
    def __init__(self,
        ac_model,
        lr=None,
        weight_decay=None,
        eps=1e-5,
        log_interval=100,
        gamma=0.99,
        entropy_coef=.01,
        value_loss_coef=1
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
        if rollouts.__len__() <= 0:
            return 
        sps_dict = rollouts.sample(batch_size='all')
        nn = self.actor_critic
        optimizer = self.optimizer

        for key in sps_dict:
            if key not in nn.activated_agents:
                continue

            if sps_dict[key]:
                states, units, actions, next_states, rewards,  done_masks = sps_dict[key].to(device)

                value, probs = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units)
                # m = torch.distributions.Categorical(probs=probs)
                print(value)
                value_next = nn.critic_forward(next_states).detach()

                targets = (rewards + self.gamma  * value_next * done_masks)

                advantages = targets - value.detach()
                pi_sa = probs.gather(1, actions)
                # print(m.entropy())
                # input()
                # entropy = (probs * torch.log(probs)).mean()
                # print(entropy)
                # input()
                entropy_loss = (probs * torch.log(probs)).mean()
                # print(m.logits)
                # input()
                # entropy_loss = -m.entropy().mean()

                policy_loss = -(torch.log(pi_sa + self.eps) * advantages).mean()
                criteria= torch.nn.MSELoss()
                value_loss = criteria(value, targets)


                optimizer.zero_grad()
                all_loss = policy_loss + value_loss * self.value_loss_coef  + self.entropy_coef * entropy_loss

                # all_loss = value_loss * self.value_loss_coef + policy_loss - dist_entropy * self.entropy_coef
                all_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), .5)
                optimizer.step()

                results = {
                    "p_loss": policy_loss.mean(),
                    "v_loss": value_loss.mean(),
                    "rewards": rewards.mean(),
                    # "entropy_loss": entropy_loss,
                    "all_losses":all_loss,
                }

                if iter_idx % self.log_interval == 0:
                    if callback:
                        callback(iter_idx, results)
                
        rollouts.refresh()

