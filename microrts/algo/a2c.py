import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer



class A2C:
    def __init__(self,ac_model, lr=None, weight_decay=None, eps=None, log_interval=100, gamma=0.99):
        self.actor_critic = ac_model
        self.optimizer = optim.RMSprop(ac_model.parameters(), lr)
        # self.optimizer = optim.Adam(ac_model.parameters(), lr)

        self.gamma = gamma

        self.log_interval = log_interval


    def update(self, rollouts: ReplayBuffer, iter_idx, device="cpu", callback=None ):
        sps_dict = rollouts.sample(batch_size='all')
        nn = self.actor_critic
        optimizer = self.optimizer

        for key in sps_dict:
            if key not in nn.activated_agents:
                continue

            if sps_dict[key]:
                states, units, actions, next_states, rewards,  done_masks = sps_dict[key].to(device)

                # done_masks = torch.FloatTensor(
                #     [[0.0] if _done == 1 else [1.0] for _done in done]
                # )

                # if rewards[0][0] > 0:
                #     print(rewards, actions)

                value, probs = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units)

                value_next = nn.critic_forward(next_states)

                targets = rewards + self.gamma * done_masks * value_next 
                
                pi_sa = probs.gather(1, actions)
                entropy_loss = - probs * torch.log(probs + 1e-7)
                policy_loss = - torch.log(pi_sa + 1e-7) * (targets - value)
                value_loss = torch.nn.functional.mse_loss(targets, value)

                all_loss = policy_loss.mean() + value_loss.mean() + .1 * entropy_loss.mean()
            
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                results = {
                    "p_loss": policy_loss.mean(),
                    "v_loss": value_loss.mean(),
                    # "entropy_loss": entropy_loss,
                    "all_losses":all_loss,
                }

                if iter_idx % self.log_interval == 0:
                    if callback:
                        callback(iter_idx, results)
                
        rollouts.refresh()

