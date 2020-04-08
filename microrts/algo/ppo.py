import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer
import copy

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class PPO:
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
        self.target_net = copy.deepcopy(ac_model)

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
        # nn = self.actor_critic
        optimizer = self.optimizer
        value_criteria= torch.nn.MSELoss()

        total_loss = 0
        total_rewards = 0

        for key in sps_dict:
            if key not in self.actor_critic.activated_agents:
                continue

            if sps_dict[key]:
                states, units, actions, next_states, rewards, hxses, done_masks = sps_dict[key].to(device)
                if self.actor_critic.recurrent:
                    value = self.target_net.critic_forward(states)
                    probs, _ = self.actor_critic.actor_forward(actor_type=key,spatial_feature=states,unit_feature=units,hxs=hxses.unsqueeze(0))
                    # value, probs, _ = self.actor_critic.forward(actor_type=key,spatial_feature=states,unit_feature=units,hxs=hxses.unsqueeze(0))
                else:
                    # value = self.target_net.critic_forward(states)
                    value_old, probs_old, _ = self.target_net.forward(actor_type=key,spatial_feature=states,unit_feature=units)
                    # probs_old, _ = self.target_net.actor_forward(actor_type=key,spatial_feature=states,unit_feature=units)
                    # probs, _ = self.actor_critic.actor_forward(actor_type=key,spatial_feature=states,unit_feature=units)
                    value, probs, _ = self.actor_critic.forward(actor_type=key,spatial_feature=states,unit_feature=units)
                # m = torch.distributions.Categorical(probs=probs)

                entropy = - (probs * torch.log(probs)).sum(dim=1)
                value_next = self.target_net.critic_forward(next_states).detach()
                # value_next = self.actor_critic.critic_forward(next_states).detach()

                # probs_next, _ = nn.actor_forward(actor_type=key,spatial_feature=states,unit_feature=units)
                # m = torch.distributions.Categorical(probs=probs_next)

                # rewards = rewards + m.entropy().unsqueeze(0)
                print(value)
                # rewards = (rewards - rewards.mean()) / (rewards.std()+1e-3) 
                pi_sa = probs.gather(1, actions)
                pi_sa_old = probs_old.gather(1,actions)
                log_pi_sa_old = torch.log(pi_sa_old + self.eps).detach()
                log_pi_sa = torch.log(pi_sa + self.eps).detach()
                # print(log_pi_sa)
                # print(rewards)
                # print(torch.tanh(rewards))
                # input()
                targets = rewards -  .2 * log_pi_sa + self.gamma  * value_next * done_masks

                advantages = (targets - value_old).detach()
                # print(m.entropy())
                # input()
                entropy_loss = -entropy.mean()
                ratio = pi_sa / pi_sa_old
                clip_ratio = 0.2
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
                policy_loss = -(torch.min(ratio * advantages, clip_adv)).mean() # + 0.04 * entropy_loss
                # policy_loss = -(torch.log(pi_sa + self.eps) * advantages.detach()).mean()
                # policy_loss = (-pi_sa/ pi_sa_old * advantages).mean()

                value_loss = value_criteria(value, targets)


                all_loss = policy_loss + value_loss * self.value_loss_coef  #+ self.entropy_coef * entropy_loss
                
                total_loss += all_loss
                total_rewards += rewards.mean()
                # all_loss = value_loss * self.value_loss_coef + policy_loss - dist_entropy * self.entropy_coef
                # all_loss.backward()


        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), .5)
        optimizer.step()

        results = {
            # "p_loss": policy_loss.mean(),
            # "v_loss": value_loss.mean(),
            # "rewards": total_rewards.mean(),
            # "entropy_loss": entropy_loss,
            "all_losses":all_loss,
        }

        if iter_idx % self.log_interval == 0:
            if callback:
                callback(iter_idx, results)
        
        # if iter_idx % 10 == 0:
        #     self.target_net = copy.deepcopy(self.actor_critic)
        #     # self.target_net.parameters = 0.001 * self.actor_critic.parameters + 0.999 * self.target_net.parameters
        with torch.no_grad():
            soft_update(self.target_net, self.actor_critic, tau=.001)           
        rollouts.refresh()

