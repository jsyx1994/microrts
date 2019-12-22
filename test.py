import gym
import microrts.rts_wrapper
import os
from microrts.rts_wrapper.envs.datatypes import List, Any
from microrts.rts_wrapper.envs.utils import unit_feature_encoder,network_action_translator, encoded_utt_dict, network_simulator
import torch
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.model import ActorCritic
import microrts.settings as settings
from microrts.rts_wrapper.envs.utils import action_sampler_v1


from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.replay_buffer import ReplayBuffer


def evaluate():
    env = gym.make("EvalAgainstRandom-v0")
    players = env.players
    assert env.player1 is not None, "player No.1 can not be missed"
    eval_model = load_model(os.path.join(settings.models_dir, "_1M.pth"), env.map_size)
    env.player1.load_brain(eval_model)
    # env.player1.load_brain(os.path.join(settings.models_dir, "1M.pth"), env.map_size[0], env.map_size[1])
    # input()
    for _ in range(env.max_episodes):
        obses = env.reset()  # p1 and p2 reset
        while not obses[0].done:
            actions = []
            for i in range(len(players)):
                # players[i].think(obses[i])
                # print(players[i].think(action_sampler_v1, obs=obses[i].observation, info=obses[i].info))
                actions.append(players[i].think(action_sampler_v1, obs=obses[i].observation, info=obses[i].info))
                # input()
                # actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses = env.step(actions)
            # print(obses)
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)


def self_play(nn_path=None):
    """self play program
    
    Arguments:
        nn_path {str} -- path to model, if None, start from scratch
        map_size {tuple} -- (height, width)
    """     

    env = gym.make("CurriculumBaseWorker-v0")
    assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"
    memory = ReplayBuffer(10000)

    start_from_scratch = nn_path is None
    
    players = env.players

    if start_from_scratch:
        nn = ActorCritic(env.map_size)
    else:
        nn = load_model(nn_path, env.map_size)


    for p in players:
        p.load_brain(nn)
    
    
    
    for _ in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        while not obses_t[0].done:
            actions = []
            for i in range(len(players)):
                # players[i].think(obses[i])
                # print(players[i].think(action_sampler_v1, obs=obses[i].observation, info=obses[i].info))
                actions.append(players[i].think(obs=obses_t[i].observation, info=obses_t[i].info))
                # input()
                # print(actions)
                # input()
                # actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses_tp1 = env.step()
            # print(obses_tp1[0].reward)


            memory.refresh()
            for i in range(len(players)):
                if actions[i]:
                    memory.push(
                        obs_t=obses_t[i].observation,
                        action=actions[i],
                        obs_tp1=obses_tp1[i].observation,
                        reward=obses_tp1[i].reward,
                        done=obses_tp1[i].done
                    )
            sps_dict = memory.sample(batch_size="all")
            for key in sps_dict:
                if key not in nn.activated_agents:
                    continue

                if sps_dict[key]:
                    # print(sps_dict[key].to("cpu"))
                    # print(sps_dict[key].rewards)

                    # states : np.array
                    # units: np.array
                    # actions: np.array
                    # next_states: np.array
                    # rewards: np.array
                    # done_masks: np.array
                    states, units, actions, next_states, rewards,  done_masks = sps_dict[key].__dict__.values()

                    states = torch.from_numpy(states).float()
                    units = torch.from_numpy(units).float()
                    actions = torch.from_numpy(actions).long().unsqueeze(1)
                    next_states = torch.from_numpy(next_states).float()
                    rewards = torch.from_numpy(rewards).float().unsqueeze(1)
                    done_masks=torch.from_numpy(done_masks).bool().unsqueeze(1)

                    value, probs = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units)

                    value_next = nn.critic_forward(next_states)
                    
                    # torch.gather()
                    print(value)
                    pi_sa = probs.gather(1, actions)

                    policy_loss = - torch.log(pi_sa + 1e-7) * (rewards + value_next - value)
                    value_loss = torch.nn.functional.mse_loss(rewards + value_next, value)

                    # print(policy_loss.size())
                    all_loss = policy_loss.mean() + value_loss.mean()

                    optimizer = torch.optim.RMSprop(nn.parameters(),lr=1e-4,weight_decay=1e-5)

                    optimizer.zero_grad()
                    all_loss.backward()
                    optimizer.step()

                    # pi_sa = torch.gather(probs, 1, actions.unsqueeze(1))
                    # pi_sa = probs.gather(1,actions.unsqueeze(0))

                    # print(states)
                    # input() 

            # input()







            obses_t = obses_tp1
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)



if __name__ == '__main__':
    self_play()
    # evaluate()
# print(rts_wrapper.base_dir_path)
# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
