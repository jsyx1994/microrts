import gym
import microrts.rts_wrapper
import os
from microrts.rts_wrapper.envs.datatypes import List, Any
from microrts.rts_wrapper.envs.utils import unit_feature_encoder,network_action_translator, encoded_utt_dict, network_simulator
import torch
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.model import ActorCritic


def action_sampler_v1(model:ActorCritic, state, info, mode='stochastic'):
    assert mode in ['stochastic', 'deterministic']
    time_stamp = info["time_stamp"]
    # if time_stamp % 1 != 0:
    #     return []
    unit_valid_actions = info["unit_valid_actions"]  # unit and its valid actions
    height, width = info["map_size"]
    player_resources = info["player_resources"]  # global resource situation, default I'm player 0
    current_player = info["current_player"]

    spatial_feature = torch.from_numpy(state).float().unsqueeze(0)
    samples = []
    for uva in unit_valid_actions:
        u  = uva.unit
        unit_feature = torch.from_numpy(unit_feature_encoder(u, height, width)).float().unsqueeze(0)
        encoded_utt = torch.from_numpy(encoded_utt_dict[u.type]).float().unsqueeze(0)

        unit_feature = torch.cat([unit_feature, encoded_utt], dim=1)
        if mode == 'stochastic':
            sampled_unit_action = model.stochastic_action_sampler(u.type, spatial_feature, unit_feature)
        elif mode == 'deterministic':
            sampled_unit_action = model.deterministic_action_sampler(u.type, spatial_feature, unit_feature)

        samples.append((uva, sampled_unit_action))

    return network_action_translator(samples)


def test_pve():
    env = gym.make("CurriculumBaseWorker-v0")
    # env = gym.make("Microrts-v0")
    # env = gym.make("OneWorkerAndBaseWithResources-v0")
    map_height, map_width = env.config.height, env.config.width
    model = ActorCritic(map_height, map_width)
    # model.load_state_dict(torch.load("./models/100k.pth"))
    replay_buffer = ReplayBuffer(100)
    for _ in range(env.config.max_episodes):
        obs_t, _, done, info_t = env.reset()  # deserting the reward
        while not done:
            # action = env.sample(info["unit_valid_actions"])
            # action = env.network_simulate(info_t["unit_valid_actions"])
            # action = action_sampler_v0(actor, critic, state_t, info_t)
            action = action_sampler_v1(model, obs_t, info_t)
            # print(action)
            state_tp1, reward, done, info_tp1 = env.step(action)

            obs_t, info_t = state_tp1, info_tp1  # assign new state and info from environment
            # replay_buffer.push(state_t,info_t,ac)

        winner = env.get_winner()  # required
        print(winner)

    env.close()



def main():
    env = gym.make("CurriculumBaseWorker-v0")
    players = env.players

    for _ in range(env.max_episodes):
        obses = env.reset()  # p1 and p2 reset
        while not obses[0].done:
            actions = []
            for i in range(len(players)):
                # players[i].think(obses[i])
                actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses = env.step(actions)
            # print(obses)
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)

if __name__ == '__main__':
    main()
# print(rts_wrapper.base_dir_path)
# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
