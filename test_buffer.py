import gym
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
    """deprecated"""
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
                actions.append(players[i].think(obs=obses[i].observation, info=obses[i].info))
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
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    env = gym.make("battle2v2LightMelee-v0")
    assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"
    memory = ReplayBuffer(10000)

    start_from_scratch = nn_path is None
    
    players = env.players

    if start_from_scratch:
        nn = ActorCritic(env.map_size)
    else:
        nn = load_model(nn_path, env.map_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    nn.to(device)
    from torch.utils.tensorboard import SummaryWriter
    import time
    writer = SummaryWriter()
    iter_idx = 0


    for p in players:
        p.load_brain(nn)
    

    # print(players[0].brain is players[1].brain) # True

    optimizer = torch.optim.RMSprop(nn.parameters(),lr=1e-6,weight_decay=1e-4)

    for epi_idx in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        start_time = time.time()
        players_G0 = [0, 0]
        while not obses_t[0].done:
            # actions = []
            for i in range(len(players)):
                # actions.append(players[i].think(obs=obses_t[i].observation, info=obses_t[i].info, accelerator=device))
                players[i].think(obses=obses_t[i], accelerator=device, mode="train")
            obses_tp1 = env.step()

            for i in range(len(players)):
                players_G0[i] += obses_tp1[i].reward

            if obses_tp1[0].done:
                # Get the last transition from env
                for i in range(len(players)):
                    players[i].think(obses=obses_tp1[i], accelerator=device, mode="train")
            obses_t = obses_tp1
            if obses_t[0].reward > 0 or obses_t[1].reward > 0:
                print(obses_t[0].reward, obses_t[1].reward)
            
            for i in range(len(players)):
                players[i].learn(optimizer=optimizer, iter_idx=iter_idx, batch_size="all", accelerator=device, callback=logger)
                iter_idx += 1

        print(players_G0)

        winner = env.get_winner()
        writer.add_scalar("TimeStamp",obses_t[i].info["time_stamp"], epi_idx)
        writer.add_scalar("Return_diff",abs(players_G0[0] - players_G0[1]) , epi_idx)
        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
        
    print(env.setup_commands)
    torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))



if __name__ == '__main__':
    from microrts.settings import models_dir
    import os
    # self_play(nn_path=os.path.join(models_dir, "rl.pth"))
    self_play()
    # evaluate()
# print(rts_wrapper.base_dir_path)\
# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
