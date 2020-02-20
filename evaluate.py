import gym
import torch
import os
import microrts.settings as settings
from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.agents import Agent
from microrts.rts_wrapper.envs.utils import get_config
import argparse

def evaluate(env_id, ai2_type="socketAI", nn_path=None):
    """self play program
    
    Arguments:
        nn_path {str} -- path to model, if None, start from scratch
        map_size {tuple} -- (height, width)
    """     
    # def logger(iter_idx, results):
    #     for k in results:
    #         writer.add_scalar(k, results[k], iter_idx)

    # env = gym.make("Evalbattle2v2LightMelee-v0")

    config = get_config(env_id)
    # print(config)
    # input()
    config.ai2_type = ai2_type
    env = gym.make(env_id)
    # assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"

    start_from_scratch = nn_path is None
    
    players = env.players

    if start_from_scratch:
        nn = ActorCritic(env.map_size)
    else:
        nn = load_model(nn_path, env.map_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    nn.to(device)
    # from torch.utils.tensorboard import SummaryWriter
    import time
    # writer = SummaryWriter()
    agents = [Agent(model=nn) for _ in range(env.players_num)]

    # print(players[0].brain is players[1].brain) # True

    # optimizer = torch.optim.RMSprop(nn.parameters(), lr=1e-5, weight_decay=1e-7)

    for _ in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        start_time = time.time()
        for a in agents:
            a.forget()
            
        while not obses_t[0].done:
            actions = []
            for i in range(len(players)):
                # actions.append(players[i].think(obs=obses_t[i].observation, info=obses_t[i].info, accelerator=device))
                # _st = time.time()
                action = agents[i].think(obses=obses_t[i], accelerator=device,mode="eval")
                print(action)
                input()
                # input()
                # print((time.time() - _st))

                # action = players[i].think(obses=obses_t[i], accelerator=device, mode="train")
                actions.append(action)
                # if trans:
                #     memory.push(**trans)
            
            obses_tp1 = env.step(actions)
            obses_t = obses_tp1

        winner = obses_tp1[0].info["winner"]
        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('m_path')
    args = parser.parse_args()

    evaluate("EvaldoubleBattle-v0","NaiveMCTS", nn_path=os.path.join(settings.models_dir, args.m_path))