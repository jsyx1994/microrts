import gym
import torch
import os
import microrts.settings as settings
from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.agents import Agent
from microrts.rts_wrapper.envs.utils import get_config
import argparse

def evaluate(
        env_id, 
        ai2_type="socketAI",
        nn_path=None, 
        fast_forward=False, 
        episodes=1000,
        stochastic=True,
        ):
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
    config.max_episodes = episodes
    config.ai2_type = ai2_type
    if fast_forward:
        config.render = 0
        config.period = 1
    else:
        config.render = 1
        config.period = 20
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
    winning_count=[0,0,0]

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
                if stochastic:
                    action = agents[i].think(obses=obses_t[i], way="stochastic", accelerator=device,mode="eval")
                else:
                    action = agents[i].think(obses=obses_t[i], way="deterministic", accelerator=device,mode="eval")
                if not fast_forward:
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
        winning_count[winner] += 1
        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
    return winning_count

if __name__ == "__main__":
    # python3 evaluate.py --model-path rl2v2.pth --fast-forward True --episodes 100
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', help='path of the model to be evaluated')
    parser.add_argument(
        '--fast-forward',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--episodes',
        # default=10e6,
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--stc',
        type=bool,
        default=False
    )
    args = parser.parse_args()
    print(args.fast_forward)
    winning_count = evaluate("singleBattle-v0","socketAI", nn_path=os.path.join(settings.models_dir, args.model_path), fast_forward=args.fast_forward, episodes=args.episodes,stochastic=args.stc)
    print(winning_count)
