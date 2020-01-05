import gym
import torch
import os
import microrts.settings as settings
from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic


def evaluate(nn_path=None):
    """self play program
    
    Arguments:
        nn_path {str} -- path to model, if None, start from scratch
        map_size {tuple} -- (height, width)
    """     
    # def logger(iter_idx, results):
    #     for k in results:
    #         writer.add_scalar(k, results[k], iter_idx)

    # env = gym.make("Evalbattle2v2LightMelee-v0")
    env = gym.make("Evalbattle2v2LightMelee-v0")
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
    iter_idx = 0


    for p in players:
        p.load_brain(nn)
    
    # print(players[0].brain is players[1].brain) # True

    # optimizer = torch.optim.RMSprop(nn.parameters(),lr=1e-5,weight_decay=1e-7)

    for epi_idx in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        start_time = time.time()
        while not obses_t[0].done:
            # actions = []
            for i in range(len(players)):
                # actions.append(players[i].think(obs=obses_t[i].observation, info=obses_t[i].info, accelerator=device))
                players[i].think(obses=obses_t[i], accelerator=device, mode="eval")
            obses_tp1 = env.step()
            if obses_tp1[0].done:
                # Get the last transition from env
                for i in range(len(players)):
                    players[i].think(obses=obses_tp1[i], accelerator=device, mode="eval")
            obses_t = obses_tp1
            
            # for i in range(len(players)):
            #     players[i].learn(optimizer=optimizer, iter_idx=iter_idx, batch_size="all", accelerator=device, callback=logger)
            #     iter_idx += 1
        

        winner = env.get_winner()
        # writer.add_scalar("TimeStamp",obses_t[i].info["time_stamp"], epi_idx)
        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
        
    print(env.setup_commands)

if __name__ == "__main__":
    evaluate(nn_path=os.path.join(settings.models_dir, "rl2000.pth"))