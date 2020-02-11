import gym
import os
from microrts.rts_wrapper.envs.datatypes import List, Any
from microrts.rts_wrapper.envs.utils import unit_feature_encoder,network_action_translator, encoded_utt_dict, network_simulator, get_config
import torch
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.model import ActorCritic
import microrts.settings as settings
from microrts.rts_wrapper.envs.utils import action_sampler_v1


from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.a2c import A2C
from microrts.algo.agents import Agent

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


def self_play(env_id, render=0, opponent="socketAI", nn_path=None):
    """self play program
    
    Arguments:
        nn_path {str} -- path to model, if None, start from scratch
        map_size {tuple} -- (height, width)
    """     
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    def memo_inserter(transitions):
        if transitions['reward'] > 0:
            print(transitions['reward'])
        memory.push(**transitions)
    
    
    get_config(env_id).render = render
    get_config(env_id).ai2_type = opponent

    env = gym.make(env_id)
    # assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"
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
    
    import copy
    agents = [Agent(model=copy.deepcopy(nn)) for _ in range(env.players_num)]
    del nn

    # agents = [Agent(model=nn) for _ in range(env.players_num)]


    # print(players[0].brain is players[1].brain) # True

    # optimizer = torch.optim.RMSprop(nn.parameters(), lr=1e-5, weight_decay=1e-7)
    a2cs = [A2C(agents[i].brain, lr=7e-4, weight_decay = 1e-7, entropy_coef=0.01, value_loss_coef=.5, log_interval=5, gamma=.9) for i in range(env.players_num)]
    print(len(a2cs))
    # input()
    # algo = A2C(nn, lr=7e-4, weight_decay = 1e-7, entropy_coef=0.02, value_loss_coef=.1, log_interval=5, gamma=.9)
    update_step = 20 # + agents[0].random_rollout_steps

    step = 0

    for epi_idx in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        # print("reseted")
        start_time = time.time()
        players_G0 = [0, 0]
        while not obses_t[0].done:
            actions = []
            for i in range(len(players)):
                action = agents[i].think(callback=memo_inserter, obses=obses_t[i], accelerator=device, mode="train")
                actions.append(action)
            obses_tp1 = env.step(actions)
            step += 1

            if obses_tp1[0].done:
                for agent in agents:
                    agent.sum_up(callback=memo_inserter, obses=obses_tp1[i], accelerator=device, mode="train")
                    agent.forget()

            if step >= update_step:
                for i in range(env.players_num):
                    a2cs[i].update(agents[i].get_memory(), iter_idx, device, logger)
                    iter_idx += 1
                step = 0
            # if step >= update_step:
            #     algo.update(memory, iter_idx, device, logger)
            #     iter_idx += 1
            #     step = 0


            # just for analisis
            for i in range(len(players)):
                players_G0[i] += obses_tp1[i].reward
            

            obses_t = obses_tp1


        if (epi_idx + 1) % 100 == 0:
            for i in range(len(agents)):
                torch.save(agents[i].brain.state_dict(), os.path.join(settings.models_dir, "ai"+ str(i) + "_rl" + str(epi_idx) + ".pth"))

        print(players_G0)
        winner = obses_tp1[0].info["winner"]

        writer.add_scalar("Return_diff",abs(players_G0[0] - players_G0[1]) , epi_idx)
        writer.add_scalar("TimeStamp", obses_t[i].info["time_stamp"]  , epi_idx)

        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
        
    print(env.setup_commands)
    torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))



if __name__ == '__main__':
    torch.manual_seed(0)
    self_play("singleBattle-v0", render=0, opponent="socketAI")
    # self_play(nn_path=os.path.join(settings.models_dir, "rl999.pth"))
