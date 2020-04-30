import gym
import os
from microrts.rts_wrapper.envs.datatypes import List, Any
from microrts.rts_wrapper.envs.utils import unit_feature_encoder,network_action_translator, encoded_utt_dict, network_simulator, get_config
import torch
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.model import ActorCritic
import microrts.settings as settings
from microrts.rts_wrapper.envs.utils import action_sampler_v1
import argparse


from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.a2c import A2C
from microrts.algo.ppo import PPO

from microrts.algo.agents import Agent

def self_play(args):
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    def memo_inserter(transitions):
        if transitions['reward'] > 0:
            print(transitions['reward'])
        # if transitions['done'] == 2:
        #     print(transitions['done'])
        #     input()
        memory.push(**transitions)
    
    
    get_config(args.env_id).render = args.render
    get_config(args.env_id).ai2_type = args.opponent

    env = gym.make(args.env_id)
    # assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"
    memory = ReplayBuffer(10000)
    nn_path = args.model_path
    start_from_scratch = nn_path is None
    
    players = env.players
 
    if start_from_scratch:
        nn = ActorCritic(env.map_size, recurrent=args.recurrent)
    else:
        nn = load_model(os.path.join(settings.models_dir, nn_path), env.map_size, args.recurrent)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    nn.to(device)
    from torch.utils.tensorboard import SummaryWriter
    import time
    writer = SummaryWriter()
    iter_idx = 0
    

    agents = [Agent(model=nn, smooth_sample_ratio=0) for _ in range(env.players_num)]
    if args.algo == "a2c":
        algo = A2C(
            ac_model=nn,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            weight_decay=3e-6,
            log_interval=args.log_interval,
            gamma=args.gamma,
        )
    elif args.algo == "ppo":
        algo = PPO(
            ac_model=nn,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            weight_decay=3e-6,
            log_interval=args.log_interval,
            gamma=args.gamma,
            )
    # algo = A2C(nn,lr=args.lr, weight_decay=1e-7, entropy_coef=args.entropy, value_loss_coef=args.value_loss_coef, log_interval=5, gamma=args.gamma)
    # update_step = 64 #+ agents[0].random_rollout_steps
    # step = 0
    # bg_state = None
    for epi_idx in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        # print(bg_state == obses_t[0])
        # bg_state = obses_t[0]
        # input()
        # print("reseted")
        start_time = time.time()
        players_G0 = [0, 0]
        while not obses_t[0].done:
            actions = []
            for i in range(len(players)):
                if args.algo == 'ppo':
                    action = agents[i].think(sp_ac=algo.target_net,callback=memo_inserter, obses=obses_t[i], accelerator=device, mode="train")
                elif args.algo == 'a2c':
                    action = agents[i].think(callback=memo_inserter, obses=obses_t[i], accelerator=device, mode="train")
                actions.append(action)
            obses_tp1 = env.step(actions)
            if obses_tp1[0].done:
                # print(obses_tp1[0].done)
                for agent in agents:
                    if args.algo == 'ppo':
                        agents[i].sum_up(sp_ac=algo.target_net,callback=memo_inserter, obses=obses_tp1[i], accelerator=device, mode="train")
                    elif args.algo == 'a2c':
                        agents[i].sum_up(callback=memo_inserter, obses=obses_tp1[i], accelerator=device, mode="train")
                    agent.forget()
            # if len(memory) >= update_step:
            # # if step >= 5:
            #     algo.update(memory, iter_idx, device, logger)
            #     iter_idx += 1
                # step = 0

            # just for analisis
            # for i in range(len(players)):
            #     players_G0[i] += obses_tp1[i].reward
            obses_t = obses_tp1

        algo.update(memory, iter_idx, device, logger)
        iter_idx += 1
        if (epi_idx + 1) % 100 == 0:
            torch.save(nn.state_dict(), os.path.join(settings.models_dir, args.saving_prefix + str(epi_idx) + ".pth"))

        # print(players_G0)
        winner = obses_tp1[0].info["winner"]
        writer.add_scalar("P0_rewards", agents[0].rewards/obses_t[i].info["time_stamp"], epi_idx)
        # writer.add_scalar("P1_rewards", agents[1].rewards/obses_t[i].info["time_stamp"], epi_idx)
        # writer.add_scalar("Return_diff", agents[0].rewards - agents[1].rewards , epi_idx)
        writer.add_scalar("TimeStamp", obses_t[i].info["time_stamp"]  , epi_idx)

        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
        
    print(env.setup_commands)
    torch.save(nn.state_dict(), os.path.join(settings.models_dir, args.saving_prefix + ".pth"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default="minigame-v0"
    )
    parser.add_argument(
        '--model-path', help='path of the model to be loaded',
        default=None
    )
    parser.add_argument(
        '--episodes',
        # default=10e6,
        type=int,
        default=10e4,
    )
    parser.add_argument(
        '--recurrent',
        action="store_true",
        # type=bool,
        default=False,
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=int(100),
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--value_loss_coef',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99
    )
    parser.add_argument(
        '--render',
        type=int,
        default=1
    )
    parser.add_argument(
        '--opponent',
        default="Passive"
    )
    parser.add_argument(
        "--saving-prefix",
        default='rl',
    )
    parser.add_argument(
        "--algo",
        default='a2c',
    )
    args = parser.parse_args()
    print(args)
    torch.manual_seed(0)
    self_play(args)
    # self_play(nn_path=os.path.join(settings.models_dir, "rl999.pth"))