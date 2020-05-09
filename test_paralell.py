import torch
from microrts.algo.model import ActorCritic
from microrts.algo.utils import load_model
from microrts.rts_wrapper.envs.multi_envs import make_env, make_vec_envs
from microrts.algo.agents import Agent
from torch.utils.tensorboard import SummaryWriter
from microrts.algo.replay_buffer import ReplayBuffer

from microrts.rts_wrapper.envs.utils import action_sampler_v2, network_simulator
from microrts.algo.ppo import PPO
from microrts.algo.a2c import A2C
from microrts.rts_wrapper.envs.datatypes import Config
import microrts.settings as settings 
import os
import argparse
from functools import partial

# AI added in league combat
    # private static AI getAIType(String type, UnitTypeTable utt){
    #     switch (type){
    #         case "WorkerRush"   : return new WorkerRush(utt);
    #         case "Random"       : return new RandomAI();
    #         case "RandomBiased" : return new RandomBiasedAI(utt);
    #         case "NaiveMCTS"    : return new NaiveMCTS(utt);
    #         case "MC"           : return new MonteCarlo(utt);
    #         case "UCT"          : return new UCT(utt);
    #         case "ABCD"         : return new ABCD(utt);
    #         case "MCTS"         : return new MLPSMCTS(utt);
    #         case "WorkerDefense": return new WorkerDefense(utt);
    #         case "LightRush"    : return new LightRush(utt);
    #         case "WorkerRushPP" : return new WorkerRushPlusPlus(utt);
    #         case "PortfolioAI"  : return new PortfolioAI(utt);
    #         case "SCV"          : return new SCV(utt);
    #         default             : return new PassiveAI();
    #     }
    # }
def get_config(env_id) -> Config :
        from microrts.rts_wrapper import environments
        for registered in environments:
            if registered["id"] == env_id:
                return registered['kwargs']['config']
                # return registered['kwargs']['config'].height, registered['kwargs']['config'].width


def play(args):
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    def memo_inserter(transitions):
        nonlocal T
        T += 1
        # if transitions['reward'] < 0:
        # print(transitions['reward'])
        memory.push(**transitions)


    nn_path = args.model_path
    start_from_scratch = nn_path is None
    config = get_config(args.env_id)
    config.render = args.render
    config.ai2_type = args.opponent
    config.max_episodes = int(args.episodes)
    # config.render=1
    map_size = config.height, config.width
    # max_episodes = args.episodes

    memory = ReplayBuffer(10000)

    if start_from_scratch:
        nn = ActorCritic(map_size)
    else:
        nn = load_model(nn_path, map_size)
    
    # nn.share_memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    # input()
    nn.to(device)
    num_process = args.num_process
    league = ['socketAI' for _ in range(num_process)]
    cmd_league  =args.league.split(',')
    if num_process < len(cmd_league):
        print('The league input is larger than the number of process, will not use league learning')
    else:
        print("league learning staring")
        for i, x in enumerate(cmd_league):
            print(x)
            league[i] = x
    print('All leagues participated are', league)

    envs, agents = make_vec_envs(args.env_id, num_process, "fork", nn, league=league)
    buffers = [ReplayBuffer(config.max_cycles + 100) for _ in range(len(agents))]
    import time
    frames = 0
    st = time.time()
    obses_n = envs.reset()
    update_steps = 32
    T = 1
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
    writer = SummaryWriter()
    iter_idx = 0
    epi_idx = 0
    while 1:
        time_stamp  = []
        actions_n = []
        for i in range(num_process):
            action_i = []
            for j in range(len(obses_n[i])):
                if T % (update_steps * num_process) == 0:
                    T = 1
                    # print('Update...')
                    # input() 
                    algo.update(memory, iter_idx, callback=logger, device=device)
                    iter_idx += 1

                if not obses_n[i][j].done:
                    if args.algo == 'ppo':
                        action = agents[i][j].think(sp_ac=algo.target_net,callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                    elif args.algo == 'a2c':
                        action = agents[i][j].think(callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                else:
                    action = [] # reset
                    epi_idx += .5
                    time_stamp.append(obses_n[i][j].info["time_stamp"])
                    writer.add_scalar("rewards", agents[i][j].rewards / (obses_n[i][j].info["time_stamp"]), epi_idx)
                    if args.algo == 'ppo':
                        agents[i][j].sum_up(sp_ac=algo.target_net,callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                    elif args.algo == 'a2c':
                        agents[i][j].sum_up(callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                    # buffers[i]
                    agents[i][j].forget()
                action_i.append(action)
                if (epi_idx + 1) % 100 == 0:
                    torch.save(nn.state_dict(), os.path.join(settings.models_dir, args.saving_prefix + str(int(epi_idx)) + ".pth"))
            
            # if obses_n[i][0].done:
            #     print(len(buffers[i]))
            #     algo.update(buffers[i], iter_idx, callback=logger, device=device)
            #     if T % (update_steps * num_process) == 0:
            #         T = 1
            #         # print('Update...')
            #         # input() 
            #         algo.update(memory, iter_idx, callback=logger, device=device)
            #         iter_idx += 1


            actions_n.append(action_i)
    
        if time_stamp:
            writer.add_scalar("TimeStamp", sum(time_stamp) / (len(time_stamp)), epi_idx)
        obses_n = envs.step(actions_n)
        frames += 1
        
        if frames >= 1000:
            print("fps", frames * num_process / (time.time() - st))
            frames = 0
            st = time.time()
            # torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default="attackHome-v0"
    )
    parser.add_argument(
        '--model-path', help='path of the model to be loaded',
        default=None
    )
    parser.add_argument(
        '--num-process',
        type=int,
        default=4
    )
    parser.add_argument(
        '--smooth-ratio',
        type=float,
        default=.0
    )
    parser.add_argument(
        '--episodes',
        # default=10e6,
        type=int,
        default=10e4,
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=int(100),
    )
    parser.add_argument(
        '--recurrent',
        action="store_true",
        # type=bool,
        default=False,
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
        default=0.5
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99
    )
    parser.add_argument(
        '--render',
        type=int,
        default=0
    )
    parser.add_argument(
        '--opponent',
        default="socketAI"
    )
    parser.add_argument(
        "--saving-prefix",
        default='rl',
    )
    parser.add_argument(
        "--algo",
        default='ppo',
    )
    parser.add_argument(
        "--league",
        default='socketAI',
    )
    args = parser.parse_args()
    print(args.league)
    # input()
    torch.manual_seed(0)
    play(args) #, nn_path=os.path.join(settings.models_dir,"rl39699.pth"))

