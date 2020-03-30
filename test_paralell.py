import torch
from microrts.algo.model import ActorCritic
from microrts.algo.utils import load_model
from microrts.rts_wrapper.envs.multi_envs import make_env, make_vec_envs
from microrts.algo.agents import Agent
from torch.utils.tensorboard import SummaryWriter
from microrts.algo.replay_buffer import ReplayBuffer

from microrts.rts_wrapper.envs.utils import action_sampler_v2, network_simulator
from microrts.algo.ppo import PPO
from microrts.rts_wrapper.envs.datatypes import Config
import microrts.settings as settings 
import os
import argparse

def get_config(env_id) -> Config :
        from microrts.rts_wrapper import environments
        for registered in environments:
            if registered["id"] == env_id:
                return registered['kwargs']['config']
                # return registered['kwargs']['config'].height, registered['kwargs']['config'].width


def play(env_id, nn_path=None):
    
    
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    def memo_inserter(transitions):
        # if transitions['reward'] > 0:
        #     print(transitions['reward'])
        memory.push(**transitions)


    start_from_scratch = nn_path is None

    config = get_config(env_id)
    # config.render=1
    map_size = config.height, config.width
    max_episodes = config.max_episodes

    memory = ReplayBuffer(10000)

    if start_from_scratch:
        nn = ActorCritic(map_size)
    else:
        nn = load_model(nn_path, map_size)
    
    # nn.share_memory()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    # input()
    nn.to(device)
    num_process = 4
    envs, agents = make_vec_envs(env_id, num_process, "fork", nn)
    import time
    frames = 0
    st = time.time()
    obses_n = envs.reset()
    # bg_state = obses_n[0][0]
    # print(len(obses_n[0]))
    # input()
    update_steps = 16
    ppo = PPO(nn,3e-4,entropy_coef=0.04,value_loss_coef=.5, weight_decay=3e-6, log_interval=1)
    writer = SummaryWriter()
    iter_idx = 0
    epi_idx = 0
    # print(id(agents[0][1].brain), id(agents[4][0].brain))
    # input()
    # input()
    T = 0
    while 1:
        T += 1
        time_stamp  = []
        actions_n = []
        for i in range(num_process):
            action_i = []
            for j in range(len(obses_n[i])):
                # if obses_n[i][j].reward > 0:
                #     print(obses_n[i][j].reward)
                # print(obses_n[i][0].done == obses_n[i][1].done)
                # input()
                if not obses_n[i][j].done:
                    action = agents[i][j].think(sp_ac=ppo.target_net,callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                    # print(action)
                    # input()
                    # if len(memory) % (update_steps * num_process) == 0:
                    if T % (update_steps * num_process) == 0:
                        # print(memory.__len__())
                        ppo.update(memory, iter_idx, callback=logger, device=device)
                        iter_idx += 1
                else:
                    action = [] # reset
                    epi_idx += .5
                    time_stamp.append(obses_n[i][j].info["time_stamp"])
                    agents[i][j].sum_up(sp_ac=ppo.target_net,callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                    # algo.update(memory, iter_idx, callback=logger, device=device)
                    # iter_idx += 1
                    agents[i][j].forget()
                    # print(i, j)
                action_i.append(action)
                if (epi_idx + 1) % 100 == 0:
                    torch.save(nn.state_dict(), os.path.join(settings.models_dir, 'rl_fg' + str(int(epi_idx)) + ".pth"))
            actions_n.append(action_i)
    
           
        # if obses_n[0][0].done:
        #     print(2)
        #     print(actions_n)
        #     print(obses_n[0][0].info["time_stamp"])
        #     input()
        if time_stamp:
            # print("logged", iter_idx)
            writer.add_scalar("TimeStamp", sum(time_stamp) / (len(time_stamp)), epi_idx)
       
        # if obses_n[0][0].done:
        #     obses_n = envs.step(actions_n)
        #     print(obses_n[0][0].observation == bg_state.observation)
        #     input()
        # else:
        obses_n = envs.step(actions_n)
        

        # print(time.time() - _st)

        frames += 1
        
        # print(time.time() - st)
        
        # if memory.__len__() >= update_steps * num_process:
        #     algo.update(memory, iter_idx, callback=logger, device=device)
        #     iter_idx += 1
        
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
        '-lr',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--entropy',
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
        default=1
    )
    parser.add_argument(
        '--opponent',
        default="socketAI"
    )
    parser.add_argument(
        "--saving-prefix",
        default='rl',
    )
    args = parser.parse_args()
    print(args)
    torch.manual_seed(0)
    play("doubleBattle-v0") #, nn_path=os.path.join(settings.models_dir,"rl39699.pth"))

