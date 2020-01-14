import torch
from microrts.algo.model import ActorCritic
from microrts.algo.utils import load_model
from microrts.rts_wrapper.envs.multi_envs import make_env, make_vec_envs
from microrts.algo.agents import Agent
from torch.utils.tensorboard import SummaryWriter
from microrts.algo.replay_buffer import ReplayBuffer

from microrts.rts_wrapper.envs.utils import action_sampler_v2, network_simulator
from microrts.algo.a2c import A2C
from microrts.rts_wrapper.envs.datatypes import Config
import microrts.settings as settings 
import os, psutil

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
    map_size = config.height, config.width
    max_episodes = config.max_episodes

    memory = ReplayBuffer(10000)

    if start_from_scratch:
        nn = ActorCritic(map_size)
    else:
        nn = load_model(nn_path, map_size)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    # input()
    nn.share_memory()

    nn.to(device)

    num_process = 1
    envs = make_vec_envs(env_id, num_process, context="fork")


    import time

    # action = agent.think(callback=memo_inserter, obses=obses_n[0][0], accelerator=device, mode="train")
    # o = obses_n[0][0]
    # print(o)
    # st = time.time()
    # # action = agents[i][j].think(callback=memo_i nserter, obses=obses_n[i][j], accelerator=device, mode="train")
    # action = agent.think(callback=memo_inserter, obses=o, accelerator=device, mode="train")
    # print((time.time() - st))
    # input()


    frames = 0
    st = time.time()

    obses_n = envs.reset()
    agents = [[Agent(nn) for _ in obs] for obs in obses_n]
    # print(agents[1][0].brain is agents[4][0].brain)
    # print(len(agents))
    # input()
    update_steps = 20
    algo = A2C(nn, 2e-5, weight_decay=1e-3)
    writer = SummaryWriter()
    iter_idx = 0
    epi_idx = 0

    while 1:
        # for p in envs.procs:
        #     print(p.is_alive())
        actions_n = []
        for i in range(num_process):
            action_i = []
            for j in range(len(obses_n[i])):
                action = agents[i][j].think(callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                action_i.append(action)

            
            actions_n.append(action_i)
        # _st = time.time()
        # print(actions_n)
        # print()
        # input()
        obses_n = envs.step(actions_n)
        # print(time.time() - _st)

        frames += 1
        
        # print(time.time() - st)
        if frames % update_steps == 0:
            print(memory.__len__())
            algo.update(memory, iter_idx, callback=logger, device=device)
            iter_idx += 1
        
        # if memory.__len__() >= update_steps * num_process:
        #     algo.update(memory, iter_idx, callback=logger, device=device)
        #     iter_idx += 1
        



        if frames == 1000:
            print(frames * num_process / (time.time() - st))
            frames = 0
            st = time.time()
            # torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))

            
        time_stamp  = []
        for i in range(len(obses_n)):
            for j in range(len(obses_n[i])):
                # print(obses_n[i][j].done)
                if obses_n[i][j].done:
                    epi_idx += 1
                    if epi_idx % 5000 == 0:
                        torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl" + str(epi_idx) + ".pth"))
                    
                    time_stamp.append(obses_n[i][j].info["time_stamp"])
                    agents[i][j].forget()
                    # print("done")
                # if obs[0].done:
                #     print("DONE")
                # end_time += obs[0].info["time_stamp"]
                # end_num += 1
        if time_stamp:
            # print("logged", iter_idx)
            writer.add_scalar("TimeStamp", sum(time_stamp) / (len(time_stamp)), iter_idx)

        # print(x)

    # print(obses[0])
    # print(len(obses))

    # print(envs.get_players())
    # envs = ParallelVecEnv(envs)
    # input()

    # print(envs)
    # print(type(envs.reset()))


if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    # print(p.nice())
    p.nice(10)
    # input()
    play("attackHome-v0")

