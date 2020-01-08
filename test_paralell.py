import torch
from microrts.algo.model import ActorCritic
from microrts.algo.utils import load_model
from microrts.rts_wrapper.envs.multi_envs import make_env, make_vec_envs
from microrts.algo.agents import Agent
from torch.utils.tensorboard import SummaryWriter
from microrts.algo.replay_buffer import ReplayBuffer

from microrts.rts_wrapper.envs.utils import action_sampler_v2

def play(env_id, nn_path=None):
    def get_map_size():
        from microrts.rts_wrapper import environments
        for registered in environments:
            if registered["id"] == env_id:
                return registered['kwargs']['config'].height, registered['kwargs']['config'].width
    
    def logger(iter_idx, results):
        for k in results:
            writer.add_scalar(k, results[k], iter_idx)

    def memo_inserter(transitions):
        memory.push(**transitions)


    start_from_scratch = nn_path is None
    map_size = get_map_size()

    memory = ReplayBuffer(10000)
    writer = SummaryWriter()

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

    num_process = 8
    envs = make_vec_envs(env_id, nn, num_process, context="fork")

    obses_n = envs.reset()
    action_sampler_v2(nn, obses_n[0][0].observation, obses_n[0][0].info, device=device)

    agent = Agent(nn)
    obs = obses_n[0][0]
    import time

    st = time.time()

    print(time.time() - st)
    action_sampler_v2(nn, obs.observation, obs.info, device=device)
    print((time.time() - st))
    input()
    # action = agent.think(callback=memo_inserter, obses=obses_n[0][0], accelerator=device, mode="train")
    # o = obses_n[0][0]
    # print(o)
    # st = time.time()
    # # action = agents[i][j].think(callback=memo_i nserter, obses=obses_n[i][j], accelerator=device, mode="train")
    # action = agent.think(callback=memo_inserter, obses=o, accelerator=device, mode="train")
    # print((time.time() - st))
    # input()

    # agents = [[Agent(nn) for _ in obs] for obs in obses_n]

    frames = 0
    st = time.time()

    while 1:
        actions_n = []
        for i in range(num_process):
            action_i = []
            for j in range(len(obses_n[i])):
                # st = time.time()
                # action = agents[i][j].think(callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                action = agent.think(callback=memo_inserter, obses=obses_n[i][j], accelerator=device, mode="train")
                
                # print((time.time() - st))

                action_i.append(action)

            
            actions_n.append(action_i)
        obses_n = envs.step(actions_n)
        frames += 1


        # print(time.time() - st)
        if frames == 1000:
            print(frames * num_process / (time.time() - st))
        
        # print(x)

    # print(obses[0])
    # print(len(obses))

    # print(envs.get_players())
    # envs = ParallelVecEnv(envs)
    input()

    print(envs)
    print(type(envs.reset()))


if __name__ == "__main__":
    play("CurriculumBaseWorker-v0")

