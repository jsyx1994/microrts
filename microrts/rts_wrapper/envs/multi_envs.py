import multiprocessing as mp
import gym
import torch

from .vec_env import VecEnv

from microrts.algo.model import ActorCritic
from microrts.algo.utils import load_model, CloudpickleWrapper, clear_mpi_env_vars
from microrts.algo.agents import Agent
from .utils import get_config
import time

def make_env(env_id):
    def _thunk():
        env = gym.make(env_id)
        # print(env.players[0].brain is env.players[1].brain)
        return env
    
    return _thunk


def make_vec_envs(env_id, num_processes, context, model):
    
    assert num_processes > 0, "Can not make no env!"
    envs = [make_env(env_id) for i in range(num_processes)]

    # print(envs[0]().players[0].brain is envs[1]().players[0].brain)
    # env1 = envs[0]().players[0].brain
    # env2 = envs[2]().players[0].brain
    
    # input()
    config = get_config(env_id)
    nagents = 2 if (config.ai1_type == "socketAI" and config.ai2_type == "socketAI") else 1

    agents = [[Agent(model) for _ in range(nagents)] for _ in range(num_processes)]


    envs = ParallelVecEnv(envs, context=context)
    return envs, agents


# def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys):
def _subproc_worker(pipe, parent_pipe, env_fn_wrapper):

    """
    Control a single environment instance using IPC and
    shared memory.
    """

    env = env_fn_wrapper()
    # print(env)
    # sleep(5)
    # env = env_fn_wrapper()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(env.reset())
            elif cmd == 'step':
                obs = env.step(data)
                # if obs[0].reward > 0:
                #     print(obs[0].reward)
                # if obs[0].done:
                #     obs = env.reset()
                #     for agent in agents_process:
                #         # print(agent)
                #         # time.sleep(5)
                #         agent.forget()
                pipe.send(obs)
            # elif cmd == 'getp':
            #     # print("int hitsafsd")
            #     player = env.get_players()
            #     print(player)
            #     pipe.send("1")
            # elif cmd == 'render':
            #     pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ParallelVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close() 


class ParallelVecEnv(VecEnv):
    """[summary]
    """
    def __init__(self, env_fns, context='spawn', ):
        self.waiting_step = False
        self.closed  = False
                
        ctx = mp.get_context(context)
        # if spaces:
        #     observation_space, action_space = spaces
        # else:
        #     logger.log('Creating dummy env object to get spaces')
        #     with logger.scoped_configure(format_strs=[]):
        #         dummy = env_fns[0]()
        #         observation_space, action_space = dummy.observation_space, dummy.action_space
        #         dummy.close()
        #         del dummy
        # VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        # self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        # self.obs_bufs = [
        #     {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
        #     for _ in env_fns]
        
        self.parent_pipes = []
        self.procs = []
        for i, env_fn in enumerate(env_fns):
            # wrapped_fn = CloudpickleWrapper(env_fn)
            wrapped_fn = env_fn

            parent_pipe, child_pipe = ctx.Pipe()
            # proc = ctx.Process(target=_subproc_worker,
            #             args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
            proc = ctx.Process(target=_subproc_worker,
                        args=(child_pipe, parent_pipe, wrapped_fn))
            # proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False
        self.viewer = None
        super(ParallelVecEnv, self).__init__(len(env_fns))
    

    # def get_players(self):
    #     for pipe in self.parent_pipes:
    #         pipe.send(('getp',None))

    #     return [pipe.recv() for pipe in self.parent_pipes]
    
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        if self.waiting_step:
            # logger.warn('Called reset() while waiting for the step to complete')
            print('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return [pipe.recv() for pipe in self.parent_pipes]


    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            # print(len(act))
            pipe.send(('step', act))
        self.waiting_step = True


    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """

        obs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        return obs

    # def step(self, actions):
    #     self.step_async(actions)
    #     return self.step_wait()


def play(env_id, nn_path=None):
    def get_map_size():
        from microrts.rts_wrapper import environments
        for registered in environments:
            if registered["id"] == env_id:
                return registered['kwargs']['config'].height, registered['kwargs']['config'].width
    
    start_from_scratch = nn_path is None
    map_size = get_map_size()


    if start_from_scratch:
        nn = ActorCritic(map_size)
    else:
        nn = load_model(nn_path, map_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    nn.share_memory()

    nn.to(device)

    envs = make_vec_envs(env_id, nn, 8, context="fork")
    # envs = ParallelVecEnv(envs)
    input()

    print(envs)
    print(type(envs.reset()))

if __name__ == "__main__":
    play("singleBattle-v0")

