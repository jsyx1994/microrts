
from microrts.rts_wrapper.envs.utils import action_sampler_v2, get_action_index, action_sampler_v1,network_simulator
from microrts.algo.replay_buffer import ReplayBuffer
class Agent:

    brain = None
    units_on_working = {}
    steps = 0
    _memory = None


    def __init__(self, model, memory_size=10000, random_rollout_steps=128):
        self.brain = model
        self.random_rollout_steps = random_rollout_steps
        self._memory = ReplayBuffer(memory_size)

    def forget(self):
        self.units_on_working.clear()
        self.steps = 0
    
    def get_memory(self):
        return self._memory
    
    def sum_up(self, callback=None, **kwargs):
        self.think(callback, **kwargs)

    def think(self, callback=None, **kwargs):
        """figure out the action according to helper function and obs, store env related action to itself and \
            nn related result to Replay Buffer. More
        Arguments:
            kwargs: 
                obses {dataclass} -- observation: Any,  reward: float , done: bool, info: Dict
                accelerator {str} -- device to load torch tensors
                mode {str} -- "train" or "eval", if "train" is present, then transitions should be sampled
        Returns:
            [(Unit, int)] -- list of NETWORK unit actions    
        """
        assert self.brain is not None
        # self.player_actions.clear()
        obses = kwargs["obses"]

        obs     = obses.observation
        info    = obses.info
        reward  = obses.reward
        done    = obses.done

        if "accelerator" in kwargs:
            device = kwargs["accelerator"]
        else:
            device = "cpu"
        
        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            mode = "eval"
        
        del kwargs

        # import time

        # st = time.time()
        # sampler = action_sampler_v2 if self.random_rollout_steps <= self.steps or mode == "eval" else network_simulator
        # print(sampler)
        # input()
        sampler = action_sampler_v2
        self.steps += 1
        player_actions = sampler(
            info=info,
            model=self.brain,
            state=obs,
            device=device)
        # print(time.time() - st)
        # print(player_actions)
        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in player_actions]

        transition = {}
        if mode == "train" and sampler is not network_simulator:
            # sample the transition in a correct way
            for u, a in network_unit_actions:
                _id = str(u.ID)
                if _id in self.units_on_working:
                    transition = {
                        "obs_t":self.units_on_working[_id][0],
                        "action":self.units_on_working[_id][1],
                        "obs_tp1":obs,
                        "reward":reward,
                        "done":done,
                        }   
                    # self._memorize(
                        # obs_t=self.units_on_working[_id][0],
                        # action=self.units_on_working[_id][1],
                        # obs_tp1=obs,
                        # reward=reward,
                        # done=done,
                    #     )
                
                self.units_on_working[str(u.ID)] = (obs, (u, a))
            # push to agents' memory
            if transition:
                self._memory.push(**transition)
        
        if transition and callback:
            callback(transition)
        
        return player_actions