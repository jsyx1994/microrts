
from microrts.rts_wrapper.envs.utils import action_sampler_v2, get_action_index, action_sampler_v1

class Agent:

    brain = None
    units_on_working = {}


    def __init__(self, model):
        self.brain = model

    def forget(self):
        self.units_on_working.clear()

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
        player_actions = action_sampler_v2(self.brain, obs, info, device=device)
        # print(time.time() - st)
        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in player_actions]

        transition = {}
        if mode == "train":
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
        
        if transition and callback:
            callback(transition)
        
        return player_actions