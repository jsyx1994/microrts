
from microrts.rts_wrapper.envs.utils import action_sampler_v2, get_action_index, action_sampler_v1,network_simulator
from microrts.algo.replay_buffer import ReplayBuffer
import numpy as np

class FrameBuffer:
    def __init__(self, map_size, feature_size,size=8):
        self._maxsize = size
        self._next_idx = 0        # next pos to store the new data
        h, w = map_size
        self._shape = (feature_size, h, w)

        self._storage = np.zeros((size, feature_size, h, w))
        pass
    
    def refresh(self):
        self._storage = np.zeros((self._maxsize, *self._shape))

    def __len__(self):
        return len(self._storage)
    
    def push(self, frame):
        self._storage[:-1] = self._storage[1:]
        self._storage[-1] = frame
        # if self._next_idx >= len(self._storage):
        #     self._storage.append(frame)
        # else:
        #     self._storage[self._next_idx] = frame
        # self._next_idx = (self._next_idx + 1) % self._maxsize
    def fetch(self):
        return self._storage.reshape(-1, *self._shape[-2:])

class Agent:
    def __init__(self, model, memory_size=10000, random_rollout_steps=128):
        self.units_on_working = {}
        self._hidden_states = {} # id -> hidden_states
        self._frame_buffer = FrameBuffer(size=16,map_size=(4,4),feature_size=48)
        self.brain = model
        self.random_rollout_steps = random_rollout_steps
        self._memory = ReplayBuffer(memory_size)

        # self.last = []

    def forget(self):
        self.units_on_working.clear()
        self.steps = 0
        self._hidden_states.clear()
        self._frame_buffer.refresh()

    
    def get_memory(self):
        return self._memory
    
    def sum_up(self, callback=None, **kwargs):
        self.think(callback, **kwargs)

    def think(self, callback=None,way="stochastic",**kwargs):
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


        # self._frame_buffer.push(obs)

        # obs = self._frame_buffer.fetch()
        # import torch
        # print(self.brain.critic_forward(torch.from_numpy(obs).float().unsqueeze(0)))
        # input()


        sampler = action_sampler_v2
        # self.steps += 1
        samples, hxses = sampler(
                info=info,
                model=self.brain,
                state=obs,
                device=device,
                mode=way,
                hidden_states=self._hidden_states,
                )
        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in samples]

        if self.brain.recurrent:
            for i, s in enumerate(samples): # figure out the subject of the hxses:
                self._hidden_states[str(s[0].unit.ID)] = hxses[:][i][:]

        transition = {}
        count = 0



        if mode == "train" and sampler is not network_simulator:
            # sample the transition in a correct way

            # check dead units and insert transaction
            # condition that unit is on working but is not on battlefield, update. Action cancel dead
            units_on_field = info["units_on_field"]
            key_to_del = []
            for _id in self.units_on_working:
                if int(_id) not in units_on_field and callback:
                    # print("yes")
                    # input()
                    key_to_del.append(_id)
                    print(done, reward)
                    # input()
                    callback({
                        "obs_t":self.units_on_working[_id][0],
                        "action":self.units_on_working[_id][1],
                        "obs_tp1":np.copy(obs),
                        "reward":reward,
                        "hxs":self._hidden_states[_id] if _id in self._hidden_states else None,
                        "done":done,
                    })
            for k in key_to_del:
                del self.units_on_working[k]


            for u, a in network_unit_actions:
                _id = str(u.ID)
                if _id in self.units_on_working:
                    # print(reward, a)
                    # input()
                    transition = {
                        "obs_t":self.units_on_working[_id][0],
                        "action":self.units_on_working[_id][1],
                        "obs_tp1":np.copy(obs),
                        "reward":reward,
                        "hxs":self._hidden_states[_id] if _id in self._hidden_states else None,
                        "done":done,
                        }
                    # print(reward)
                    # print(self.brain.critic_forward(torch.from_numpy(transition['obs_t']).float().unsqueeze(0)))
                    count += 1
                
                self.units_on_working[str(u.ID)] = (np.copy(obs), (u, a))
                # push to agents' memory
                # if transition:
                #     self._memory.push(**transition)
                #     transition.clear()
                if transition and callback:
                    callback(transition)
                    # print(count)
                    count = 0
                    transition.clear()
        # print(unzip(*samples))
        # input()


        return samples