
from microrts.rts_wrapper.envs.utils import action_sampler_v2, get_action_index, action_sampler_v1,network_simulator
from microrts.algo.replay_buffer import ReplayBuffer
import numpy as np
import torch
from microrts.rts_wrapper.envs.datatypes import AGENT_ACTIONS_MAP
import copy
from microrts.rts_wrapper.envs.utils import unit_feature_encoder

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
        return self._storage
    
    def flatten(self):
        return self._storage.reshape(-1, *self._shape[-2:])

class Agent:
    gamma = 0.99
    def __init__(self, model, memory_size=10000, random_rollout_steps=128, smooth_sample_ratio=None, map_size=(4,4)):
        self.rewards = 0
        self.steps = 0
        self.units_on_working = {}
        self._hidden_states = {} # id -> hidden_states
        self._frame_buffer = FrameBuffer(size=16,map_size=(4,4),feature_size=50)
        self.brain = model
        self.random_rollout_steps = random_rollout_steps
        self.smooth_sample_ratio = smooth_sample_ratio
        self.memory = ReplayBuffer(memory_size)
        self.map_size = map_size
        # print(self.gamma)
        # input()
        # self.last = []

    def forget(self):
        self.rewards = 0
        self.units_on_working.clear()
        self.steps = 0
        self._hidden_states.clear()
        # self._frame_buffer.refresh()
    
    
    def reward_util(self,info, obs_tp1, ev_sp, end_at, punish_ratio=-.001):
        """duration rewards craft
        
        Arguments:
            start_at {[type]} -- [description]
            end_at {[type]} -- [description]
        """
        # critical
        obs_t, ua, start_at, ev_s, = info
        # ua = info[1]
        # ev_s = info[3]
        # start_at = info[2]
        # u_x, u_y = ua[0].x, ua[0].y
        # ctf_obs = obs_tp1.copy()
        # ctf_obs[:][u_x][u_y] = 0
        # with torch.no_grad():
        #     diff_r = self.brain.critic_forward(torch.from_numpy(ctf_obs).unsqueeze(0).float()) \
        #         - self.brain.critic_forward(torch.from_numpy(obs_t).unsqueeze(0).float())
        #     print(diff_r)
            # input()
        reward = ev_sp - ev_s
        # unit. = units_on_field[_id].x
        self.rewards += reward
        return reward # punish_ratio #+ punish_ratio * (end_at - start_at)

    def ctf_adv(self,info,device):
        # return 0
        obs_t, ua, start_at, ev_s, = info
        obs_t = torch.from_numpy(obs_t).unsqueeze(0).float().to(device)
        unit, act = ua
        u_f = torch.from_numpy(unit_feature_encoder(unit,self.map_size)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            v, p, _ = self.brain.forward(obs_t, u_f, unit.type)
            # obs_act = copy.deepcopy(obs_t)
            # obs_act[0,-7:,unit.x, unit.y][act] = 1 
            # v = self.brain.critic_forward(obs_act)
            # p, _ = self.brain.actor_forward(unit.type, obs_t, u_f)

            u_x, u_y = unit.x, unit.y
            # print(torch.eye(AGENT_ACTIONS_MAP[ua[0].type].__members__.items().__len__(), 7))
            # input()
            act_list = [i for i in torch.eye(AGENT_ACTIONS_MAP[unit.type].__members__.items().__len__(), 7)]
            ctf_obs_li = []
            for x in act_list:
                obs_t[0,-7:,u_x,u_y] = x
                ctf_obs_li.append(copy.deepcopy(obs_t))
            
            ctf_v = self.brain.critic_forward(torch.cat(ctf_obs_li))

            adv = sum(p.squeeze() * ctf_v.squeeze()) 

        return float(adv)

    # def get_memory(self):
    #     return self._memory
    def semi_mdp_rew(self, dura_rews, discount):
        di_rew = .0
        for r in dura_rews[::-1]:
            di_rew = .99 * di_rew + r
        return di_rew
    
    def sum_up(self,sp_ac=None, callback=None, **kwargs):
        self.think(sp_ac,callback, **kwargs)

    def think(self,sp_ac=None, callback=None,way="stochastic", **kwargs):
        """call this function in every time step,figure out the action according to helper function and obs, store env related action to itself and \
            nn related result to Replay Buffer. More
        Arguments:
            kwargs: 
                obses {dataclass} -- observation: Any,  ev: float , done: bool, info: Dict
                accelerator {str} -- device to load torch tensors
                mode {str} -- "train" or "eval", if "train" is present, then transitions should be sampled
        Returns:
            [(Unit, int)] -- list of NETWORK unit actions    
        """
        def push2buffer():
            rewards = self.units_on_working[_id][3][1:]
            irew = self.gamma ** len(rewards) * rewards[-1]
            di_rew = self.semi_mdp_rew(rewards,0.99)
            print(di_rew)
            # input()
            transitions ={
                    "obs_t":self.units_on_working[_id][0],
                    "action":self.units_on_working[_id][1],
                    "obs_tp1":np.copy(obs),
                    # "reward":self.units_on_working[_id][3],
                    "reward":di_rew,
                    # "reward": self.reward_util(ev_sp=ev,obs_tp1=obs, end_at=time_stamp,info=self.units_on_working[_id]),
                    # "reward":reward - 0.1 * (time_stamp - self.units_on_working[_id][2]),
                    "hxs":self._hidden_states[_id] if _id in self._hidden_states else None,
                    "done":done,
                    "duration": time_stamp - self.units_on_working[_id][2],
                    "ctf": self.ctf_adv(self.units_on_working[_id],device),
                    "irew":irew,

                }
                # self.memory.push(**transitions)
            if callback:
                callback(transitions)
            # print(self.units_on_working[_id][3])


            # input()
    
        assert self.brain is not None
        # self.player_actions.clear()
        obses = kwargs["obses"]

        obs     = obses.observation
        info    = obses.info
        reward      = obses.reward 
        done    = obses.done
        time_stamp = info["time_stamp"]


        if "accelerator" in kwargs:
            device = kwargs["accelerator"]
        else:
            device = "cpu"
        
        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            mode = "eval"
        
        del kwargs

        for _id in self.units_on_working:
            # print(self.units_on_working[_id][3])
            self.units_on_working[_id][3].append(reward)
        
        # self._frame_buffer.push(obs)
        # # obs = self._frame_buffer.fetch()

        # obs = self._frame_buffer.flatten()

        if mode=='train' and done == 1: # gameover state, should not sample actions, add transition by force
            # print( info['unit_valid_actions']) # []
            for _id in self.units_on_working:
                push2buffer()
            return []


        sp_ac = sp_ac if sp_ac else self.brain
        # import torch
        # print(self.brain.critic_forward(torch.from_numpy(obs).float().unsqueeze(0)))
        # input()
        sp_ac.eval()

        sampler = action_sampler_v2

        if self.smooth_sample_ratio is not None:
            rollout = np.random.sample()
            if rollout < self.smooth_sample_ratio:
                sampler = network_simulator
                # print(sampler)
            # self.steps += 1
            # if self.steps > self.smooth_sample_step:
            #     sampler = network_simulator
            #     self.steps = 0
        

        samples, hxses = sampler(
                info=info,
                model=sp_ac,
                state=obs,
                device=device,
                mode=way,
                hidden_states=self._hidden_states,
                )


        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in samples]

        # if self.smooth_sample_ratio

        if self.brain.recurrent:
            for i, s in enumerate(samples): # figure out the subject of the hxses:
                self._hidden_states[str(s[0].unit.ID)] = hxses[:][i][:]


        if mode == "train" and sampler is not network_simulator:
        # if mode == "train" and sampler:
            # sample the transition in a correct way

            # check dead units and insert transaction
            # condition that unit is on working but is not on battlefield, update. Action cancel dead
            units_on_field = info["units_on_field"]
            key_to_del = []
            for _id in self.units_on_working:
                if int(_id) not in units_on_field:
                    key_to_del.append(_id)
                    push2buffer()
            for k in key_to_del:
                del self.units_on_working[k]

            for u, a in network_unit_actions:
                _id = str(u.ID)
                if _id in self.units_on_working: 
                    push2buffer()
                ####################################      0          1         2      3
                self.units_on_working[str(u.ID)] = [np.copy(obs), (u, a), time_stamp, [reward]]
        else:
            # print("network simulator")
            pass
        # print(unzip(*samples))
        # input()

        return samples