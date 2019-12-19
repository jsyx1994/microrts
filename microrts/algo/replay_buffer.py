import numpy as np
import random
from dataclasses import dataclass
from collections import namedtuple
from microrts.rts_wrapper.envs.datatypes import GameState, List, PlayerAction

@dataclass
class Prerequisite:
    state  : np.array
    info   : dict

@dataclass
class Transition:
    obs_t   : Prerequisite
    joint_action  : List[PlayerAction]
    reward  : float
    obs_tp1 : Prerequisite

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len=1):
        """Create Replay buffer
        Arguments:
            size {int} -- Storage capacity i.e. xax number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        
        Keyword Arguments:
            frame_history_len {int} -- Num of frames taken for trainning input (default: {1})
        """

        self._storage = []
        self._maxsize = size
        self._next_idx = 0        # next pos to store the new data
        self._frame_history_len = frame_history_len

    def __len__(self):
        """Show current capacity
        
        Returns:
            int -- how many samples stored in the buffer?
        """
        return len(self._storage)

    def push(self, s_t, info_t, joint_action, reward, s_tp1, info_tp1, done):
        """Saves a transition
        
        Arguments: 
        args:
            obs_t {np.array} -- [description]
            info_t {dict} -- [description]
            action {List of Player Action} -- [description]
            reward {float} -- [description]
            obs_tp1 {np.array} -- [description]
            info_tp1 {dict} -- [description]
            done {bool} -- [description]
        """
        trans = Transition(
            obs_t=Prerequisite(s_t, info_t),
            action=joint_action,
            reward=reward,
            obs_tp1=Prerequisite(s_tp1, info_tp1)
        )
        if self._next_idx >= len(self._storage):
            self._storage.append(trans)
        else:
            self._storage[self._next_idx] = trans
        self._next_idx = (self._next_idx + 1) % self._maxsize
    



if __name__ == '__main__':
    replay_buffer = ReplayBuffer(size=4)
    pass