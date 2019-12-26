import socket
from microrts.algo.utils import load_model
from .utils import signal_wrapper, network_action_translator, pa_to_jsonable, action_sampler_v1, get_action_index
from microrts.algo.replay_buffer import ReplayBuffer
# import microrts.algo.replay_buffer as rb

class Player(object):
    """Part of the gym environment, need to handle issues with java end interaction
    """
    conn = None
    type = None
    port = None
    _client_ip = None
    id = None


    # very long memory
    brain = None

    # long memory
    _memory = None

    # short memories
    last_actions = None
    units_on_working = {}
    player_actions = None   # used to interact with env

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __init__(self, pid, client_ip, port, memory_size=10000):
        self.id = pid
        self.port = port
        self._client_ip = client_ip

        self._memory = ReplayBuffer(memory_size)
    
    def forget(self):
        """Forget what short memories stored, remain the long and very long 's
        """
        self.last_actions = None
        self.units_on_working.clear()
        self.player_actions = None

    def load_brain(self, nn):
        self.brain = nn
    
    def _memorize(self, obs_t, action, obs_tp1, reward, done):
        assert self._memory is not None
        self._memory.push(
            obs_t=obs_t,
            action=action,
            obs_tp1=obs_tp1,
            reward=reward,
            done=done,
            )
        if reward > 0:
            print(action)

    
    def learn(self, algo=None):
        assert algo is not None
        raise NotImplementedError

    def join(self):
        """
        hand shake with java end
        """
        server_socket = socket.socket()
        server_socket.bind((self._client_ip, self.port))
        server_socket.listen(5)
        print("Player{} Wait for Java client connection...".format(self.id))
        self.conn, address_info = server_socket.accept()

        self.greetings()

    def greetings(self):
        print("Player{}: Send welcome msg to client...".format(self.id))
        self._send_msg("Welcome msg sent!")
        print(self._recv_msg())

    def reset(self):
        print("Server: Send reset command...")
        self._send_msg('reset')
        raw = self._recv_msg()
        # print(raw)
        return signal_wrapper(raw)

    def think(self, **kwargs):
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

        self.player_actions = action_sampler_v1(self.brain, obs, info, device=device)

        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in self.player_actions]

        if mode == "train":
            # sample the transition in a correct way
            for u, a in network_unit_actions:
                _id = str(u.ID)
                if _id in self.units_on_working:
                    self._memorize(
                        obs_t=self.units_on_working[_id][0],
                        action=self.units_on_working[_id][1],
                        obs_tp1=obs,
                        reward=reward,
                        done=done,
                    )
                
                self.units_on_working[str(u.ID)] = (obs, a)
            

        # if not network_unit_actions:
        #     network_unit_actions = self.last_actions
        # self.last_actions = network_unit_actions
        # return network_unit_actions
        # return action_sampler_v1(self.brain, obs, info)

    def act(self):
        """
        do some action in the env together with other players
        """
        assert self.player_actions is not None
        action = network_action_translator(self.player_actions)
        pa = pa_to_jsonable(action)
        self._send_msg(pa)

    def observe(self):
        """
        observe the feedback from the env
        """
        raw = self._recv_msg()
        # print(raw)
        return signal_wrapper(raw)

    def expect(self):
        """Expecting and waiting for the msg from environment
        
        Returns:
            str -- the msg received from remote
        """
        return self._recv_msg()

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occurred: ", err)
        # return self.conn.recv(65536).decode('utf-8')

    def _recv_msg(self):
        return self.conn.recv(65536).decode('utf-8')


if __name__ == "__main__":
    print("OK")