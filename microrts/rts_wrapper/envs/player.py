import socket
# from microrts.algo.utils import load_model
from .utils import signal_wrapper, network_action_translator, pa_to_jsonable, action_sampler_v1, get_action_index
from microrts.algo.replay_buffer import ReplayBuffer
import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import action_sampler_v2

# import microrts.algo.replay_buffer as rb

class Player(object):
    """Part of the gym environment, need to handle low-level issues with java end interaction
    some of the member function are deprecated because of contianing high-level operations (Moved to
    microrts.algo.agents)
    
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
    
    # def forget(self):
    #     """Forget what short memories stored, remain the long and very long 's
    #     """
    #     self.last_actions = None
    #     self.units_on_working.clear()
    #     self.player_actions = None

    def join(self):
        """
        hand shake with java end
        """
        server_socket = socket.socket()
        server_socket.bind((self._client_ip, self.port))
        server_socket.listen()
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
        # print("waiting")
        raw = self._recv_msg()
        # print("received")
        # print(raw)
        return signal_wrapper(raw)
            
    def act(self, action):
        """Do some action according to action_sampler in the env together with other players
        """
        # assert self.player_actions is not None
        assert action is not None
        action = network_action_translator(action)
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