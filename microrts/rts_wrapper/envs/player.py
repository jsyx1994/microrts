import socket
from microrts.algo.utils import load_model
from .utils import signal_wrapper, network_action_translator, pa_to_jsonable, action_sampler_v1, get_action_index


class Player(object):
    """Part of the gym environment, need to handle issues with java end interaction
    """
    conn = None
    type = None
    port = None
    brain = None
    _client_ip = None
    id = None
    last_actions = None
    
    memory = None

    player_actions = None

    def __str__(self):
        pass

    def __init__(self, pid, client_ip, port):
        self.id = pid
        self.port = port
        self._client_ip = client_ip

    # def load_brain(self, **kwargs):
    #     pass

    # def load_brain(self, nn_path, height, width):
    #     self.brain = load_model(nn_path, height, width)

    def load_brain(self, nn):
        self.brain = nn

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
        """figure out the action according to helper function and obs
        Arguments:
            kwargs: obs, info, accelerator
        Returns:
            [(Unit, int)] -- list of NETWORK unit actions    
        """
        assert self.brain is not None
        # self.player_actions.clear()
        obs = kwargs["obs"]
        info = kwargs["info"]
        if "accelerator" in kwargs:
            device = kwargs["accelerator"]
        else:
            device = 'cpu'

        self.player_actions = action_sampler_v1(self.brain, obs, info, device=device)

        network_unit_actions = [(s[0].unit, get_action_index(s[1])) for s in self.player_actions]
        # if not network_unit_actions:
        #     network_unit_actions = self.last_actions
        # self.last_actions = network_unit_actions
        return network_unit_actions
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