import socket
from microrts.algo.utils import load_model
from .utils import signal_wrapper, network_action_translator, pa_to_jsonable, action_sampler_v1


class Player(object):
    """Part of gym Environment
    """
    conn = None
    type = None
    port = None
    brain = None
    client_ip = None
    id = None

    action = []

    def __str__(self):
        pass

    def __init__(self, pid, client_ip, port):
        self.id = pid
        self.port = port
        self.client_ip = client_ip

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
        server_socket.bind((self.client_ip, self.port))
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
            kwargs:
                model
                obs
                info
        Raises:
            NotImplementedError: [description]
        """
        assert self.brain is not None
        obs = kwargs["obs"]
        info = kwargs["info"]
        return action_sampler_v1(self.brain, obs, info)

    def act(self, action):
        """
        do some action in the env together with other players
        """
        action = network_action_translator(action)
        pa = pa_to_jsonable(action)
        self._send_msg(pa)

    def observe(self):
        """
        observe the feedback from the env
        """
        raw = self._recv_msg()
        return signal_wrapper(raw)

    def expect(self):
        return self._recv_msg()

    def _send_msg(self, msg: str):
        try:
            self.conn.send(('%s\n' % msg).encode('utf-8'))
        except Exception as err:
            print("An error has occured: ", err)
        # return self.conn.recv(65536).decode('utf-8')

    def _recv_msg(self):
        return self.conn.recv(65536).decode('utf-8')


if __name__ == "__main__":
    print("OK")