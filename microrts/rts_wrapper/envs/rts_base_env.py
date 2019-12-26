from typing import List
import os
from subprocess import PIPE, Popen
from gym import spaces
import gym

from .datatypes import Config, UnitValidAction, AGENT_ACTIONS_MAP, BaseAction, BarracksAction, WorkerAction, \
    LightAction, HeavyAction, RangedAction
from .utils import network_action_translator, get_available_port
import microrts.settings as settings


class BaseEnv(gym.Env):
    """Used to create the trainning environment for RL
    and EVERYTHING correlated to the java game Env communication should be processed here
    
    Arguments:
        gym {[type]} -- [description]
    
    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]
    
    Returns:
        [type] -- [description]
    """
    setup_commands = None
    config = None
    DEBUG = False
    # players = []

    def __init__(self, config: Config):
        """
        initialize, do not call any other member function, leaving it to the successor
        :param config:
        """
        self.config = config
        self._init_client()
        # self._counting_players()
        self.action_space = ({
            'Base': spaces.Discrete(BaseAction.__members__.items().__len__()),
            'Barracks': spaces.Discrete(BarracksAction.__members__.items().__len__()),
            'Worker': spaces.Discrete(WorkerAction.__members__.items().__len__()),
            'Light': spaces.Discrete(LightAction.__members__.items().__len__()),
            'Heavy': spaces.Discrete(HeavyAction.__members__.items().__len__()),
            'Ranged': spaces.Discrete(RangedAction.__members__.items().__len__()),
        })

    @property
    def map_size(self):
        return self.config.height, self.config.width

    @property
    def max_episodes(self):
        return self.config.max_episodes

    @property
    def ai1_type(self):
        return self.config.ai1_type

    @property
    def ai2_type(self):
        return self.config.ai2_type

    def _add_commands(self, option, args):
        assert isinstance(option, str), "Option should be string"
        assert option.startswith("--"), "Invalid option"
        assert args is not None, "Args should be filled"

        self.setup_commands.append(option)
        self.setup_commands.append(args)

    def _init_client(self):
        """
        before-interacting setting-ups, and open the java program. Need to add port in kids
        """
        self.setup_commands = [
            "java",
            "-jar", settings.jar_dir,
            "--map", os.path.join(self.config.microrts_path, self.config.map_path),
            "--ai1_type", self.config.ai1_type,
            "--ai2_type", self.config.ai2_type,
            "--maxCycles", str(self.config.max_cycles),
            "--maxEpisodes", str(self.config.max_episodes),
            "--period", str(self.config.period),
            "--render", str(self.config.render),
            # "--port", str(self.port),
            # "more",
            # "options"
        ]

    def start_client(self):
        print(' '.join(self.setup_commands))
        java_client = Popen(
            self.setup_commands,
            stdin=PIPE,
            stdout=PIPE
        )
        stdout, stderr = java_client.communicate()
        print(stdout.decode("utf-8"))
        pass

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        """
        all players say hello to server
        """
        raise NotImplementedError

    def render(self, mode='human'):
        pass


