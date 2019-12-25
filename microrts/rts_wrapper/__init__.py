from gym.envs.registration import register
import os
from .envs.datatypes import Config
import microrts.settings as settings

base_dir_path = os.path.dirname(os.path.realpath(__file__))
# print(base_dir_path)
# register will call this:
"""A specification for a particular instance of the environment. Used
to register the parameters for official evaluations.
Args:
    id (str): The official environment ID
    entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
    reward_threshold (Optional[int]): The reward threshold before the task is considered solved
    kwargs (dict): The kwargs to pass to the environment class
    nondeterministic (bool): Whether this environment is non-deterministic even after seeding
    tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags
    max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
Attributes:
    id (str): The official environment ID
"""

register(
    id='SingleAgent-v0',
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path=os.path.join(base_dir_path, 'maps/6x6/attackHome6x6.xml'),
        height=6,
        width=6,
        render=0,
        max_cycles=1000,
        max_episodes=1000,

    )}
)

register(
    id='attackHome-v0',
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path=os.path.join(base_dir_path, 'maps/4x4/attackHome4x4.xml'),
        height=4,
        width=4,
        render=0,
        max_cycles=1000,
        max_episodes=1000,
    )}
)

register(
    id='attackHome-v1',
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path=os.path.join(base_dir_path, 'maps/6x6/attackHome6x6.xml'),
        height=6,
        width=6,
        render=0,
        max_cycles=1000,
        max_episodes=10000,
    )}
)

register(
    id='CurriculumBaseWorker-v0',
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='socketAI',
        map_path=os.path.join(base_dir_path, 'maps/6x6/baseWorkerResources6x6.xml'),
        height=6,
        width=6,
        render=1,
        max_cycles=3000,
        max_episodes=1000,

    )}
)

register(
    id='CurriculumBaseWorker-v1',
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type='socketAI',
        ai2_type='Passive',
        map_path=os.path.join(base_dir_path, 'maps/6x6/baseTwoWorkersMaxResources6x6.xml'),
        height=6,
        width=6,
        render=1,
        max_cycles=3000,
        max_episodes=10000,

    )}
)

# eval
register(
    id="EvalAgainstRandom-v0",
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="RandomBiased",
        map_path=os.path.join(base_dir_path, 'maps/6x6/baseTwoWorkersMaxResources6x6.xml'),
        height=6,
        width=6,
        render=1,
        period=1,
        max_cycles=3000,
        max_episodes=10000,
        )}
)

register(
    id="EvalAgainstPassive-v0",
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="Passive",
        map_path=os.path.join(base_dir_path, 'maps/6x6/baseTwoWorkersMaxResources6x6.xml'),
        height=6,
        width=6,
        render=1,
        period=1,
        max_cycles=3000,
        max_episodes=10000,
        )}
)

register(
    id="EvalAgainstSocketAI-v0",
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="socketAI",
        map_path=os.path.join(base_dir_path, 'maps/6x6/baseTwoWorkersMaxResources6x6.xml'),
        height=6,
        width=6,
        render=1,
        period=1,
        max_cycles=3000,
        max_episodes=10000,
        )}
)

# test
register(
    id="LargeMapTest-v0",
    entry_point='microrts.rts_wrapper.envs:BattleEnv',
    kwargs={'config': Config(
        ai1_type="socketAI",
        ai2_type="socketAI",

        map_path=os.path.join(base_dir_path, "maps/16x16/basesWorkers16x16.xml"),
        height=16,
        width=16,
        render=1,
        period=1,
        max_cycles=5000,
        max_episodes=10000,
        )}
)