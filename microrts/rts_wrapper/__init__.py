from gym.envs.registration import register
import os
from .envs.datatypes import Config
import microrts.settings as settings
from gym import envs
import copy

# settings.map_dir

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

environments = [
    {
        'id': "SingleAgent-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs': 
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='Passive',
                    map_path=os.path.join(settings.map_dir, '6x6/attackHome6x6.xml'),
                    height=6,
                    width=6,
                    max_cycles=1000,
                    max_episodes=1000,
                    ),
            }
    },

    {
        'id': "attackHome-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='Passive',
                    map_path=os.path.join(settings.map_dir, '4x4/attackHome4x4.xml'),
                    height=4,
                    width=4,
                    max_cycles=1000,
                    max_episodes=1000000,
                ),

            }
    },

    {
        'id': "attackHome-v1",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='Passive',
                    map_path=os.path.join(settings.map_dir, '4x4/attackHome4x4-v1.xml'),
                    height=4,
                    width=4,
                    max_cycles=1000,
                    max_episodes=1000,
                ),

            }
    },

    {
        'id': "singleBattle-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '4x4/singleBattle4x4.xml'),
                    height=4,
                    width=4,
                    self_play=True,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=1000000000,
                ),

            }
    },
    {
        'id': "doubleBattle-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '4x4/doubleBattle4x4.xml'),
                    height=4,
                    width=4,
                    self_play=True,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=1000000000,
                ),

            }
    },
    {
        'id': "tripleBattle-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '4x4/tripleBattle4x4.xml'),
                    height=4,
                    width=4,
                    self_play=True,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=1000000000,
                ),

            }
    },
    {
        'id': "LightMelee-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '6x6/LightMelee6x6.xml'),
                    height=4,
                    width=4,
                    self_play=True,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=10000,
                ),

            }
    },
    {
        'id': "fullgame-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '6x6/fullgame6x6.xml'),
                    height=6,
                    width=6,
                    self_play=True,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=1000000000,
                ),

            }
    },

    {
        'id': "battle2v2LightMelee-v0",
        'entry_point': "microrts.rts_wrapper.envs:BattleEnv",
        'kwargs':
            {
                'config': Config(
                    ai1_type='socketAI',
                    # ai2_type='NaiveMCTS',
                    # ai2_type='WorkerRush',
                    ai2_type='Random',
                    # ai2_type='RandomBiased',
                    # ai2_type='socketAI',
                    self_play=True,
                    map_path=os.path.join(settings.map_dir, '6x6/battle2v2LightMelee6x6.xml'),
                    height=6,
                    width=6,
                    # period=20,
                    max_cycles=1000,
                    max_episodes=50000,
                ),
            }
    },

    {
        "id": 'CurriculumBaseWorker-v0',
        "entry_point": 'microrts.rts_wrapper.envs:BattleEnv',
        "kwargs":
            {
                'config': Config(
                    ai1_type='socketAI',
                    ai2_type='socketAI',
                    map_path=os.path.join(settings.map_dir, '6x6/baseWorkerResources6x6.xml'),
                    height=6,
                    width=6,
                    self_play=True,
                    # render=1,
                    max_cycles=3000,
                    max_episodes=10000,
                ),
            }
    },

]

eval_envs = []
for env in environments:
    # env for training 
    env["kwargs"]["config"].render=0
    env["kwargs"]["config"].period=1
    # print(env)
    register(
        id=env["id"],
        entry_point=env["entry_point"],
        kwargs=env["kwargs"]
    )

    eval_env = copy.deepcopy(env)
    eval_env["kwargs"]["config"].render=1
    eval_env["kwargs"]["config"].period=20
    eval_env["id"] = "Eval" + eval_env["id"]
    # print(env["id"])
    eval_envs.append(eval_env)
    register(
        id=eval_env["id"],
        entry_point=eval_env["entry_point"],
        kwargs=eval_env["kwargs"]
    )

environments.extend(eval_envs)

# register(
#     id='CurriculumBaseWorker-v0',
#     entry_point='microrts.rts_wrapper.envs:BattleEnv',
#     kwargs={'config': Config(
#         ai1_type='socketAI',
#         ai2_type='socketAI',
#         map_path=os.path.join(base_dir_path, 'maps/6x6/baseWorkerResources6x6.xml'),
#         height=6,
#         width=6,
#         render=1,
#         max_cycles=3000,
#         max_episodes=10000,

#     )}
# )

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