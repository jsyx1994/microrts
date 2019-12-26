import gym
import os
from microrts.rts_wrapper.envs.datatypes import List, Any
from microrts.rts_wrapper.envs.utils import unit_feature_encoder,network_action_translator, encoded_utt_dict, network_simulator
import torch
from microrts.algo.replay_buffer import ReplayBuffer
from microrts.algo.model import ActorCritic
import microrts.settings as settings
from microrts.rts_wrapper.envs.utils import action_sampler_v1


from microrts.algo.utils import load_model
from microrts.algo.model import ActorCritic
from microrts.algo.replay_buffer import ReplayBuffer


def evaluate():
    """deprecated"""
    env = gym.make("EvalAgainstRandom-v0")
    players = env.players
    assert env.player1 is not None, "player No.1 can not be missed"
    eval_model = load_model(os.path.join(settings.models_dir, "_1M.pth"), env.map_size)
    env.player1.load_brain(eval_model)
    # env.player1.load_brain(os.path.join(settings.models_dir, "1M.pth"), env.map_size[0], env.map_size[1])
    # input()
    for _ in range(env.max_episodes):
        obses = env.reset()  # p1 and p2 reset
        while not obses[0].done:
            actions = []
            for i in range(len(players)):
                # players[i].think(obses[i])
                # print(players[i].think(action_sampler_v1, obs=obses[i].observation, info=obses[i].info))
                actions.append(players[i].think(obs=obses[i].observation, info=obses[i].info))
                # input()
                # actions.append(network_simulator(obses[i].info["unit_valid_actions"]))
            obses = env.step(actions)
            # print(obses)
        winner = env.get_winner()
        print(winner)

    print(env.setup_commands)


def self_play(nn_path=None):
    """self play program
    
    Arguments:
        nn_path {str} -- path to model, if None, start from scratch
        map_size {tuple} -- (height, width)
    """     

    env = gym.make("attackHome-v0")
    # assert env.ai1_type == "socketAI" and env.ai2_type == "socketAI", "This env is not for self-play"
    memory = ReplayBuffer(10000)

    start_from_scratch = nn_path is None
    
    players = env.players

    if start_from_scratch:
        nn = ActorCritic(env.map_size)
    else:
        nn = load_model(nn_path, env.map_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    nn.to(device)
    from torch.utils.tensorboard import SummaryWriter
    import time
    writer = SummaryWriter()



    for p in players:
        p.load_brain(nn)
    
    

    iter_idx = 0
    for epi_idx in range(env.max_episodes):
        obses_t = env.reset()  # p1 and p2 reset
        start_time = time.time()
        while not obses_t[0].done:
            # actions = []
            for i in range(len(players)):
                # actions.append(players[i].think(obs=obses_t[i].observation, info=obses_t[i].info, accelerator=device))
                players[i].think(obses=obses_t[i], accelerator=device, mode="train")
            obses_tp1 = env.step()
            if obses_tp1[0].reward > 0:
                print(obses_tp1[0].reward)






            memory.refresh()
            optimizer = torch.optim.RMSprop(nn.parameters(),lr=1e-5,weight_decay=1e-7)
            iter_idx += 1
            # for i in range(len(players)):
                # players[i].memorize(
                #     obs_t=obses_t[i].observation,
                #     action=actions[i],
                #     obs_tp1=obses_tp1[i].observation,
                #     reward=obses_tp1[i].reward,
                #     done=obses_tp1[i].done
                #     )

                # if actions[i]:
                #     memory.push(
                        # obs_t=obses_t[i].observation,

                        # action=actions[i],
                        # obs_tp1=obses_tp1[i].observation,
                        # reward=obses_tp1[i].reward,
                        # done=obses_tp1[i].done
                #     )

            # sps_dict = memory.sample(batch_size="all")
            # for key in sps_dict:
            #     if key not in nn.activated_agents:
            #         continue

            #     if sps_dict[key]:
            #         states, units, actions, next_states, rewards,  done_masks = sps_dict[key].to(device)


            #         value, probs = nn.forward(actor_type=key,spatial_feature=states,unit_feature=units)

            #         value_next = nn.critic_forward(next_states)
                    
            #         if rewards[0][0] > 0:
            #             print(rewards)
            #         pi_sa = probs.gather(1, actions)
            #         entropy_loss = - probs * torch.log(probs)
            #         policy_loss = - torch.log(pi_sa + 1e-7) * (rewards + value_next - value)
            #         value_loss = torch.nn.functional.mse_loss(rewards + value_next, value)

            #         all_loss = policy_loss.mean() + value_loss.mean() # + .01 * entropy_loss.sum()
                    
            #         if iter_idx % 100 == 0:
            #             writer.add_scalar("p_loss", policy_loss.mean(), iter_idx)
            #             writer.add_scalar("v_loss", value_loss.mean(), iter_idx)
            #             writer.add_scalar("all_loss", all_loss, iter_idx)

            #         optimizer.zero_grad()
            #         all_loss.backward()
            #         optimizer.step()
            
            obses_t = obses_tp1
        winner = env.get_winner()
        writer.add_scalar("TimeStamp",obses_t[i].info["time_stamp"], epi_idx)
        print("Winner is:{}, FPS: {}".format(winner,obses_t[i].info["time_stamp"] / (time.time() - start_time)))
        
    print(env.setup_commands)
    torch.save(nn.state_dict(), os.path.join(settings.models_dir, "rl.pth"))



if __name__ == '__main__':
    from microrts.settings import models_dir
    import os
    # self_play(nn_path=os.path.join(models_dir, "rl.pth"))
    self_play()
    # evaluate()
# print(rts_wrapper.base_dir_path)
# print(os.path.join(rts_wrapper.base_dir_path, 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))
