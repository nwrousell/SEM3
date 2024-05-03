from replay_buffer import ReplayBuffer
import tensorflow as tf
from image_pre import process_inputs
import numpy as np
import yaml
from agent import Agent, linearly_decaying_epsilon
from time import time
import argparse
from atari import Atari
from atari import AtariMonitor
from logger import Logger

data_spec = [
    tf.TensorSpec(shape=(84,84), name="observation", dtype=np.float32),
    tf.TensorSpec(shape=(), name="action", dtype=np.int32),
    tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
    tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
]

train_log_fields = ['environment_step', 'gradient_step', 'spr_loss', 'td_error', 'num_episodes', 'episode_reward', 'episode_length']
eval_log_fields = ['environment_step', 'gradient_step', 'num_train_episodes', 'mean_episode_reward', 'mean_episode_length']
    
def train(agent: Agent, env, args, run_name):
    train_logger = Logger(args['summaries_dir']+run_name+"/", "train_log.csv", train_log_fields)
    eval_logger = Logger(args['summaries_dir']+run_name+"/", "eval_log.csv", eval_log_fields)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=agent.optimizer, model=agent.online_model)
    manager = tf.train.CheckpointManager(checkpoint, args['model_dir']+run_name, max_to_keep=3)
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=args['replay_capacity'], 
                                 batch_size=args['batch_size'], 
                                 update_horizon=args['start_update_horizon'], 
                                 gamma=args['start_gamma'], 
                                 n_envs=1, 
                                 stack_size=args['stack_frames'],
                                 subseq_len=args['subseq_len'],
                                 observation_shape=(84,84),
                                 rng=np.random.default_rng(seed=17)
                                )

    observation, _ = env.reset()
    
    episode_rewards = [0.0]
    max_mean_reward = None
    num_grad_steps = 0
    prev_num_episodes_log = -1
    spr_error = -1
    td_error = -1
    episode_length = 0

    
    start_time = time()
    
    if args['process_inputs']:
        observation = process_inputs(observation, scale_type=args['scale_type'])

    current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
    
    for t in range(args['num_env_steps']):
        epsilon = linearly_decaying_epsilon(args['epsilon_decay_period'], t, args['initial_collect_steps'], args['epsilon_train'])
        
        action = agent.choose_action(current_state, epsilon)
        observation, reward, terminated, _, _ = env.step(action)
        episode_length += 1
        
        if args['process_inputs']:
            observation = process_inputs(observation, scale_type=args['scale_type'])
        
        current_state = np.concatenate([current_state[:,:,1:], observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
        
        episode_rewards[-1] += reward
        
        if args['clip_reward']:
            reward = np.clip(reward, -1, 1)
            
        replay_buffer.add(observation, action, reward, np.array([int(terminated)]))
        
        if terminated:
            print("TERMINATED")
            log_data = {
                "environment_step": t,
                "gradient_step": num_grad_steps,
                "spr_loss": spr_error,
                "td_error": td_error,
                "num_episodes": len(episode_rewards),
                "episode_reward": episode_rewards[-1],
                "episode_length": episode_length
            }
            train_logger.log(log_data)
            observation, _ = env.reset()
            episode_length = 0
            if args['process_inputs']:
                observation = process_inputs(observation, scale_type=args['scale_type'])
            current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
            episode_rewards.append(0.0)
        
        if t > args['initial_collect_steps']:
            update_horizon = round(agent.update_horizon_scheduler(num_grad_steps % args['reset_every']))
            gamma = agent.gamma_scheduler(num_grad_steps % args['reset_every'])
            
            for s in range(args['replay_ratio']):
                num_grad_steps += 1
                batch = replay_buffer.sample_transition_batch(update_horizon=update_horizon,
                                                      gamma=gamma, 
                                                      subseq_len=update_horizon)
                    
                loss, td_error, spr_error = agent.train_step(update_horizon, *batch)
                
                if num_grad_steps % args['target_update_frequency'] == 0:
                    agent.update_target()
                
                if num_grad_steps % args['reset_every'] == 0:
                    print("WEIGHTS RESET")
                    agent.reset_weights()
        
        num_episodes = len(episode_rewards)
        if num_episodes > args['min_episodes']:
            mean_reward = np.mean(episode_rewards[-(args['min_episodes']+1):-1])
        
        if num_episodes > args['min_episodes'] and t > args['initial_collect_steps'] and t % args['print_frequency'] == 0:
            print(f"Gradient steps: {num_grad_steps}. Environment steps: {t}")
            if num_episodes != prev_num_episodes_log:
                elapsed = (time() - start_time) // 60
                print(f"Finished episode #{num_episodes-1} with reward: {episode_rewards[-2]}")
                prev_num_episodes_log = num_episodes
                print(f"- Num episodes: {num_episodes}\n- TD error: {td_error}\n- SPR error: {spr_error}\n- Time elapsed: {elapsed}m\n- Epsilon: {epsilon}\n- Update horizon: {update_horizon}\n- Gamma: {gamma}")
                if max_mean_reward is None or mean_reward > max_mean_reward:
                    max_mean_reward = mean_reward
                    print(f"improvement in mean_{args['min_episodes']}ep_reward: {max_mean_reward}")
                else:
                    print(f"No improvement in max mean_{args['min_episodes']}ep_reward. Achieved: {mean_reward}, max: {max_mean_reward}")
                print()
        
        if num_episodes > args['min_episodes'] and t > args['initial_collect_steps'] and t % args['eval_frequency'] == 0:
            checkpoint.step.assign_add(1)
            save_path = manager.save()
            print("saved current model")
            eval_mean_reward, eval_mean_length = evaluate(agent, env, args, run_name)
            print(f"Evaluation reward at {t} step is {eval_mean_reward}")
            log_data = {
                'environment_step': s,
                'gradient_step': num_grad_steps, 
                'num_train_episodes': len(episode_rewards),
                'mean_episode_reward': eval_mean_reward, 
                'mean_episode_length': eval_mean_length
            }
            eval_logger.log(log_data)
   
def evaluate(agent: Agent, env, args, run_name, restore=False, play=False):
    print("beginning evaluation")
    if restore:
        checkpoint = tf.train.Checkpoint(model=agent.online_model)
        latest_snapshot= tf.train.latest_checkpoint(args['model_dir']+run_name)
        # latest_snapshot = tf.train.latest_checkpoint(args['model_dir'])
        if not latest_snapshot:
            raise Exception(f"No model snapshot found in {args['model_dir']+run_name}")
        
        checkpoint.restore(latest_snapshot)
        print("Restored model from latest snapshot")
    
    observation, _ = env.reset()
    
    eval_episode_rewards = [0.0]
    eval_episode_lengths = [0]
    
    if args['process_inputs']:
        observation = process_inputs(observation, scale_type=args['scale_type'])

    current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
    epsilon = args['evaluation_epsilon']
    
    while True:
        action = agent.choose_action(current_state, epsilon)
        observation, reward, terminated, _, _ = env.step(action)

        if args['process_inputs']:
            observation = process_inputs(observation, scale_type=args['scale_type'])
            
        current_state = np.concatenate([current_state[:,:,1:], observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
        
        eval_episode_rewards[-1] += reward
        eval_episode_lengths[-1] += 1
        
        if terminated:
            print("TERMINATED")
            eval_mean_reward = np.mean(eval_episode_rewards)
            eval_mean_length = np.mean(eval_episode_lengths)
            observation, _ = env.reset()
            if args['process_inputs']:
                observation = process_inputs(observation, scale_type=args['scale_type'])
            current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], axis=-1)
            
            num_episodes = len(eval_episode_rewards)
            print(f"[Eval] Mean reward after {num_episodes} episodes is {round(eval_mean_reward, 2)}")
            if play:
                break
            if num_episodes >= args['evaluation_episodes']:
                break
            eval_episode_rewards.append(0.0)
            eval_episode_lengths.append(0)
    
    return round(eval_mean_reward, 2), eval_mean_length
        
def main():
    config_fname = "config.yaml"
    with open(config_fname, 'r') as file:
        config_args = yaml.safe_load(file)
        
    for k, v in config_args.items():
        print(f"{k}: {v}")   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train an agent to find optimal policy', action='store_true')
    parser.add_argument('--evaluate', help='evaluate trained policy of an agent', action='store_true')
    parser.add_argument('--play', help='let trained agent play', action='store_true')
    parser.add_argument('--name', help="name to use for checkpoint/logger/video files")
    parser.add_argument("--seed", help="seed to initialize RNGs")
    # parser.add_argument('--env', nargs=1, help='Atari 2600 game used as environment', type=str)
    
    terminal_args = parser.parse_args()
    
    print("seed:", terminal_args.seed)

    render_mode = 'human' if terminal_args.play else 'rgb_array'
    
    env = Atari("../roms/assault.bin")
    # env = gym.make(config_args['game'], 
    #                render_mode="rgb_array", 
    #                obs_type='grayscale', 
    #                frameskip=config_args['frameskip'])
    
    # n_actions = env.action_space.n
    n_actions = env.n_actions
    
    agent = Agent(config_args['stack_frames'], 
                  config_args['encoder_network'],
                  n_actions,
                  config_args['hidden_dim'],
                  config_args['learning_rate'],
                  config_args['weight_decay'],
                  config_args['start_gamma'],
                  config_args['end_gamma'],
                  config_args['target_action_selection'],
                  config_args['spr_loss_weight'],
                  config_args['start_update_horizon'],
                  config_args['end_update_horizon'],
                  config_args['target_ema_tau'],
                  config_args['shrink_factor'],
                  config_args['spr_prediction_depth'],
                  (84, 84),
                  config_args['renormalize'], # ! not implemented
                  config_args['double_DQN'],
                  config_args['distributional_DQN'],
                  config_args['dueling_DQN'],
                  config_args['vmax'],
                  config_args['num_atoms'],
                  terminal_args.seed,
                  config_args['data_augmentation'],
                  config_args['reset_target']
                  )
    
    if terminal_args.train:
        train(agent, env, config_args, terminal_args.name)
    
    if terminal_args.evaluate:
        test_env = AtariMonitor(env, config_args['video_dir']+'testing')
        evaluate(agent, test_env, config_args)
        # test_env.close()
    
    if terminal_args.play:
        play_env = AtariMonitor(env, config_args['video_dir']+'play')
        evaluate(agent, play_env)
        # play_env.close()
    
    # env.close()
    
if __name__ == "__main__":
    print("\nDevices available: ", tf.config.list_physical_devices('GPU'))
    print(tf.test.gpu_device_name())
    main()
