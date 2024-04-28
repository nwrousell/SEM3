import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from replay_buffer import ReplayBuffer
import tensorflow as tf
from image_pre import process_inputs
import numpy as np
import yaml
from agent import Agent
import argparse

data_spec = [
    tf.TensorSpec(shape=(84,84), name="observation", dtype=np.float32),
    tf.TensorSpec(shape=(), name="action", dtype=np.int32),
    tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
    tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
]

def train(agent: Agent, env, args):
    train_writer = tf.summary.create_file_writer(args['summaries_dir']+'train')
    test_writer = tf.summary.create_file_writer(args['summaries_dir']+'test')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=agent.optimizer, model=agent.online_model)
    manager = tf.train.CheckpointManager(checkpoint, args['model_dir'], max_to_keep=3)
    
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
    
    if args['process_inputs']:
        observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)

    current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
    epsilon = args['eps_greedy']
    
    for t in range(args['num_env_steps']):
        action = agent.choose_action(current_state, epsilon)
        observation, reward, terminated, truncated, info = env.step(action)
        terminated = np.array([terminated])
        
        if args['process_inputs']:
            observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)
        
        current_state = np.concatenate([current_state[:,:,1:], observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
        
        episode_rewards[-1] += reward
        
        if args['clip_reward']:
            reward = np.clip(reward, -1, 1)
            
        replay_buffer.add(observation, action, reward, terminated)
        
        if terminated:
            obs = env.reset()
            if args['process_inputs']:
                obs = process_inputs(obs, linear_scale=args['linear_scale'], augmentation=False)
            current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
            episode_rewards.append(0.0)
        
        if t > args['initial_collect_steps']:
            update_horizon = agent.update_horizon_scheduler(num_grad_steps)
            gamma = agent.gamma_scheduler(num_grad_steps)
            
            for s in range(args['replay_ratio']):
                num_grad_steps += 1
                batch = replay_buffer.sample_transition_batch(update_horizon=update_horizon,
                                                      gamma=gamma, 
                                                      subseq_len=update_horizon)
                
                agent.train_step(update_horizon, *batch)
                
                if s % args['target_update_frequency'] == 0:
                    agent.update_target()
                
                if s % args['reset_every'] == 0:
                    agent.reset_weights()
        
        num_episodes = len(episode_rewards)
        if num_episodes > 100:
            mean_reward = np.mean(episode_rewards[-101:-1])
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % args['eval_frequency']:
            if max_mean_reward is None or mean_reward > max_mean_reward:
                max_mean_reward = mean_reward
                print(f"improvement in mean_100ep_reward: {max_mean_reward}")
            else:
                print(f"No improvement in max mean_100ep_reward. Achieved: {mean_reward}, max: {max_mean_reward}")
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % args['eval_frequency']:
            eval_mean_reward = evaluate(agent, env)
            checkpoint.step.assign_add(1)
            save_path = manager.save()
            print(f"Evaluation reward at {t} step is {eval_mean_reward}")
            with test_writer.as_default():
                tf.summary.scalar('eval_reward', eval_mean_reward, step=t)
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % args['train_log_frequency']:
            with train_writer.as_default():
                tf.summary.scalar("mean_100ep_reward", mean_reward, step=t)
            
            with train_writer.as_default():
                agent.layers_summary(t)
   
def evaluate(agent: Agent, env, args, restore=False, play=False):
    if restore:
        checkpoint = tf.train.checkpoint(model=agent.online_model())
        latest_snapshot= tf.train.latest_checkpoint(args['model_dir'])
        if not latest_snapshot:
            raise Exception(f"No model snapshot found in {args['model_dir']}")
        
        checkpoint.restore(latest_snapshot)
        print("Restored model from latest snapshot")
    
    observation, _ = env.reset()
    
    eval_episode_rewards = [0.0]
    
    if args['process_inputs']:
        observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)

    current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
    epsilon = args['evaluation_epsilon']
    
    while True:
        action = agent.choose_action(current_state, epsilon)
        observation, reward, terminated, truncated, info = env.step(action)
        if args['process_inputs']:
            observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)
        current_state = np.concatenate([current_state[:,:,1:], observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
        
        eval_episode_rewards[-1] += reward
        
        eval_mean_reward = np.mean(eval_episode_rewards)
        
        if terminated:
            observation, _ = env.reset()
            if args['process_inputs']:
                observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)
            current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], axis=-1)
            
            num_episodes = len(eval_episode_rewards)
            if restore:
                print(f"Mean reward after {num_episodes} episodes is {round(eval_mean_reward, 2)}")
            if play:
                break
            if num_episodes >= args['evaluation_episodes']:
                break
            eval_episode_rewards.append(0.0)
    
    return round(eval_mean_reward, 2)
        

def main():
    config_fname = "config.yaml"
    with open(config_fname, 'r') as file:
        config_args = yaml.safe_load(file)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train an agent to find optimal policy', action='store_true')
    parser.add_argument('--evaluate', help='evaluate trained policy of an agent', action='store_true')
    parser.add_argument('--play', help='let trained agent play', action='store_true')
    # parser.add_argument('--env', nargs=1, help='env used for DQN', type=str)
    
    terminal_args = parser.parse_args()
    
    render_mode = 'human' if terminal_args.play else 'rgb_array'
    
    env = gym.make(config_args['game'], 
                   render_mode="human", 
                   obs_type='grayscale', 
                   frameskip=config_args['frameskip'])
    
    n_actions = env.action_space.n
    
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
                  config_args['renormalize'] # ! not implemented
                  )
    
    if terminal_args.train:
        train(agent, env, config_args)
    
    if terminal_args.evaluate:
        test_env = gym.wrappers.Monitor(env, config_args['video_dir']+'testing', force=True)
        evaluate(agent, test_env, config_args)
        test_env.close()
    
    if terminal_args.play:
        play_env = gym.wrappers.Monitor(env, config_args['video_dir']+'play', force=True)
        evaluate(agent, play_env)
        play_env.close()
    
    env.close()
    
if __name__ == "__main__":
    print("GPU Available: ", tf.test.is_gpu_available())
    main()

        
    