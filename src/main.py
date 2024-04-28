import gymnasium as gym
from replay_buffer import ReplayBuffer
import tensorflow as tf
from image_pre import process_inputs
import numpy as np
import yaml
from agent import Agent
from time import time
import argparse

data_spec = [
    tf.TensorSpec(shape=(84,84), name="observation", dtype=np.float32),
    tf.TensorSpec(shape=(), name="action", dtype=np.int32),
    tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
    tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
]

# stolen from Dopamine: https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py#L41C1-L62C25
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.

    Args:
        decay_period: float, the period over which epsilon is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before epsilon is decayed.
        epsilon: float, the final value to which to decay the epsilon parameter.

    Returns:
        A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus
    
    
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
    
    start_time = time()
    
    if args['process_inputs']:
        observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)

    current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
    
    for t in range(args['num_env_steps']):
        epsilon = linearly_decaying_epsilon(args['epsilon_decay_period'], t, args['initial_collect_steps'], args['epsilon_train'])
        
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
            observation, _ = env.reset()
            if args['process_inputs']:
                observation = process_inputs(observation, linear_scale=args['linear_scale'], augmentation=False)
            current_state = np.concatenate([np.zeros((84,84,args['stack_frames']-1)), observation[:,:,np.newaxis]], dtype=np.float32, axis=-1)
            episode_rewards.append(0.0)
        
        if t > args['initial_collect_steps']:
            update_horizon = round(agent.update_horizon_scheduler(num_grad_steps))
            gamma = agent.gamma_scheduler(num_grad_steps)
            
            for s in range(args['replay_ratio']):
                num_grad_steps += 1
                batch = replay_buffer.sample_transition_batch(update_horizon=update_horizon,
                                                      gamma=gamma, 
                                                      subseq_len=update_horizon)
                    
                loss, td_error, spr_error = agent.train_step(update_horizon, *batch)
                
                if num_grad_steps % args['target_update_frequency'] == 0:
                    agent.update_target()
                
                if num_grad_steps % args['reset_every'] == 0:
                    agent.reset_weights()
        
        num_episodes = len(episode_rewards)
        if num_episodes > 100:
            mean_reward = np.mean(episode_rewards[-101:-1])
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % args['print_frequency'] == 0:
            if max_mean_reward is None or mean_reward > max_mean_reward:
                max_mean_reward = mean_reward
                print(f"improvement in mean_100ep_reward: {max_mean_reward}")
            else:
                print(f"No improvement in max mean_100ep_reward. Achieved: {mean_reward}, max: {max_mean_reward}")
            elapsed = time() - start_time
            print(f"Environment steps: {t}. Gradient updates: {num_grad_steps}. TD error: {td_error}. SPR error: {spr_error}. Time elapsed: {elapsed}s")
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % args['eval_frequency'] == 0:
            eval_mean_reward = evaluate(agent, env)
            checkpoint.step.assign_add(1)
            save_path = manager.save()
            print(f"Evaluation reward at {t} step is {eval_mean_reward}")
            with test_writer.as_default():
                tf.summary.scalar('eval_reward', eval_mean_reward, step=t)
        
        if num_episodes > 100 and t > args['initial_collect_steps'] and t % (args['train_log_frequency']*100) == 0:
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
        
    for k, v in config_args.items():
        print(f"{k}: {v}")   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train an agent to find optimal policy', action='store_true')
    parser.add_argument('--evaluate', help='evaluate trained policy of an agent', action='store_true')
    parser.add_argument('--play', help='let trained agent play', action='store_true')
    # parser.add_argument('--env', nargs=1, help='Atari 2600 game used as environment', type=str)
    
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
    print("Devices available: ", tf.config.list_physical_devices('GPU'))
    main()


# get running on Oscar

# data augmentation
# what to do to target network during reset?
# renormalization

# audio support for ALE
# different architectures with audio + run experiments

# add distributional DQN
# add dueling DQN
# add double DQN
# maybe add prioritized replay buffer? (can probably mostly copy from BBF)