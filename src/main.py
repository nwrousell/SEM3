import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from networks import BBFModel, interpolate_weights, exponential_decay_scheduler, get_weight_dict, set_weights, weights_reset
from replay_buffer import ReplayBuffer
import tensorflow as tf
from image_pre import process_inputs
import numpy as np
import math
import yaml

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L856
def get_target_q_values(rewards, terminals, cumulative_gamma, target_network, next_states, update_horizon):
    """Builds the DQN target Q-values."""
    is_terminal_multiplier = 1.0 - terminals.astype(np.float32)
    
    # Incorporate terminal state to discount factor.
    gamma_with_terminal = cumulative_gamma * is_terminal_multiplier # (update_horizon, )

    q_values = tf.vectorized_map(lambda s: target_network(s)[0], next_states) # (update_horizon, batch_size, num_actions)

    replay_q_values = tf.reduce_max(q_values, axis=2) # (update_horizon, batch_size)
    replay_q_values = tf.transpose(replay_q_values) # (batch_size, update_horizon)
        
    # TODO - rewrite with vectorized operations to replace for loop
    target = np.zeros((next_states.shape[1], update_horizon))
    for k in range(1, update_horizon+1):
        prefix = tf.reduce_sum(rewards[:, :k] * gamma_with_terminal[:, :k], axis=1) # multiply discounts over rewards up to k and sum along time dimension
        target_pred = replay_q_values[:, k-1]
        target[:, k-1] = prefix + target_pred

    return tf.stop_gradient(target) # (batch_size, update_horizon)

huber_loss = tf.keras.losses.Huber()

# loss function: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L674
def compute_spr_loss(spr_targets, spr_predictions, same_trajectory):
    spr_predictions = spr_predictions / tf.norm(spr_predictions, axis=-1, keepdims=True)
    spr_targets = spr_targets / tf.norm(spr_targets, axis=-1, keepdims=True)
    spr_loss = tf.reduce_sum(tf.pow(spr_predictions - spr_targets, 2), axis=-1)
    spr_loss = tf.reduce_sum(spr_loss * tf.cast(tf.transpose(same_trajectory, [1,0]), dtype=np.float32), axis=-1)
    
    return spr_loss

def compute_RL_loss(target_q_values, q_values, actions):
    first_actions = actions[:, 0]
    chosen_q = tf.gather(q_values, indices=first_actions, axis=1, batch_dims=1)
    td_error = tf.vectorized_map(lambda _target: huber_loss(_target, chosen_q), tf.transpose(target_q_values))
    
    return td_error

def train_model(args):
    game = args['game']
    replay_ratio = args['replay_ratio']
    batch_size = args['batch_size']
    subseq_len = args['subseq_len']
    tau = args['tau']
    eps_greedy = args['eps_greedy']
    initial_collect_steps = args['initial_collect_steps']
    start_gamma = args['start_gamma']
    end_gamma = args['end_gamma']
    start_update_horizon = args['start_update_horizon']
    end_update_horizon = args['end_update_horizon']
    hidden_dim = args['hidden_dim']
    num_atoms = args['num_atoms']
    spr_loss_weight = args['spr_loss_weight']
    encoder_network = args['encoder_network']
    weight_decay = args['weight_decay']
    learning_rate = args['learning_rate']
    frameskip = args['frameskip']
    target_update_period = args['target_update_period']
    target_action_selection = args['target_action_selection']
    renormalize = args['renormalize']
    shrink_factor = args['shrink_factor']
    perturb_factor = args['perturb_factor']
    replay_capacity = args['replay_capacity']
    spr_prediction_depth = args['spr_prediction_depth']
    linear_scale = args['linear_scale']
    data_augmentation = args['data_augmentation']
    stack_frames = args['stack_frames']
    num_env_steps = args['num_env_steps']
    
    print("training model with args:", args)
    
    env = gym.make(game, render_mode="human", obs_type='grayscale', frameskip=frameskip)
    
    gamma_scheduler = exponential_decay_scheduler(10000, 0, start_gamma, end_gamma)
    update_horizon_scheduler = exponential_decay_scheduler(10000, 0, start_update_horizon, end_update_horizon)

    obs_shape = env.observation_space.shape
    print("observation shape: ", obs_shape)
    n_valid_actions = env.action_space.n
    print("# valid actions:", n_valid_actions)
    
    # obs_shape = (*obs_shape, 1)
    processed_obs_shape = (84, 84)
    
    data_spec = [
        tf.TensorSpec(shape=processed_obs_shape, name="observation", dtype=np.float32),
        tf.TensorSpec(shape=(), name="action", dtype=np.int32),
        tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
        tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
    ]
    
    bbf_online = BBFModel(input_shape=(*processed_obs_shape, stack_frames), 
                          encoder_network=encoder_network,
                          num_actions=n_valid_actions, 
                          hidden_dim=hidden_dim, 
                          num_atoms=num_atoms
                        )
    
    bbf_target = BBFModel(input_shape=(*processed_obs_shape, stack_frames), 
                          encoder_network=encoder_network,
                          num_actions=n_valid_actions, 
                          hidden_dim=hidden_dim, 
                          num_atoms=num_atoms
                        )
    bbf_target.trainable = False
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=replay_capacity, 
                                 batch_size=batch_size, 
                                 update_horizon=start_update_horizon, 
                                 gamma=start_gamma, 
                                 n_envs=1, 
                                 stack_size=stack_frames, # ? no idea what this is
                                 subseq_len=subseq_len,
                                 observation_shape=obs_shape,
                                 rng=np.random.default_rng(seed=17)
                                )
    
    # observation = process_inputs(observation, linear_scale=linear_scale, augmentation=False)
    fake_state = np.zeros((1, *processed_obs_shape, stack_frames))
    _ = bbf_online(fake_state, do_rollout=True, actions=np.random.randint(0, n_valid_actions, (1, spr_prediction_depth)))
    _ = bbf_target(fake_state, do_rollout=True, actions=np.random.randint(0, n_valid_actions, (1, spr_prediction_depth)))
    
    print(bbf_online.summary())
    # print(bbf_online.encoder.summary())
    
    # initially, set target network to clone of online network
    online_weights = bbf_online.get_weights()
    bbf_target.set_weights(online_weights)

    observation, _ = env.reset()

    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
    num_gradient_updates = 0
    current_state = np.zeros((*processed_obs_shape, stack_frames))
    for num_steps in range(num_env_steps):
        
        # select action based on policy
        if replay_buffer.num_elements() < initial_collect_steps:
            action = np.random.randint(0, n_valid_actions)
        else:
            prob = np.random.random()
            if prob < eps_greedy:
                action = np.random.randint(0, n_valid_actions)
            else:
                if target_action_selection:
                    q_values, _, _ = bbf_target(current_state[np.newaxis,:,:,:])
                else:
                    q_values, _, _ = bbf_online(current_state[np.newaxis,:,:,:])
                action = tf.argmax(q_values, axis=-1)
        
        # step environment, deposit experience in replay buffer
        observation, reward, terminated, truncated, info = env.step(action)
        terminated = np.array([terminated])
        reward = np.clip(reward, -1, 1)
        
        observation = process_inputs(observation,
                                               linear_scale=linear_scale,
                                               augmentation=False)
        
        current_state = np.concatenate([current_state[:,:,1:], observation[:,:,np.newaxis]], axis=-1)
        
        replay_buffer.add(observation, action, reward, terminated)
                
        # perform gradient updates by sampling from replay buffer (if possible):
        for _ in range(replay_ratio):
            num_gradient_updates += 1
            # print("elements in buffer:", replay_buffer.num_elements())
            if replay_buffer.num_elements() < initial_collect_steps:
                break
                
            update_horizon = round(update_horizon_scheduler(num_gradient_updates))
            gamma = gamma_scheduler(num_gradient_updates)
            
            # sample a batch from the replay buffer
            (observations, # (batch_size, subseq_len, *obs_shape, stack_size)
            actions, # (batch_size, subseq_len)
            rewards, # (batch_size, subseq_len)
            returns, # (batch_size, subseq_len)
            discounts,
            next_states, # (batch_size, subseq_len, *obs_shape, stack_size)
            next_actions, # (batch_size, subseq_len)
            next_rewards, # (batch_size, subseq_len)
            terminals, # (batch_size, subseq_len)
            same_trajectory, # (batch_size, subseq_len)
            indices # (batch_size, )
            ) = replay_buffer.sample_transition_batch(update_horizon=update_horizon,
                                                      gamma=gamma, 
                                                      subseq_len=update_horizon)
                                    
            # swap batch and time dimensions
            next_states = tf.transpose(next_states, perm=[1,0,2,3,4])
                        
            first_state = observations[:,0,:,:,:] # (84, 84, 4)
            
            # with tape active, FF to predict Q values and future state representations
            with tf.GradientTape() as tape:
                q_values, spr_predictions, _ = bbf_online(first_state, do_rollout=True, actions=next_actions[:,:spr_prediction_depth])
                
                # compute targets
                spr_targets = tf.vectorized_map(lambda x: bbf_target.encode_project(x, True, False), next_states[:spr_prediction_depth])
                # ! should we be passing in next_rewards instead of rewards
                q_targets = get_target_q_values(rewards, terminals, discounts, bbf_target, next_states, update_horizon)
                
                # compute TD error and SPR loss
                td_error = compute_RL_loss(q_targets, q_values, actions)
                spr_loss = compute_spr_loss(spr_targets, spr_predictions, same_trajectory[:, :spr_prediction_depth])
                
                td_error = tf.reduce_mean(td_error)
                spr_loss = tf.reduce_mean(spr_loss)
                                
                loss = td_error + spr_loss_weight * spr_loss
            
            gradients = tape.gradient(loss, bbf_online.trainable_variables)
            optimizer.apply_gradients(zip(gradients, bbf_online.trainable_variables))
            
            all_losses = {
                "Total Loss": loss.numpy(),
                "TD Error": td_error.numpy(),
                "SPR Loss": spr_loss.numpy()
            }
            
            print(all_losses)
            
            # update target networks with EMAs
            if num_steps % target_update_period == 0:
                target_weights = get_weight_dict(bbf_target)
                online_weights = get_weight_dict(bbf_online)
                new_target_weights = interpolate_weights(target_weights, online_weights, tau)
                set_weights(bbf_target, new_target_weights)
        
        # shrink-and-perturb every 40,000 gradient steps
        if num_gradient_updates % 40000 == 0:
            new_weights = weights_reset(get_weight_dict(bbf_online))
            set_weights(bbf_online, new_weights)
            #! should I do smth with the target network too?
                        
        # evaluate performance and log

if __name__ == "__main__":
    config_fname = "config.yaml"
    with open(config_fname, 'r') as file:
        args = yaml.safe_load(file)
        
    train_model(args)    

# TODO 
# next_rewards or rewards?
# check how I'm computing the losses - maybe use returns and vectorized operations
# logging / saving models
# eval
# update_horizon vs. subseq_len in replay buffer?
# setup on Oscar


## BBF
# look at 'jumps' - is that frameskip?
# Look at AtariPreprocessing class
# Look at DataEfficientAtariRunner - set up a similar agent
#   https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/eval_run_experiment.py#L173
# set up human normalized score thing so we can compare