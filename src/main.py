import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from networks import BBFModel, interpolate_weights
from replay_buffer import ReplayBuffer
import tensorflow as tf
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
    stack_size = args['stack_size']
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
    
    print("training model with args:", args)
    
    env = gym.make(game, render_mode="human")

    obs_shape = env.observation_space.shape
    print("observation shape: ", obs_shape)
    n_valid_actions = env.action_space.n
    print("# valid actions:", n_valid_actions)
    
    data_spec = [
        tf.TensorSpec(shape=obs_shape, name="observation", dtype=np.uint8),
        tf.TensorSpec(shape=(), name="action", dtype=np.int32),
        tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
        tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
    ]
    
    bbf_online = BBFModel(input_shape=obs_shape, 
                          encoder_network=encoder_network,
                          num_actions=n_valid_actions, 
                          hidden_dim=hidden_dim, 
                          num_atoms=num_atoms
                        )
    
    bbf_target = BBFModel(input_shape=obs_shape, 
                          encoder_network=encoder_network,
                          num_actions=n_valid_actions, 
                          hidden_dim=hidden_dim, 
                          num_atoms=num_atoms
                        )
    bbf_target.trainable = False
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=10000, 
                                 batch_size=batch_size, 
                                 update_horizon=start_update_horizon, 
                                 gamma=start_gamma, 
                                 n_envs=1, 
                                 stack_size=stack_size, # ? no idea what this is
                                 subseq_len=subseq_len,
                                 observation_shape=obs_shape,
                                 rng=np.random.default_rng(seed=17)
                                )
    
    observation, _ = env.reset()
    observation = tf.convert_to_tensor(tf.expand_dims(observation, axis=0))
    _ = bbf_online(observation, do_rollout=True, actions=np.random.randint(0, n_valid_actions, (1, subseq_len)))
    _ = bbf_target(observation, do_rollout=True, actions=np.random.randint(0, n_valid_actions, (1, subseq_len)))

    
    print(bbf_online.summary())
    
    # initially, set target network to clone of online network
    online_weights = bbf_online.get_weights()
    bbf_target.set_weights(online_weights)
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cumulative_gamma = np.array([math.pow(start_gamma, i) for i in range(1, start_update_horizon+1)], dtype=np.float32)
    
    for i in range(10000):
        
        # select action based on policy
        if replay_buffer.num_elements() < initial_collect_steps:
            action = np.random.randint(0, n_valid_actions)
        else:
            prob = np.random.random()
            if prob < eps_greedy:
                action = np.random.randint(0, n_valid_actions)
            else:
                q_values, _, _ = bbf_online(observation)
                action = tf.argmax(q_values)
        
        # step environment, deposit experience in replay buffer
        observation, reward, terminated, truncated, info = env.step(action)
        observation = np.array([observation])
        terminated = np.array([terminated])
        replay_buffer.add(observation, action, reward, terminated)
        
        # perform gradient updates by sampling from replay buffer (if possible):
        # ! idk if this is how it's done --> look at their training loop
        for _ in range(1):
            print("elements in buffer:", replay_buffer.num_elements())
            if replay_buffer.num_elements() < initial_collect_steps:
                break
            
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
            ) = replay_buffer.sample_transition_batch()
            
            # remove stack dimension
            observations = np.squeeze(observations)
            next_states = np.squeeze(next_states)
            # swap batch and time dimensions
            next_states = tf.transpose(next_states, perm=[1,0,2,3,4])
                        
            current_state = observations[:,0,:,:,:]
            
            # with tape active, FF to predict Q values and future state representations
            with tf.GradientTape() as tape:
                q_values, spr_predictions, _ = bbf_online(tf.convert_to_tensor(current_state), do_rollout=True, actions=tf.convert_to_tensor(next_actions))
                
                # compute targets
                spr_targets = tf.vectorized_map(lambda x: bbf_target.encode_project(x, True, False), tf.convert_to_tensor(next_states))
                # ? should we be passing in next_rewards instead of rewards
                target = get_target_q_values(tf.convert_to_tensor(next_rewards), terminals, tf.convert_to_tensor(cumulative_gamma), bbf_target, tf.convert_to_tensor(next_states), start_update_horizon)
                
                # compute TD error and SPR loss
                td_error = compute_RL_loss(target, q_values, actions)
                spr_loss = compute_spr_loss(spr_targets, spr_predictions, tf.convert_to_tensor(same_trajectory))
                
                loss = td_error + spr_loss_weight * spr_loss
                mean_loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(mean_loss, bbf_online.trainable_variables)
            optimizer.apply_gradients(zip(gradients, bbf_online.trainable_variables))
            
            all_losses = {
                "Total Loss": mean_loss.numpy(),
                "TD Error": tf.reduce_mean(td_error).numpy(),
                "SPR Loss": tf.reduce_mean(spr_loss).numpy()
            }
            
            print(all_losses)
            
            # update target networks with EMAs
            # ? this is kinda slow...
            if i % 10 == 0:
                new_target_weights = interpolate_weights(bbf_target.get_weights(), bbf_online.get_weights(), tau)
                bbf_target.set_weights(new_target_weights)
        
        # exponentially interpolate discount factor and update horizon
        
        # if i % 40000 == 0: shrink_and_perturb
        
        # evaluate performance and log

if __name__ == "__main__":
    config_fname = "config.yaml"
    with open(config_fname, 'r') as file:
        args = yaml.safe_load(file)
        
    train_model(args)    
    

# decay scheduler: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L311

# select action: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L377

# loss function: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L674