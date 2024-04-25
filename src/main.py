import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from networks import BBFModel
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
import math

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

def train_model():
    # env = gym.make("Amidar-v4", render_mode="human")
    env = gym.make("Assault-v4", render_mode="human")

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
    
    # Simplifications:
    # stack_size = 1 / no pre-processing (grayscale, down-scaling, etc.)
    # No target network (so no EMAs)
    # No regularization (weight decay, shrink-and-perturb)
    # No dueling or distributional shenanigans
    # no frame-skip
    
    num_env_steps = 1000
    replay_ratio = 2
    batch_size = 32
    stack_size = 1
    subseq_len = 3
    gamma = 0.97
    update_horizon = subseq_len
    hidden_dim = 2048
    num_atoms = 1 # for distributional, ignore for now
    spr_loss_weight = 2
    bbf = BBFModel(input_shape=obs_shape, 
                   num_actions=n_valid_actions, 
                   hidden_dim=hidden_dim, 
                   num_atoms=num_atoms
                   )
    
    # _input = tf.keras.layers.Input(obs_shape)
    # actions = tf.keras.layers.Input(shape=(n_valid_actions,))    
    # q_values, preds, rep = bbf(_input, do_rollout=True, actions=actions)
    # bbf = tf.keras.Model(
    #     inputs=[_input, actions],
    #     outputs=[q_values, preds, rep]
    # )
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=10000, 
                                 batch_size=batch_size, 
                                 update_horizon=update_horizon, 
                                 gamma=gamma, 
                                 n_envs=1, 
                                 stack_size=stack_size, # ? no idea what this is
                                 subseq_len=subseq_len,
                                 observation_shape=obs_shape,
                                 rng=np.random.default_rng(seed=17)
                                )
    
    observation, info = env.reset()
    
    cumulative_gamma = np.array([math.pow(gamma, i) for i in range(1, update_horizon+1)], dtype=np.float32)
    
    huber_loss = tf.keras.losses.Huber()
    
    for i in range(num_env_steps):
        
        # selection action based on policy
        action = np.random.randint(0, n_valid_actions)
        
        # step environment, deposit experience in replay buffer
        observation, reward, terminated, truncated, info = env.step(action)
        observation = np.array([observation])
        terminated = np.array([terminated])
        replay_buffer.add(observation, action, reward, terminated)
        
        # perform gradient updates by sampling from replay buffer (if possible):
        # ! idk if this is how it's done --> look at their training loop
        for _ in range(replay_ratio):
            print("elements in buffer:", replay_buffer.num_elements())
            # skip gradient update until there are enough elements in the buffer
            if replay_buffer.num_elements() < batch_size:
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
                q_values, spr_predictions, _ = bbf(current_state, do_rollout=True, actions=next_actions)
                
                spr_targets = tf.vectorized_map(lambda x: bbf.encode_project(x, True, False), next_states)
                
                # compute TD error
                target = get_target_q_values(rewards, terminals, cumulative_gamma, bbf, next_states, update_horizon)
                first_actions = actions[:, 0]
                chosen_q = tf.gather(q_values, indices=first_actions, axis=1, batch_dims=1)
                
                # ? do we compare the chosen_q to each of the three targets?
                td_error = tf.vectorized_map(lambda _target: huber_loss(_target, chosen_q), tf.transpose(target))
                
                # compute SPR Loss
                spr_predictions = spr_predictions / tf.norm(spr_predictions, axis=-1, keepdims=True)
                spr_targets = spr_targets / tf.norm(spr_targets, axis=-1, keepdims=True)
                spr_loss = tf.reduce_sum(tf.pow(spr_predictions - spr_targets, 2), axis=-1)
                spr_loss = tf.reduce_sum(spr_loss * tf.cast(tf.transpose(same_trajectory, [1,0]), dtype=np.float32), axis=-1)
                
                loss = td_error + spr_loss_weight * spr_loss
                mean_loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(mean_loss, bbf.trainable_variables)
            print(gradients)
            
            all_losses = {
                "Total Loss": mean_loss.numpy(),
                "TD Error": tf.reduce_mean(td_error).numpy(),
                "SPR Loss": tf.reduce_mean(spr_loss).numpy()
            }
            print(all_losses)
            
                
            
            
            # compute and apply gradients
                
    # update target networks with EMAs
    
    # exponentially interpolate discount factor and update horizon
    
    # if i % 40000 == 0: shrink_and_perturb
    
    # evaluate performance and log

if __name__ == "__main__":
    train_model()    
    
# network_def that can be given parameters (so can be used easily for both online / target networks)

# how do they use the stack / subseq len
# where do they compute the losses

# decay scheduler: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L311

# select action: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L377

# loss function: https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L674

# sanity check number of params and encoder (ImpalaCNN) output dimensions - the encoder output is a much higher dimension than the original CNN model

# TODO
# What is the resulting dimensions after the ImpalaCNN
# how does the transition work?

# BBF regularization stuff
# does BBF combine / preprocess observations like Mnih 2015?

# how to setup the replay buffer and actually perform training
# optimize q-learning for all actions at once?

# how does CASL work fully with the options/hierarichal RL


# TODO: code losses
# TODO: replay buffer
# TODO: make BBF agent
# TODO: set up bare-bones training loop, test

# TODO: add EMAs for target encoder, target projection, target Q network
# TODO: find out and code preprocess function for observations
# TODO: add weight decay
# TODO: add exponential schedule for gamma/n
# TODO: add shrink-and-perturb
# TODO: code augmentation fn for future observations