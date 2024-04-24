import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from networks import BBFModel
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np


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
    # random latent and hidden dim
    # stack_size = 1 / no pre-processing (grayscale, down-scaling, etc.)
    # n=1 for computing TD error
    # No target network (so no EMAs)
    # No regularization (weight decay, shrink-and-perturb)
    # No dueling or distributional shenanigans
    
    num_env_steps = 1000
    replay_ratio = 2
    batch_size = 32
    stack_size = 1
    subseq_len = 3
    update_horizon= 1 
    hidden_dim = 2048
    num_atoms = 1
    bbf = BBFModel(input_shape=obs_shape, 
                   num_actions=n_valid_actions, 
                   hidden_dim=hidden_dim, 
                   num_atoms=num_atoms
                   )
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=10000, 
                                 batch_size=batch_size, 
                                 update_horizon=update_horizon, 
                                 gamma=0.97, 
                                 n_envs=1, 
                                 stack_size=stack_size, # ? no idea what this is
                                 subseq_len=subseq_len, # ? ditto
                                 observation_shape=obs_shape,
                                 rng=np.random.default_rng(seed=17)
                                )
    
    observation, info = env.reset()
    
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
            discounts, # ! don't think this is being computed correctly
            next_states, # (batch_size, subseq_len, *obs_shape, stack_size)
            next_actions, # (batch_size, subseq_len)
            next_rewards, # (batch_size, subseq_len)
            terminal, # (batch_size, subseq_len)
            same_trajectory, # (batch_size, subseq_len)
            indices # (batch_size, )
            ) = replay_buffer.sample_transition_batch(update_horizon=1)
            
            # with tape active, FF to predict Q values and future state representations
            
            observations = np.squeeze(observations)
                        
            print("observations shape, next_states shape:",observations.shape, next_states.shape)
            
            print("actions shape, next_actions shape:", actions.shape, next_actions.shape)
            print("discounts shape:", discounts.shape, discounts[0])
            
            initial_observation = observations[:,0,:,:,:]
            q_values, pred_spatial_latents, representation = bbf(initial_observation, do_rollout=True, actions=next_actions)
            print("q values shape:", q_values.shape)
            print("pred latent shape:", pred_spatial_latents.shape)
            print("representation shape:", representation.shape)
            
            # target_spatial_latents = encode_project(new_states)
            
            # print("actions shape", actions.shape)
            # state_predictions = bbf.spr_rollout(latent, actions)
            # print("state_predictions shape:", state_predictions.shape)
            
            
            
            # then compute TD error (RL Loss) and SPR Loss
            # q_values, spr_predictions, _ = get_q_values(
            #     q_online, current_state, actions[:, :-1], use_spr, batch_rngs
            # )
            # q_values = jnp.squeeze(q_values)
            # replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions[:, 0])
            # dqn_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)
            # td_error = dqn_loss
            
            # spr_predictions = spr_predictions.transpose(1, 0, 2)
            # spr_predictions = spr_predictions / jnp.linalg.norm(
            #     spr_predictions, 2, -1, keepdims=True)
            # spr_targets = spr_targets / jnp.linalg.norm(
            #     spr_targets, 2, -1, keepdims=True)
            # spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
            # spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)
            
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