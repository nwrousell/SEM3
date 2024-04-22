import gymnasium as gym
# from tf_agents.trajectories import trajectory
# import tensorflow as tf
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
from networks import BBFModel
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np


# env = gym.make("Amidar-v4", render_mode="human")
# observation, info = env.reset()

# for _ in range(1000):
#     # print(env.action_space)
    
#     action = env.action_space.sample()  # agent policy that uses the observation and info
    
#     observation, reward, terminated, truncated, info = env.step(action)

#     # print(observation.shape) # (210, 160, 3)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()



# def collect_step(environment, policy, replay_buffer):
#     time_step = environment.current_time_step()
#     action_step = policy.action(time_step)
#     next_time_step = environment.step(action_step.action)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)

#     # Add trajectory to the replay buffer
#     replay_buffer.add_batch(traj)

def train_model(env, model, replay_buffer, replay_ratio=2, num_frames=100000):
    replay_ratio = 2
        
    observation, info = env.reset()

    for i in num_frames:
        
        # gather experiences to deposit in replay buffer
        action = model.get_action(observation)
        
        observation, reward, terminated, truncated, info = env.step(action)
        # replay_buffer.add_batch([observation, action, reward, terminated])
        
        for _ in range(replay_ratio):
            pass
            # sample mini-batch from replay buffer
        
            # compute loss with it (RL loss, SPR loss, weight decay)
            
            # apply gradient
            
            # update target networks with EMAs
            
            # exponentially interpolate discount factor and update horizon
            
            # if i % 40000 == 0: shrink_and_perturb
            
            # evaluate performance and log

if __name__ == "__main__":
    # env = gym.make("Amidar-v4", render_mode="human")
    env = gym.make("Assault-v4", render_mode="human")

    obs_shape = env.observation_space.shape
    print("observation shape: ", obs_shape)
    n_valid_actions = env.action_space.n
    
    data_spec = [
        tf.TensorSpec(shape=obs_shape, name="observation", dtype=np.uint8),
        tf.TensorSpec(shape=(n_valid_actions), name="action", dtype=np.int32),
        tf.TensorSpec(shape=(), name="reward", dtype=np.float32),
        tf.TensorSpec(shape=(), name="terminal", dtype=np.uint8),
    ]
    
    replay_buffer = ReplayBuffer(data_spec, 
                                 replay_capacity=10000, 
                                 batch_size=32, 
                                 update_horizon=10, 
                                 gamma=0.97, 
                                 n_envs=1, 
                                 stack_size=4, # ? no idea what this is
                                 subseq_len=10, # ? ditto
                                 observation_shape=obs_shape,
                                 rng=np.random.default_rng(seed=17)
                                )
    
    observation, info = env.reset()
    
    for i in range(1000):
        action = np.random.randint(0, n_valid_actions)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        observation = np.array([observation])
        terminated = np.array([terminated])
        
        replay_buffer.add(observation, action, reward, terminated)
    
    # sample a batch from the replay buffer
    (observations, # (batch_size, subseq_len, *obs_shape, stack_size)
     actions, # (batch_size, subseq_len)
     rewards, # (batch_size, subseq_len)
     returns, # (batch_size, subseq_len)
     discounts, # ()
     next_states, # (batch_size, subseq_len, *obs_shape, stack_size)
     next_actions, # (batch_size, subseq_len)
     next_rewards, # (batch_size, subseq_len)
     terminal, # (batch_size, subseq_len)
     same_trajectory, # (batch_size, subseq_len)
     indices # (batch_size, )
     ) = replay_buffer.get_transition_elements()
    # [(32, 10, 210, 160, 3, 3), ()]
        
    # model = BBFModel(input_shape=(), num_actions=n_valid_actions)
    
    
    
    # train_model(env, model)
    
    
# network_def that can be given parameters (so can be used easily for both online / target networks)