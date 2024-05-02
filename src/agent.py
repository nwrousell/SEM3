from networks import BBFModel
import numpy as np
import tensorflow as tf
import portpicker
import multiprocessing

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)
  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

class Agent:
    def __init__(
        self,
        strategy,
        stack_frames,
        encoder_network,
        n_actions,
        hidden_dim,
        learning_rate,
        weight_decay,
        start_gamma,
        end_gamma,
        target_action_selection,
        spr_loss_weight,
        start_update_horizon,
        end_update_horizon,
        target_ema_tau,
        shrink_factor,
        spr_prediction_depth=5,
        input_shape=(84,84),
        renormalize=False,
        ):
        
        self.spr_prediction_depth = spr_prediction_depth
        self.input_shape = input_shape
        self.target_action_selection = target_action_selection
        self.n_actions = n_actions
        self.spr_loss_weight = spr_loss_weight
        self.target_ema_tau = target_ema_tau
        self.shrink_factor = shrink_factor
        
        ##########################################
        NUM_WORKERS = 2
        NUM_PS = 2
        # cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
        
        # variable_partitioner = (
        #     tf.distribute.experimental.partitioners.MinSizePartitioner(
        #         min_shard_bytes=(256 << 10),
        #         max_shards=NUM_PS))

        # self.strategy = tf.distribute.ParameterServerStrategy(
        #         cluster_resolver,
        #         variable_partitioner=variable_partitioner)
        
        self.strategy = strategy
    
        with self.strategy.scope():
            self.num_grad_steps = 0

            self.online_model = BBFModel(input_shape=(*input_shape, stack_frames), 
                            encoder_network=encoder_network,
                            num_actions=n_actions, 
                            hidden_dim=hidden_dim, 
                            num_atoms=1,
                            renormalize=renormalize
                )
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
            self.target_model = BBFModel(input_shape=(*input_shape, stack_frames), 
                            encoder_network=encoder_network,
                            num_actions=n_actions, 
                            hidden_dim=hidden_dim, 
                            num_atoms=1,
                            renormalize=renormalize
                            )
            self.target_model.trainable = False
            fake_state = np.zeros((1, *input_shape, stack_frames))
            _ = self.online_model(fake_state, do_rollout=True, actions=np.random.randint(0,  n_actions, (1, spr_prediction_depth)))
            _ = self.target_model(fake_state, do_rollout=True, actions=np.random.randint(0, n_actions, (1, spr_prediction_depth)))
            self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
            self.compute_spr_error = self.compute_spr_error
            self.compute_td_error = self.compute_td_error
            self.get_target_q_values = self.get_target_q_values

        #print(self.online_model.summary())
        # print(bbf_online.encoder.summary())
        
        # initially, set target network to clone of online network
        online_weights = self.online_model.get_weights()
        self.target_model.set_weights(online_weights)
        
        # self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
        self._gamma_scheduler = exponential_decay_scheduler(10000, 0, start_gamma, end_gamma)
        self._update_horizon_scheduler = exponential_decay_scheduler(10000, 0, start_update_horizon, end_update_horizon)
    
    def gamma_scheduler(self):
        return self._gamma_scheduler(self.num_grad_steps)
    
    def update_horizon_scheduler(self):
        return self._update_horizon_scheduler(self.num_grad_steps)
    
    def choose_action(self, observation, epsilon):
        observation = observation[np.newaxis,:,:,:]
        
        prob = np.random.random()
        if prob < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            if self.target_action_selection:
                q_values, _, _ = self.target_model(observation)
            else:
                q_values, _, _ = self.online_model(observation)
            action = tf.argmax(q_values, axis=-1)
        
        return action
    
    def compute_td_error(self, q_values, actions, target_q_values):
        first_actions = actions[:, 0]
        chosen_q = tf.gather(q_values, indices=first_actions, axis=1, batch_dims=1)
        td_error = tf.vectorized_map(lambda _target: self.huber_loss(_target, chosen_q), tf.transpose(target_q_values))
        
        return td_error
    
    def compute_spr_error(self, spr_targets, spr_predictions, same_trajectory):
        spr_predictions = spr_predictions / tf.norm(spr_predictions, axis=-1, keepdims=True)
        spr_targets = spr_targets / tf.norm(spr_targets, axis=-1, keepdims=True)
        spr_loss = tf.reduce_sum(tf.pow(spr_predictions - spr_targets, 2), axis=-1)
        spr_loss = tf.reduce_sum(spr_loss * tf.cast(tf.transpose(same_trajectory, [1,0]), dtype=np.float32), axis=-1)
        
        return spr_loss

    def get_target_q_values(self, rewards, terminals, cumulative_gamma, target_network, next_states, update_horizon):
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
    
    def train_step(self, 
                   update_horizon,
                   observations, # (batch_size, subseq_len, *obs_shape, stack_size)
                   actions, # (batch_size, subseq_len)
                   rewards, # (batch_size, subseq_len)
                   returns, # (batch_size, subseq_len)
                   discounts,
                   next_states, # (batch_size, subseq_len, *obs_shape, stack_size)
                   next_actions, # (batch_size, subseq_len)
                   next_rewards, # (batch_size, subseq_len)
                   terminals, # (batch_size, subseq_len)
                   same_trajectory, # (batch_size, subseq_len)
                   indices):
        
        # swap batch and time dimensions
        next_states = tf.transpose(next_states, perm=[1,0,2,3,4])
                    
        first_state = observations[:,0,:,:,:] # (84, 84, 4)
        
        with tf.GradientTape() as tape:
            q_values, spr_predictions, _ = self.online_model(first_state, do_rollout=True, actions=next_actions[:,:self.spr_prediction_depth])
            
            # compute targets
            spr_targets = tf.vectorized_map(lambda x: self.target_model.encode_project(x, True, False), next_states[:self.spr_prediction_depth])
            q_targets = self.get_target_q_values(next_rewards, terminals, discounts, self.target_model, next_states, update_horizon)
            
            # compute TD error and SPR loss
            td_error = self.compute_td_error(q_values, actions, q_targets)
            spr_loss = self.compute_spr_error(spr_targets, spr_predictions, same_trajectory[:, :self.spr_prediction_depth])
            
            td_error = tf.reduce_mean(td_error)
            spr_loss = tf.reduce_mean(spr_loss)
                            
            loss = td_error + self.spr_loss_weight * spr_loss
        
        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_model.trainable_variables))
        self.num_grad_steps += 1
        
        # all_losses = {
        #     "Total Loss": loss.numpy(),
        #     "TD Error": td_error.numpy(),
        #     "SPR Loss": spr_loss.numpy()
        # }
        
        return loss.numpy(), td_error.numpy(), spr_loss.numpy()
    
    def update_target(self):
        target_weights = get_weight_dict(self.target_model)
        online_weights = get_weight_dict(self.online_model)
        new_target_weights = interpolate_weights(target_weights, online_weights, self.target_ema_tau)
        set_weights(self.target_model, new_target_weights)
    
    def reset_weights(self):
        new_weights = weights_reset(get_weight_dict(self.online_model), self.shrink_factor)
        set_weights(self.online_model, new_weights)
        # ! do something here for target model?
    
    def layers_summary(self, step):
        for layer in self.online_model.layers:
            if len(layer.weights) != 0:
                for weights in layer.weights:
                    tf.summary.histogram(weights.name, weights,step=step)
    
def get_weight_dict(model):
    weight_dict = {}
    for layer in model.layers:
        weight_dict[layer.name] = layer.get_weights()
    return weight_dict

def set_weights(model, weight_dict):
    for layer in model.layers:
        # layer.set_weights(weight_dict[layer.name])
        for param_to_set, new_param in zip(layer.trainable_variables, weight_dict[layer.name]):
	        param_to_set.assign(new_param)
        

def interpolate_weights(weights_a, weights_b, tau, layers=None):
    """
    Interpolates weights_a to weights_b by amount tau. 
    If layers isn't None, only weights in layers are interpolated
    """
    new_weights = {}
    for layer_name in weights_a.keys():
        if layers == None or layer_name in layers:
            w = []
            for w_a, w_b in zip(weights_a[layer_name], weights_b[layer_name]):
                w.append((1-tau) * w_a + tau * w_b)
            new_weights[layer_name] = w
        else:
            new_weights[layer_name] = weights_a[layer_name]
            
    return new_weights

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L311
def exponential_decay_scheduler(decay_period, warmup_steps, initial_value, final_value, reverse=False):
    """Instantiate a logarithmic schedule for a parameter.

    By default the extreme point to or from which values decay logarithmically
    is 0, while changes near 1 are fast. In cases where this may not
    be correct (e.g., lambda) pass reversed=True to get proper
    exponential scaling.

    Args:
        decay_period: float, the period over which the value is decayed.
        warmup_steps: int, the number of steps taken before decay starts.
        initial_value: float, the starting value for the parameter.
        final_value: float, the final value for the parameter.
        reverse: bool, whether to treat 1 as the asmpytote instead of 0.

    Returns:
        A decay function mapping step to parameter value.
    """
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value)
    end = np.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step):
        steps_left = max(decay_period + warmup_steps - step, 0)
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value)
        if reverse:
            new_value = 1 - new_value
            
        return new_value

    return scheduler

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/agents/spr_agent.py#L203
def weights_reset(weights_dict, shrink_factor=0.5, initializer=tf.initializers.GlorotUniform()):
    """
    reset weights of Q head; interpolate encoder and transition model weights 50% to random
    """
    # generate random weights
    rand_weights = {}
    for layer_name in weights_dict.keys():
        weights = []
        for w in weights_dict[layer_name]:
            weights.append(initializer(shape=w.shape))
        rand_weights[layer_name] = weights
    
    # reset weights of Q head and projection fully
    new_weights = interpolate_weights(weights_dict, rand_weights, 1, layers=["head", "projection"])
    
    # interpolate encoder/transition weights 50% (shrink-and-perturb)
    new_weights = interpolate_weights(new_weights, rand_weights, shrink_factor, layers=["encoder", "transition"])
    
    return new_weights