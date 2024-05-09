from networks import BBFModel
import numpy as np
import tensorflow as tf
from image_pre import drq_image_aug


class Agent:
    def __init__(
        self,
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
        double_DQN=True,
        distributional_DQN=True,
        dueling_DQN=True,
        vmax=10,
        num_atoms=51,
        seed=17,
        augment_spr=True,
        reset_target=True,
        audio=False,
        scale=True,
        batch_size=32,
        ):
        
        self.spr_prediction_depth = spr_prediction_depth
        self.input_shape = input_shape
        self.target_action_selection = target_action_selection
        self.n_actions = n_actions
        self.spr_loss_weight = spr_loss_weight
        self.target_ema_tau = target_ema_tau
        self.shrink_factor = shrink_factor
        self.double_dqn = double_DQN
        self.distributional_dqn = distributional_DQN
        self.dueling_dqn = dueling_DQN
        self.vmax = vmax
        self.vmin = -vmax
        self.num_atoms = num_atoms
        self.support = tf.cast(tf.linspace(self.vmin, vmax, num_atoms), dtype=np.float32)
        self.seed = int(seed)
        self.initializer = tf.initializers.GlorotUniform(seed=self.seed)
        self.augment_spr = augment_spr
        self.reset_target = reset_target
        self.audio = audio
        self.downscale = scale
        self.batch_size = batch_size

        self.online_model = BBFModel(input_shape=(*input_shape, stack_frames), 
                          encoder_network=encoder_network,
                          num_actions=n_actions, 
                          hidden_dim=hidden_dim, 
                          num_atoms=self.num_atoms,
                          renormalize=renormalize,
                          dueling=self.dueling_dqn,
                          distributional=self.distributional_dqn,
                          initializer=self.initializer,
                          audio=self.audio,
                          scale=self.downscale,
                        )
    
        self.target_model = BBFModel(input_shape=(*input_shape, stack_frames), 
                            encoder_network=encoder_network,
                            num_actions=n_actions, 
                            hidden_dim=hidden_dim, 
                            num_atoms=self.num_atoms,
                            renormalize=renormalize,
                            dueling=self.dueling_dqn,
                            distributional=self.distributional_dqn,
                            initializer=self.initializer,
                            audio=self.audio,
                            scale=self.downscale,
                            )
        # self.target_model.trainable = False
        
        fake_video = np.zeros((1, *input_shape, stack_frames))
        fake_audio = np.zeros((1, 512, stack_frames))
        fake_state = (fake_video, fake_audio)
        _ = self.online_model(fake_state, self.support, do_rollout=True, actions=np.random.randint(0,  n_actions, (1, spr_prediction_depth)))
        _ = self.target_model(fake_state, self.support, do_rollout=True, actions=np.random.randint(0, n_actions, (1, spr_prediction_depth)))
        
        # Initialize the EMA
        self.ema = tf.train.ExponentialMovingAverage(decay=(1-self.target_ema_tau))

        # Apply EMA to all trainable variables
        self.ema_op = self.ema.apply(self.online_model.trainable_variables)
        
        # initially, set target network to clone of online network
        online_weights = self.online_model.get_weights()
        self.target_model.set_weights(online_weights)
        
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=0.00015)
        # self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        
        self.gamma_scheduler = exponential_decay_scheduler(10000, 0, start_gamma, end_gamma)
        self.update_horizon_scheduler = exponential_decay_scheduler(10000, 0, start_update_horizon, end_update_horizon)
    
    def choose_action(self, video, audio, epsilon):
        prob = np.random.random()
        if prob < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            video = video[np.newaxis,:,:,:]
            audio = audio[np.newaxis,:,:]
            observation = (video, audio)
            if self.target_action_selection:
                q_values = self.target_model(observation, self.support)[0]
            else:
                q_values = self.online_model(observation, self.support)[0]
            action = tf.argmax(q_values, axis=-1)
        
        return action
    
    # https://github.com/google-research/google-research/blob/master/bigger_better_faster/bbf/agents/spr_agent.py#L661
    def compute_td_error(self, q_values, actions, target, logits=None):
        first_actions = actions[:, 0]
        if self.distributional_dqn:
            logits = tf.squeeze(logits) # (batch_size, num_actions, num_atoms)
            chosen_action_logits = tf.gather(logits, indices=first_actions, axis=1, batch_dims=1) # (batch_size, num_atoms)
            dqn_loss = tf.nn.softmax_cross_entropy_with_logits(target, chosen_action_logits)
            td_error = dqn_loss + tf.reduce_sum(np.nan_to_num(target * tf.math.log(target)), axis=-1)
        else:
            chosen_q = tf.gather(q_values, indices=first_actions, axis=1, batch_dims=1)
            td_error = huber_loss(target, chosen_q)
        
        return td_error
    
    def compute_spr_error(self, spr_targets, spr_predictions, same_trajectory):
        spr_predictions = spr_predictions / tf.norm(spr_predictions, axis=-1, keepdims=True)
        spr_targets = spr_targets / tf.norm(spr_targets, axis=-1, keepdims=True)
        spr_loss = tf.reduce_sum(tf.pow(spr_predictions - spr_targets, 2), axis=-1)
        spr_loss = tf.reduce_sum(spr_loss * tf.cast(tf.transpose(same_trajectory, [1,0]), dtype=np.float32), axis=-2)
        
        return spr_loss

    def get_target_q_values(self, rewards, terminals, cumulative_gamma, next_video, next_audio, update_horizon, probabilities=None):
        # compute target q values or target distribution if using distributional DQN
        
        is_terminal_multiplier = 1.0 - terminals.astype(np.float32)
        
        # Incorporate terminal state into discount factor.
        gamma_with_terminal = cumulative_gamma * is_terminal_multiplier # (batch_size, update_horizon)

        forecast_from = (next_video[update_horizon-1], next_audio[update_horizon-1])
        future_qs_target, _, probabilities, _, _ = self.target_model(forecast_from, self.support)
        
        # if double, select action using online network
        if self.double_dqn:
            future_qs_online = self.online_model(forecast_from, self.support)[0] # (batch_size, num_actions)
            action_selected = tf.argmax(future_qs_online, axis=-1)
        else:
            action_selected = tf.argmax(future_qs_target, axis=-1)
        
        discounted_reward_prefix = tf.reduce_sum(rewards * gamma_with_terminal, axis=1)
        gamma_n = cumulative_gamma[1] * cumulative_gamma[-1]
        
        if self.distributional_dqn:
            # Compute the target Q-value distribution
            probabilities = tf.squeeze(probabilities)
            next_probabilities = tf.gather(probabilities, indices=action_selected, axis=1, batch_dims=1) # (batch_size, num_atoms)
            target_support = discounted_reward_prefix[:, np.newaxis] + gamma_n * self.support
            target = project_distribution(target_support, next_probabilities, self.support)
        else:
            # target_q_values = r_1 + γr_2 + ... + γ^{n-1}r_{n-1} + γ^nQ*(s_{t+n})
            future_q = tf.gather(future_qs_target, indices=action_selected, axis=1, batch_dims=1)
            target = discounted_reward_prefix + gamma_n * future_q # ! here

        return tf.stop_gradient(target) # (batch_size, update_horizon)
    
    def train_step(self, 
                   update_horizon,
                   video, # (batch_size, subseq_len, *obs_shape, stack_size)
                   audio, # (batch_size, subseq_len, 512, stack_size)
                   actions, # (batch_size, subseq_len)
                   rewards, # (batch_size, subseq_len)
                   returns, # (batch_size, subseq_len)
                   discounts, # (update_horizon)
                   next_video, # (batch_size, subseq_len, *obs_shape, stack_size)
                   next_audio,
                   next_actions, # (batch_size, subseq_len)
                   next_rewards, # (batch_size, subseq_len)
                   terminals, # (batch_size, subseq_len)
                   same_trajectory, # (batch_size, subseq_len)
                   indices,
                   sampling_probabilities=None):
        
        # swap batch and time dimensions
        next_video = tf.transpose(next_video, perm=[1,0,2,3,4])
        next_audio = tf.transpose(next_audio, perm=[1,0,2,3])
                    
        first_video = video[:,0,:,:,:] # (batch_size, 84, 84, 4)
        first_audio = audio[:,0,:,:] # (batch_size, 512, 4)

        first_state = (first_video, first_audio)

        hidden_dim = 2048
        
        with tf.GradientTape() as tape:
            q_values, logits, probabilities, spr_predictions, _ = self.online_model(first_state, self.support, do_rollout=True, actions=next_actions[:,:self.spr_prediction_depth])
            
            # compute targets
            next_video_for_spr = next_video[:self.spr_prediction_depth]
            next_audio_for_spr = next_audio[:self.spr_prediction_depth]
            if self.augment_spr:
                next_video_for_spr = drq_image_aug(next_video_for_spr)
            
            next_states_for_spr =  (next_video_for_spr, next_audio_for_spr)

            # spr_targets = tf.vectorized_map(lambda x: self.target_model.encode_project(x, True, False), next_states_for_spr)
            spr_targets = tf.map_fn(lambda x: self.target_model.encode_project(x, True, False), elems=next_states_for_spr, fn_output_signature=tf.TensorSpec(shape=(self.batch_size, hidden_dim)))
            q_targets = self.get_target_q_values(next_rewards[:,:update_horizon], terminals[:,:update_horizon], discounts, next_video[:update_horizon], next_audio[:update_horizon], update_horizon)
            
            # compute TD error and SPR loss
            td_error = self.compute_td_error(q_values, next_actions[:,:update_horizon], q_targets, logits=logits)
            spr_loss = self.compute_spr_error(spr_targets, spr_predictions, same_trajectory[:, :self.spr_prediction_depth])
            
            if self.prioritized:
                # The original prioritized experience replay uses a linear exponent
                # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
                # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
                # a fixed exponent actually performs better, except on Pong.
                loss_weights = 1.0 / tf.sqrt(sampling_probabilities + 1e-10)
                loss_weights /= tf.reduce_max(loss_weights)
                loss_weights = tf.cast(loss_weights, np.float32)

                # Rainbow and prioritized replay are parametrized by an exponent alpha,
                # but in both cases it is set to 0.5 - for simplicity's sake we leave it
                # as is here, using the more direct tf.sqrt(). Taking the square root
                # "makes sense", as we are dealing with a squared loss.
                # Add a small nonzero value to the loss to avoid 0 priority items. While
                # technically this may be okay, setting all items to 0 priority will cause
                # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
                indices = tf.reshape(indices, (-1))
                td_error = tf.reshape(td_error, (-1))
                priorities = tf.sqrt(td_error + 1e-10)
                # print("priorities:", priorities)
                self.replay.set_priority(indices, priorities)

                # Weight the loss by the inverse priorities.
                td_error = loss_weights * td_error
            
            loss = tf.reduce_mean(td_error + self.spr_loss_weight * spr_loss)
        
        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_model.trainable_variables))
        
        td_error = tf.reduce_mean(td_error)
        spr_loss = tf.reduce_mean(spr_loss)
        return loss.numpy(), td_error.numpy(), spr_loss.numpy()
    
    def update_target(self):
        self.ema.apply(self.online_model.trainable_variables)
        for target_var, online_var in zip(self.target_model.trainable_variables, self.online_model.trainable_variables):
            target_var.assign(self.ema.average(online_var))
        # target_weights = get_weight_dict(self.target_model)
        # online_weights = get_weight_dict(self.online_model)
        # new_target_weights = interpolate_weights(target_weights, online_weights, self.target_ema_tau)
        # set_weights(self.target_model, new_target_weights)
    
    def reset_weights(self):
        print("RESETTING")
        new_online_weights = weights_reset(get_weight_dict(self.online_model), self.shrink_factor)
        set_weights(self.online_model, new_online_weights)
        
        if self.reset_target:
            new_target_weights = weights_reset(get_weight_dict(self.target_model), self.shrink_factor)
            set_weights(self.target_model, new_target_weights)

def huber_loss(y_true, y_pred, delta=1.0):
    residual = tf.abs(y_true - y_pred)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

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


# from Dopamine: https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py#L344
def project_distribution(
    supports, weights, target_support, validate_args=False
):
    """Projects a batch of (support, weights) onto target_support.

    Based on equation (7) in (Bellemare et al., 2017):
        https://arxiv.org/abs/1707.06887
    In the rest of the comments we will refer to this equation simply as Eq7.

    This code is not easy to digest, so we will use a running example to clarify
    what is going on, with the following sample inputs:

        * supports =       [[0, 2, 4, 6, 8],
                            [1, 3, 4, 5, 6]]
        * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.5, 0.1, 0.1]]
        * target_support = [4, 5, 6, 7, 8]

    In the code below, comments preceded with 'Ex:' will be referencing the above
    values.

    Args:
        supports: Tensor of shape (batch_size, num_dims) defining supports for the
        distribution.
        weights: Tensor of shape (batch_size, num_dims) defining weights on the
        original support points. Although for the CategoricalDQN agent these
        weights are probabilities, it is not required that they are.
        target_support: Tensor of shape (num_dims) defining support of the projected
        distribution. The values must be monotonically increasing. Vmin and Vmax
        will be inferred from the first and last elements of this tensor,
        respectively. The values in this tensor must be equally spaced.
        validate_args: Whether we will verify the contents of the target_support
        parameter.

    Returns:
        A Tensor of shape (batch_size, num_dims) with the projection of a batch of
        (support, weights) onto target_support.

    Raises:
        ValueError: If target_support has no dimensions, or if shapes of supports,
        weights, and target_support are incompatible.
    """
    target_support_deltas = target_support[1:] - target_support[:-1]
    # delta_z = `\Delta z` in Eq7.
    delta_z = target_support_deltas[0]
    validate_deps = []
    supports.shape.assert_is_compatible_with(weights.shape)
    supports[0].shape.assert_is_compatible_with(target_support.shape)
    target_support.shape.assert_has_rank(1)
    if validate_args:
        # Assert that supports and weights have the same shapes.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
                [supports, weights],
            )
        )
        # Assert that elements of supports and target_support have the same shape.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(
                    tf.equal(tf.shape(supports)[1], tf.shape(target_support))
                ),
                [supports, target_support],
            )
        )
        # Assert that target_support has a single dimension.
        validate_deps.append(
            tf.Assert(
                tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]
            )
        )
        # Assert that the target_support is monotonically increasing.
        validate_deps.append(
            tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support])
        )
        # Assert that the values in target_support are equally spaced.
        validate_deps.append(
            tf.Assert(
                tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
                [target_support],
            )
        )

    with tf.control_dependencies(validate_deps):
        # Ex: `v_min, v_max = 4, 8`.
        v_min, v_max = target_support[0], target_support[-1]
        # Ex: `batch_size = 2`.
        batch_size = tf.shape(supports)[0]
        # `N` in Eq7.
        # Ex: `num_dims = 5`.
        num_dims = tf.shape(target_support)[0]
        # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
        # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
        #                         [[ 4.  4.  4.  5.  6.]]]`.
        clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
        # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]
        #                         [ 4.  4.  4.  6.  8.]]
        #                        [[ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]
        #                         [ 4.  4.  4.  5.  6.]]]]`.
        tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
        # Ex: `reshaped_target_support = [[[ 4.]
        #                                  [ 5.]
        #                                  [ 6.]
        #                                  [ 7.]
        #                                  [ 8.]]
        #                                 [[ 4.]
        #                                  [ 5.]
        #                                  [ 6.]
        #                                  [ 7.]
        #                                  [ 8.]]]`.
        reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
        reshaped_target_support = tf.reshape(
            reshaped_target_support, [batch_size, num_dims, 1]
        )
        # numerator = `|clipped_support - z_i|` in Eq7.
        # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
        #                     [ 1.  1.  1.  1.  3.]
        #                     [ 2.  2.  2.  0.  2.]
        #                     [ 3.  3.  3.  1.  1.]
        #                     [ 4.  4.  4.  2.  0.]]
        #                    [[ 0.  0.  0.  1.  2.]
        #                     [ 1.  1.  1.  0.  1.]
        #                     [ 2.  2.  2.  1.  0.]
        #                     [ 3.  3.  3.  2.  1.]
        #                     [ 4.  4.  4.  3.  2.]]]]`.
        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / delta_z)
        # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
        # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  1.  0.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  0.  1.]]
        #                           [[ 1.  1.  1.  0.  0.]
        #                            [ 0.  0.  0.  1.  0.]
        #                            [ 0.  0.  0.  0.  1.]
        #                            [ 0.  0.  0.  0.  0.]
        #                            [ 0.  0.  0.  0.  0.]]]]`.
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)
        # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
        #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
        weights = weights[:, None, :]
        # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
        # in Eq7.
        # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.1 0. ]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.  0.1]]
        #                     [[ 0.1  0.2  0.5  0.  0. ]
        #                      [ 0.   0.   0.   0.1 0. ]
        #                      [ 0.   0.   0.   0.  0.1]
        #                      [ 0.   0.   0.   0.  0. ]
        #                      [ 0.   0.   0.   0.  0. ]]]]`.
        inner_prod = clipped_quotient * weights
        # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
        #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
        projection = tf.reduce_sum(inner_prod, 3)
        projection = tf.reshape(projection, [batch_size, num_dims])
        return projection