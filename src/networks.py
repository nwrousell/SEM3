import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, LayerNormalization, Input, Dropout, Dense, Lambda
import numpy as np

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L315
def renormalize(tensor, has_batch=False):
    shape = tensor.shape
    if not has_batch:
        tensor = np.expand_dims(tensor, 0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    max_value = np.max(tensor, axis=-1, keepdims=True)
    min_value = np.min(tensor, axis=-1, keepdims=True)
    return ((tensor - min_value) / (max_value - min_value + 1e-5)).reshape(*shape)

def residual_stage(x, dim, width_scale, num_blocks, initializer, norm_type, dropout):
    """Residual block with two convolutional layers."""
    conv_out = Conv2D(filters=dim*width_scale,kernel_size=3,strides=1, padding='SAME', kernel_initializer=initializer)(x)
    conv_out = MaxPool2D(pool_size=3, strides=2)(conv_out)
        
    for _ in range(num_blocks):
        block_input = conv_out
        conv_out = ReLU()(block_input)
        
        if norm_type == 'batch':
            conv_out = BatchNormalization()(conv_out)
        elif norm_type == 'layer':
            conv_out = LayerNormalization()(conv_out)
        
        if dropout != 0.0:
            conv_out = Dropout(rate=dropout)(conv_out)
        
        conv_out = Conv2D(filters=dim*width_scale, kernel_size=3, strides=1, padding='SAME', kernel_initializer=initializer)(conv_out)
        
        conv_out = ReLU()(conv_out)
        if norm_type == 'batch':
            conv_out = BatchNormalization()(conv_out)
        elif norm_type == 'layer':
            conv_out = LayerNormalization()(conv_out)
        
        if dropout != 0.0:
            conv_out = Dropout(rate=dropout)(conv_out)
        
        conv_out = Conv2D(filters=dim*width_scale, kernel_size=3, strides=1, padding='SAME', kernel_initializer=initializer)(conv_out)

        conv_out += block_input
    return conv_out

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L448
def ScaledImpalaCNN(
    input_shape, 
    dims=(16,32,32), 
    width_scale=4, 
    initializer=tf.initializers.GlorotUniform(),
    dropout=0.0,
    norm_type='none' # valid options are 'batch', 'layer', or 'none'
    ):
    
    inputs = Input(shape=input_shape)
    
    x = inputs
    
    for d in dims:
        x = residual_stage(x, dim=d, width_scale=width_scale, num_blocks=2, initializer=initializer, norm_type=norm_type, dropout=dropout)
    
    x = ReLU()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="encoder")

def DQN_CNN(input_shape, padding='VALID', dims=(32, 64, 64), width_scale=1, dropout=0.0, initializer=tf.initializers.GlorotUniform()):
    inputs = Input(shape=input_shape)
    # x = x[None, Ellipsis]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    x = inputs
    for layer in range(3):
        x = Conv2D(
            filters=int(dims[layer] * width_scale),
            kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
            strides=(stride_sizes[layer], stride_sizes[layer]),
            kernel_initializer=initializer,
            padding=padding,
        )(x)
        x = Dropout(dropout)(x)
        x = ReLU()(x)
    outputs = x
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L242
class LinearHead(tf.keras.Model):
    def __init__(self, num_actions, num_atoms, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        self.advantage = Dense(units = num_actions * num_atoms, kernel_initializer=initializer, dtype=dtype)
        
    def __call__(self, x):
        logits = self.advantage(x)
        return logits
        
# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L697
class BBFModel(tf.keras.Model):
    def __init__(self, input_shape, encoder_network, num_actions, hidden_dim, num_atoms, width_scale=4, renormalize=False, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        super().__init__()
        self.renormalize = renormalize
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.width_scale = width_scale
        self.num_actions = num_actions
        
        if encoder_network == 'ImpalaWide':
            encoder_dims = (16,32,32) # pre-scaled!
            self.encoder = ScaledImpalaCNN(input_shape, dims=encoder_dims, width_scale=self.width_scale)
            latent_dim = encoder_dims[-1] * self.width_scale
        else:
            latent_dim = 64
            self.encoder = DQN_CNN(input_shape)
        
        # head used to predict Q values
        # self.head = LinearHead(num_actions=num_actions, num_atoms=num_atoms, dtype=dtype, initializer=initializer)
        self.head = Dense(units = num_actions * num_atoms, name="head")
        
        # transition model
        # self.ConvTMCell = ConvTransitionModel("transition_model", num_actions, latent_dim, renormalize, initializer, dtype=dtype)
        # self.transition_model = TransitionModel(num_actions=num_actions, latent_dim=latent_dim, renormalize=renormalize, initializer=initializer)
        self.TransitionCell = tf.keras.Sequential([
            Conv2D(filters=latent_dim, kernel_size=3, strides=1, kernel_initializer=initializer, padding="SAME", activation="relu"),
            Conv2D(filters=latent_dim, kernel_size=3, strides=1, kernel_initializer=initializer, padding="SAME", activation="relu")
        ], name="transition")
        
        # projection layer (SPR uses the first layer of the Q_head for this)
        self.projection = Dense(units=hidden_dim, kernel_initializer=initializer, dtype=dtype, name="projection")
        
        # predictor
        self.predictor = Dense(units=self.hidden_dim, kernel_initializer=initializer, name="predictor")
    
    def encode(self, x):
        latent = self.encoder(x)
        if self.renormalize:
            raise NotImplementedError("renormalization is not implemented!")
            latent = renormalize(latent)
        return latent

    def encode_project(self, x, has_batch=False, is_rollout=False):
        latent = self.encode(x)
        representation = self.flatten_spatial_latent(latent, has_batch, is_rollout)
        return self.project(representation)

    def project(self, x):
        projected = self.projection(x)
        return projected
    
    def transition(self, x, action):
        action_onehot = tf.one_hot(action, self.num_actions)
        action_onehot = tf.reshape(action_onehot, [action_onehot.shape[0], 1, 1, action_onehot.shape[1]])
        action_onehot = tf.broadcast_to(action_onehot, [action_onehot.shape[0], x.shape[-3], x.shape[-2], action_onehot.shape[-1]])
        x = tf.concat([x, action_onehot], -1)
        
        x = self.TransitionCell(x)
                            
        if self.renormalize:
            raise Exception("Renormalization has not been implemented yet")
                
        return x, x
    
    def predict_transitions(self, x, actions):
        """Predict k future states given initial state and k actions"""
        states = []
        for i in range(actions.shape[1]):
            # x, pred_state = self.ConvTMCell(x, actions[:,i])
            x, pred_state = self.transition(x, actions[:, i])
            states.append(pred_state)
        return x, tf.stack(states)
    
    # @functools.partial(jax.vmap, in_axes=(None, 0))
    def spr_predict(self, x):
        projected = self.project(x)
        return self.predictor(projected)
    
    @tf.function
    def spr_rollout(self, latent, actions):
        _, pred_latents = self.predict_transitions(latent, actions)

        representations = self.flatten_spatial_latent(pred_latents, True, True)
                
        predictions = tf.vectorized_map(self.spr_predict, representations) # changing from the jax vmap (see the commented out functools.partial above spr_predict)
        
        return predictions
    
    def flatten_spatial_latent(self, spatial_latent, has_batch=False, is_rollout=False):
        if has_batch:
            if is_rollout:
                representation = tf.reshape(spatial_latent, [spatial_latent.shape[0], spatial_latent.shape[1], -1])
            else:
                representation = tf.reshape(spatial_latent, [spatial_latent.shape[0], -1])
        else:
            representation = tf.reshape(spatial_latent, [-1])

        return representation
    
    def __call__(self, x, do_rollout=False, actions=None):
        spatial_latent = self.encode(x)
                
        representation = self.flatten_spatial_latent(spatial_latent, True) # changed has_batch to be True here (different from BBF code)
                
        # Single hidden layer
        x = self.project(representation)
        x = ReLU()(x)
        
        if do_rollout:
            spatial_latent = self.spr_rollout(spatial_latent, actions)

        logits = self.head(x)
        
        q_values = tf.squeeze(logits)
                
        return q_values, spatial_latent, representation

def get_weight_dict(model):
    weight_dict = {}
    for layer in model.layers:
        weight_dict[layer.name] = layer.get_weights()
    return weight_dict

def set_weights(model, weight_dict):
    for layer in model.layers:
        layer.set_weights(weight_dict[layer.name])

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
            new_weights[layer_name] = weights_a
            
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
def weights_reset(weights_dict, initializer=tf.initializers.GlorotUniform()):
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
    
    # reset weights of Q head fully
    new_weights = interpolate_weights(weights_dict, rand_weights, 1, layers=["head"])
    
    # interpolate encoder/transition weights 50% (shrink-and-perturb)
    new_weights = interpolate_weights(new_weights, rand_weights, 0.5, layers=["encoder", "transition"])
    
    return new_weights