import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, LayerNormalization, Input, Dropout, Dense, Lambda
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
    conv_out = Conv2D(filters=dim*width_scale,kernel_size=3,strides=1, padding='SAME', kernel_initializer=initializer)(x) # ! here: 
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
    return keras.Model(inputs=inputs, outputs=x)
    
# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L325
class ConvTransitionModelCell:
    def __init__(self, num_actions: int, latent_dim: int, renormalize: bool, initializer=tf.initializers.GlorotUniform(), dtype=np.float32):
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.renormalize = renormalize
        self.initializer = initializer
    
    def __call__(self, x, action):
        sizes = [self.latent_dim, self.latent_dim]
        kernel_sizes = [3, 3]
        stride_sizes = [1, 1]
        
        action_onehot = tf.one_hot(action, self.num_actions)
        action_onehot = tf.reshape(action_onehot, [action_onehot.shape[0], 1, 1, action_onehot.shape[1]])
        action_onehot = tf.broadcast_to(action_onehot, [action_onehot.shape[0], x.shape[-3], x.shape[-2], action_onehot.shape[-1]])
        x = tf.concat([x, action_onehot], -1)
        for i in range(len(sizes)):
            x = Conv2D(filters=sizes[i], 
                       kernel_size=kernel_sizes[i], 
                       strides=stride_sizes[i], 
                       kernel_initializer=self.initializer,
                       padding='SAME'
                       )(x)
            x = ReLU()(x)
                            
        if self.renormalize:
            raise Exception("Renormalization has not been implemented yet (go do that Noah)")
        
        return x, x
    
class TransitionModel(keras.Model):
    def __init__(self, num_actions: int, latent_dim: int, renormalize: bool, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.renormalize = renormalize
        self.initializer = initializer
        
        self.ConvTMCell = ConvTransitionModelCell(num_actions, latent_dim, renormalize, initializer, dtype=dtype)
    
    def __call__(self, x, actions):
        """Predict k future states given initial state and k actions"""
        states = []
        for i in range(actions.shape[1]):
            x, pred_state = self.ConvTMCell(x, actions[:,i])
            states.append(pred_state)
        return x, tf.stack(states)

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
    return keras.Model(inputs=inputs, outputs=outputs)
        
        
        

# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L242
class LinearHead(keras.Model):
    def __init__(self, num_actions, num_atoms, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        self.advantage = Dense(units = num_actions * num_atoms, kernel_initializer=initializer, dtype=dtype)
        
    def __call__(self, x):
        logits = self.advantage(x)
        return logits
        
# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L697
class BBFModel(keras.Model):
    def __init__(self, input_shape, num_actions, hidden_dim, num_atoms, width_scale=4, renormalize=False, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        
        self.renormalize = renormalize
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.width_scale = width_scale
        
        # self.encoder = ScaledImpalaCNN(input_shape)
        encoder_dims = (16,32,32)
        self.encoder = ScaledImpalaCNN(input_shape, dims=encoder_dims, width_scale=self.width_scale)
        latent_dim = encoder_dims[-1] * self.width_scale
        
        # head used to predict Q values
        # self.head = LinearHead(num_actions=num_actions, num_atoms=num_atoms, dtype=dtype, initializer=initializer)
        self.head = Dense(units = num_actions * num_atoms)
        
        # transition model
        self.transition_model = TransitionModel(num_actions=num_actions, latent_dim=latent_dim, renormalize=renormalize, initializer=initializer)
        
        # projection layer (SPR uses the first layer of the Q_head for this)
        self.projection = Dense(units=hidden_dim, kernel_initializer=initializer, dtype=dtype)
        
        # predictor
        self.predictor = Dense(units = self.hidden_dim, kernel_initializer=initializer)
    
    def encode(self, x):
        latent = self.encoder(x)
        if self.renormalize:
            raise NotImplementedError("renormalization is not implemented!")
            latent = renormalize(latent)
        return latent

    def encode_project(self, x):
        latent = self.encode(x)
        representation = self.flatten_spatial_latent(latent)
        return self.project(representation)

    def project(self, x):
        projected = self.projection(x)
        return projected
    
    # @functools.partial(jax.vmap, in_axes=(None, 0))
    def spr_predict(self, x):
        projected = self.project(x)
        return self.predictor(projected)
    
    def spr_rollout(self, latent, actions):
        _, pred_latents = self.transition_model(latent, actions)

        print("pred latents", pred_latents.shape)

        representations = self.flatten_spatial_latent(pred_latents, True, True)
                
        predictions = tf.vectorized_map(self.spr_predict, representations) # changing from the jax vmap (see the commented out functools.partial above spr_predict)
        
        return predictions
    
    def flatten_spatial_latent(self, spatial_latent, has_batch=False, is_rollout=False):
        # logging.info('Spatial latent shape: %s', str(spatial_latent.shape))
        # if self.use_spatial_learned_embeddings:
        #     representation = self.embedder(spatial_latent)
        if has_batch:
            if is_rollout:
                representation = tf.reshape(spatial_latent, [spatial_latent.shape[0], spatial_latent.shape[1], -1])
            else:
                representation = tf.reshape(spatial_latent, [spatial_latent.shape[0], -1])
            # representation = spatial_latent.reshape(spatial_latent.shape[0], -1)
        else:
            # representation = spatial_latent.reshape(-1)
            representation = tf.reshape(spatial_latent, [-1])
            
        # logging.info(
        #     'Flattened representation shape: %s', str(representation.shape)
        # )
        return representation
    
    def __call__(self, x, do_rollout=False, actions=None):
        spatial_latent = self.encode(x)
        
        print("post-encoder shape:", spatial_latent.shape)
        
        representation = self.flatten_spatial_latent(spatial_latent, True) # changed has_batch to be True here (different from BBF code)
        
        # Single hidden layer
        x = self.project(representation)
        x = ReLU()(x)
        print("post projection:", x.shape)

        if do_rollout:
            spatial_latent = self.spr_rollout(spatial_latent, actions)

        logits = self.head(x)
        
        q_values = tf.squeeze(logits)
        
        return q_values, spatial_latent, representation

# TODO - don't know if this is all vectorized correctly (do they do this for each action or check that all the q-values are correct?)
def Q_learning_loss(q_pred, target_q_network, states, actions, discount_factor, observed_rewards, update_horizon=1):
    assert len(actions) == update_horizon == len(states) == len(observed_rewards)
    
    # q_pred = q_network(states[0], actions[0])
    
    cum_loss = 0
    cum_obs_rewards = observed_rewards[0]
    for k in range(1, update_horizon+1):
        target_q = cum_obs_rewards + tf.math.pow(discount_factor, k) * target_q_network(states[k], actions[k])
        cum_loss += tf.square(q_pred - target_q) # ! add a sum here probably
        cum_obs_rewards += tf.math.pow(discount_factor, k) * observed_rewards[k]
        
    return cum_loss

def self_predictive_representations_loss(max_depth=5):
    # TODO
    pass
    