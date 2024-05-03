import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, LayerNormalization, Input, Dropout, Dense, Lambda, Layer
import numpy as np

# it does not work -> using this goes to NAN
# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L315
def renormalize(tensor):
    shape = tensor.shape
    tensor = tf.reshape(tensor, [tensor.shape[0], -1])
    max_value = tf.reduce_max(tensor, axis=-1, keepdims=True)
    min_value = tf.reduce_max(tensor, axis=-1, keepdims=True)
    return tf.reshape(((tensor - min_value) / (max_value - min_value + 1e-5)), shape)

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
class LinearHead(Layer):
    """A linear DQN head supporting dueling networks.

    Attributes:
        advantage: Advantage layer.
        value: Value layer (if dueling).
        dueling: Bool, whether to use dueling networks.
        num_actions: int, size of action space.
        num_atoms: int, number of value prediction atoms per action.
        dtype: np dtype.
        initializer: tensorflow initializer.
    """

    def __init__(self, dueling, num_actions, num_atoms, dtype, initializer=tf.initializers.GlorotUniform()):
        super().__init__(name="q_head", dtype=dtype)
        self.dueling = dueling
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.initializer = initializer
        
        if self.dueling:
            self.advantage = Dense(
                self.num_actions * self.num_atoms,
                dtype=dtype,
                kernel_initializer=self.initializer,
                bias_initializer=self.initializer
            )
            self.value = Dense(
                self.num_atoms,
                dtype=dtype,
                kernel_initializer=self.initializer,
                bias_initializer=self.initializer
            )
        else:
            self.advantage = Dense(
                self.num_actions * self.num_atoms,
                dtype=dtype,
                kernel_initializer=self.initializer,
                bias_initializer=self.initializer
            )

    def __call__(self, x):
        if self.dueling:
            adv = self.advantage(x)
            value = self.value(x)
            # adv = adv.reshape((self.num_actions, self.num_atoms))
            adv = tf.reshape(adv, (x.shape[0], self.num_actions, self.num_atoms))
            # value = value.reshape((1, self.num_atoms))
            value = tf.reshape(value, (x.shape[0], 1, self.num_atoms))
            # logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
            logits = value + (adv - tf.reduce_mean(adv, axis=-2, keepdims=True))
        else:
            x = self.advantage(x)
            # logits = x.reshape((self.num_actions, self.num_atoms))
            logits = tf.reshape(x, (x.shape[0], self.num_actions, self.num_atoms))
        return logits
    
    def build(self, input_shape):
        if self.dueling:
            self.advantage.build(input_shape)
            self.value.build(input_shape)
        else:
            self.advantage.build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], self.num_actions, self.num_atoms))
    
    def get_config(self):
        return {"dueling": self.dueling, "num_actions": self.num_actions, "num_atoms": self.num_atoms}
        
# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L697
class BBFModel(tf.keras.Model):
    def __init__(self, input_shape, encoder_network, num_actions, hidden_dim, num_atoms, width_scale=4, renormalize=False, dueling=True, distributional=True, dtype=np.float32, initializer=tf.initializers.GlorotUniform()):
        super().__init__()
        self.renormalize = renormalize
        self.dueling = dueling
        self.distributional = distributional
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.width_scale = width_scale
        self.num_actions = num_actions
        
        if encoder_network == 'ImpalaWide':
            encoder_dims = (16,32,32) # before scaling!
            self.encoder = ScaledImpalaCNN(input_shape, dims=encoder_dims, width_scale=self.width_scale, initializer=initializer)
            latent_dim = encoder_dims[-1] * self.width_scale
        else:
            latent_dim = 64
            self.encoder = DQN_CNN(input_shape)
        
        # head used to predict Q values
        self.head = LinearHead(dueling=dueling, num_actions=num_actions, num_atoms=num_atoms, dtype=dtype, initializer=initializer)
        
        # transition model
        self.TransitionCell = tf.keras.Sequential([
            Conv2D(filters=latent_dim, kernel_size=3, strides=1, kernel_initializer=initializer, padding="SAME", activation="relu"),
            Conv2D(filters=latent_dim, kernel_size=3, strides=1, kernel_initializer=initializer, padding="SAME", activation="relu")
        ], name="transition")
        
        self.projection = Dense(units=hidden_dim, kernel_initializer=initializer, bias_initializer=initializer, name="projection")
        
        self.predictor = Dense(units=self.hidden_dim, kernel_initializer=initializer, bias_initializer=initializer, name="predictor")
    
    def encode(self, x, has_batch=False, is_rollout=False):
        latent = self.encoder(x)
        if self.renormalize:
            latent = renormalize(latent)
        return latent

    def encode_project(self, x, has_batch=False, is_rollout=False):
        latent = self.encode(x, has_batch, is_rollout)
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
            x = renormalize(x)
                
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
    
    def __call__(self, x, support, do_rollout=False, actions=None):
        spatial_latent = self.encode(x)
                
        representation = self.flatten_spatial_latent(spatial_latent, True) # changed has_batch to be True here (different from BBF code)
                
        x = self.project(representation)
        x = ReLU()(x)
        
        if do_rollout:
            spatial_latent = self.spr_rollout(spatial_latent, actions)

        logits = self.head(x) # (batch_size, num_actions, num_atoms)
        
        if self.distributional:
            probabilities = tf.squeeze(tf.nn.softmax(logits)) # (batch_size, num_actions, num_atoms)
            q_values = tf.squeeze(tf.reduce_sum(support * probabilities, axis=-1)) # (batch_size, num_actions)
            return q_values, logits, probabilities, spatial_latent, representation
        
        q_values = tf.squeeze(logits) # if num_atoms == 1 (not distributional), removes the trailing axis
                
        return q_values, None, None, spatial_latent, representation
