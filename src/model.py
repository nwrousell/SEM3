import keras.initializers
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, ReLU, LayerNormalization, Input, Dropout

# impala CNN (15 layer ResNet) but 4x as wide
# https://arxiv.org/pdf/1802.01561.pdf

# impala architecture has multiple actors, which generate trajectories of experience, and 
# one or more learners, which learn in an off-policy fashion


def residual_stage(x, dims, width_scale, num_blocks, initializer, norm_type, dropout):
    """Residual block with two convolutional layers."""
    conv_out = Conv2D(filters=dims*width_scale,kernel_size=3,strides=1, padding='SAME', kernel_initializer=initializer)(x)
    conv_out = MaxPool2D(pool_size=3, strides=2)(x)
    
    for _ in range(num_blocks):
        block_input = conv_out
        conv_out = ReLU()(block_input)
        
        if norm_type == 'batch':
            conv_out = BatchNormalization()(conv_out)
        elif norm_type == 'layer':
            conv_out = LayerNormalization()(conv_out)
        
        if dropout != 0.0:
            conv_out = Dropout(rate=dropout)(conv_out)
        
        conv_out = Conv2D(filters=dims*width_scale, kernel_size=3, strides=1, padding='SAME', kernel_initializer=initializer)(conv_out)
        
        conv_out = ReLU()(conv_out)
        if norm_type == 'batch':
            conv_out = BatchNormalization()(conv_out)
        elif norm_type == 'layer':
            conv_out = LayerNormalization()(conv_out)
        
        if dropout != 0.0:
            conv_out = Dropout(rate=dropout)(conv_out)
        
        conv_out = Conv2D(filters=dims*width_scale, kernel_size=3, strides=1, padding='SAME', kernel_initializer=initializer)(conv_out)

        conv_out += block_input
    return conv_out


# https://github.com/google-research/google-research/blob/a3e7b75d49edc68c36487b2188fa834e02c12986/bigger_better_faster/bbf/spr_networks.py#L448
# GlorotUniform is the same as Xavier initializer: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
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
        x = residual_stage(x, dims=d, width_scale=width_scale, num_blocks=2, initializer=initializer, norm=norm_type, dropout=dropout)
    
    x = ReLU()(x)
    return x