# casl paper implementation

# Pass in the embeddings from bbf into this 
import tensorflow as tf

def _attention_multimodal(self, input_i, input_a, input_h, attn_dim, fusion_mode):
        linear_i = tf.layers.dense(inputs=input_i, units=attn_dim, activation=None, name='linear_i')
        linear_a = tf.layers.dense(inputs=input_a, units=attn_dim, activation=None, name='linear_a')
        linear_h = tf.layers.dense(inputs=input_h, units=attn_dim, activation=None, name='linear_h')

        tanh_layer = tf.add_n([linear_i, linear_a, linear_h]) 
        tanh_layer = tf.layers.dense(inputs=tanh_layer, units=attn_dim, activation=tf.tanh, name='tanh_layer')
        
        softmax_attention = tf.layers.dense(inputs=tanh_layer, units=Config.NMODES, activation=tf.nn.softmax, name='softmax_attention')
        feat_i_attention = tf.multiply(input_i, tf.reshape(softmax_attention[:,0], [-1,1]), name='feat_i_attention')
        feat_a_attention = tf.multiply(input_a, tf.reshape(softmax_attention[:,1], [-1,1]), name='feat_a_attention')

        if fusion_mode == self.FUSION_SUM:
            fused_layer = tf.add(feat_i_attention, feat_a_attention, name='fused_layer')
        elif fusion_mode == self.FUSION_CONC:
            fused_layer = tf.concat([feat_i_attention, feat_a_attention], axis=1, name='fused_layer')

        return fused_layer, softmax_attention 

# page 3 Casl Architecture 
