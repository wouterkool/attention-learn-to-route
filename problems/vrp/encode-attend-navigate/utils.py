# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tqdm import tqdm


# Embed input sequence [batch_size, seq_length, from_] -> [batch_size, seq_length, to_]
def embed_seq(input_seq, from_, to_, is_training, BN=True, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("embedding"): # embed + BN input set
        W_embed = tf.get_variable("weights",[1,from_, to_], initializer=initializer)
        embedded_input = tf.nn.conv1d(input_seq, W_embed, 1, "VALID", name="embedded_input")
        if BN == True: embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=is_training, name='layer_norm', reuse=None)
        return embedded_input


# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.variable_scope("multihead_attention", reuse=None):
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]   
        # Residual connection
        outputs += inputs # [batch_size, seq_length, n_hidden]
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
 
    return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]   
    return outputs


# Encode input sequence [batch_size, seq_length, n_hidden] -> [batch_size, seq_length, n_hidden]
def encode_seq(input_seq, input_dim, num_stacks, num_heads, num_neurons, is_training, dropout_rate=0.):
    with tf.variable_scope("stack"):
        for i in range(num_stacks): # block i
            with tf.variable_scope("block_{}".format(i)): # Multihead Attention + Feed Forward
                input_seq = multihead_attention(input_seq, num_units=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, is_training=is_training)
                input_seq = feedforward(input_seq, num_units=[num_neurons, input_dim], is_training=is_training)
        return input_seq # encoder_output is the ref for actions [Batch size, Sequence Length, Num_neurons]
            

# From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
# predict a distribution over next decoder input
def pointer(encoded_ref, query, mask, W_ref, W_q, v, C=10., temperature=1.0):
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1) # [Batch size, 1, n_hidden]
    scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1]) # [Batch size, seq_length]
    scores = C*tf.tanh(scores/temperature) # control entropy
    masked_scores =  tf.clip_by_value(scores -100000000.*mask, -100000000., 100000000.) # [Batch size, seq_length]
    return masked_scores


# From a query [Batch size, n_hidden], glimpse at a set of reference vectors (ref) [Batch size, seq_length, n_hidden]
def full_glimpse(ref, from_, to_, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("glimpse"):
        W_ref_g =tf.get_variable("W_ref_g",[1,from_, to_],initializer=initializer)
        W_q_g =tf.get_variable("W_q_g",[from_, to_],initializer=initializer)
        v_g =tf.get_variable("v_g",[to_],initializer=initializer)
        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, W_ref_g, 1, "VALID", name="encoded_ref_g") # [Batch size, seq_length, n_hidden]
        scores_g = tf.reduce_sum(v_g * tf.tanh(encoded_ref_g), [-1], name="scores_g") # [Batch size, seq_length]
        attention_g = tf.nn.softmax(scores_g, name="attention_g")
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g,2))
        glimpse = tf.reduce_sum(glimpse,1)
        return glimpse