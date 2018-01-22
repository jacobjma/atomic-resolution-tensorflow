import tensorflow as tf
import numpy as np

wd = 3e-4

def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + "/activations", x)
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(x))

def variable_summaries(var):
    
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + "/mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + "/sttdev", stddev)
            tf.summary.scalar(name + "/max", tf.reduce_max(var))
            tf.summary.scalar(name + "/min", tf.reduce_min(var))
            tf.summary.histogram(name, var)

def parametric_relu(x, name="parametric_relu"):
    with tf.name_scope(name):
        alpha = tf.get_variable("alpha", x.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(0.0, x) + tf.minimum(0.0, alpha * x)

def batch_norm(x, epsilon=1e-5, momentum=0.999, train=True, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, scope=name)        
        
def get_kernel_size(factor):
    return 2 * factor - factor % 2

def get_bias(shape):
    init = tf.constant_initializer(value=.1, dtype=tf.float32)
    var = tf.get_variable(name="biases", initializer=init, shape=shape)
    variable_summaries(var)
    return var

def get_conv_filter(shape):
    init = tf.random_normal_initializer(0, 0.1, dtype=tf.float32)

    var = tf.get_variable(name="filter", initializer=init, shape=shape, 
                        regularizer = tf.contrib.layers.l2_regularizer(scale=1.))
    
    variable_summaries(var)
    return var

def get_deconv_filter(shape):
    f = np.ceil(shape[0]/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(shape[0]):
        for y in range(shape[1]):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name="upsample_filter", initializer=init, shape=weights.shape,
                        regularizer = tf.contrib.layers.l2_regularizer(scale=1.))
    return var

def upsample_layer(x, output_shape, factor=2, name="upsample"):
    
    strides = [1, factor, factor, 1]
    kernel_size = get_kernel_size(factor)
    filter_shape = [kernel_size,kernel_size,output_shape[-1],x.get_shape()[-1]]
    
    assert output_shape[-1]<=x.get_shape()[-1]
    
    with tf.variable_scope(name):
        filter = get_deconv_filter(filter_shape)
        
        conv = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape, strides=[1, factor, factor, 1], padding="SAME")
        
        
        conv = parametric_relu(conv)
        
        conv = batch_norm(conv)
        
        activation_summary(conv)
    
        return conv

def pool_layer(x, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def conv_layer(x, output_features, kernel_size=3, name="conv"):
    
    filter_shape = [kernel_size, kernel_size, x.get_shape()[-1], output_features]
    
    with tf.variable_scope(name) as scope:
        filter = get_conv_filter(filter_shape)
        conv = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding="SAME", name="conv")

        conv_biases = get_bias(output_features)
        conv = tf.nn.bias_add(conv, conv_biases)
        
        conv = parametric_relu(conv)
        conv = batch_norm(conv)
        
        activation_summary(conv)
        
        return conv

def score_layer(x, output_features, name="score"):
    
    filter_shape = [1, 1, x.get_shape()[3].value, output_features]

    with tf.variable_scope(name) as scope:
        filter = get_conv_filter(filter_shape)
        conv = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding="SAME", name="score")
        
        conv_biases = get_bias(output_features)
        conv = tf.nn.bias_add(conv, conv_biases)

        activation_summary(conv)
        
        return conv
        
def skip(x, y):
    return tf.add(x, y)
        
def conv_repeat(x, output_features, kernel_size=3, repeats=3, name="conv_repeat"):
    
    with tf.variable_scope(name) as scope:
        for i in range(repeats):
            name = "conv_%d" % i
            x = conv_layer(x, output_features, kernel_size, name)
        
        return x

def res_block(x, output_features, kernel_size=3, repeats=3, name="res_block"):
    
    with tf.variable_scope(name) as scope:
        y = conv_repeat(x, output_features, kernel_size, repeats, name)
        return skip(x,y)
