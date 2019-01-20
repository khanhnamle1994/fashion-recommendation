'''
This is the resnet structure.
'''

import tensorflow as tf
from hyper_parameters import *

BN_EPSILON = 0.001
NUM_LABELS = 6

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    # tf.histogram_summary(tensor_name + '/activations', x)
    # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.fc_weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer)

    fc_w2 = create_variables(name='fc_weights2', shape=[input_dim, 4], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b2 = create_variables(name='fc_bias2', shape=[4], initializer=tf.zeros_initializer)


    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    fc_h2 = tf.matmul(input_layer, fc_w2) + fc_b2
    return fc_h, fc_h2


def conv_bn_relu_layer(input_layer, filter_shape, stride, second_conv_residual=False,
                       relu=True):
    out_channel = filter_shape[-1]
    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else: filter = create_variables(name='conv2', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    mean, variance = tf.nn.moments(conv_layer, axes=[0, 1, 2])

    if second_conv_residual is False:
        beta = tf.get_variable('beta', out_channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', out_channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    else:
        beta = tf.get_variable('beta_second_conv', out_channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma_second_conv', out_channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

    bn_layer = tf.nn.batch_normalization(conv_layer, mean, variance, beta, gamma, BN_EPSILON)
    if relu:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, second_conv_residual=False):
    in_channel = input_layer.get_shape().as_list()[-1]
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])

    if second_conv_residual is False:
        beta = tf.get_variable('beta', in_channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', in_channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    else:
        beta = tf.get_variable('beta_second_conv', in_channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma_second_conv', in_channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    relu_layer = tf.nn.relu(bn_layer)

    if second_conv_residual is False:
        filter = create_variables(name='conv', shape=filter_shape)
    else: filter = create_variables(name='conv2', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block_new(input_layer, output_channel, first_block=False):
    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    if first_block:
        filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
        conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    else:
        conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)
    conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1,
                               second_conv_residual=True)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse, keep_prob_placeholder):
    '''
    total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    '''
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        # activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block_new(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block_new(layers[-1], 16)
            # activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block_new(layers[-1], 32)
            # activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block_new(layers[-1], 64)
            layers.append(conv3)
        # assert conv3.get_shape().as_list()[1:] == [16, 16, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(layers[-1], axes=[0, 1, 2])
        beta = tf.get_variable('beta', in_channel, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', in_channel, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        bn_layer = tf.nn.batch_normalization(layers[-1], mean, variance, beta, gamma, BN_EPSILON)
        relu_layer = tf.nn.relu(bn_layer)


        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        cls_output, bbx_output = output_layer(global_pool, NUM_LABELS)
        layers.append(cls_output)
        layers.append(bbx_output)

    return cls_output, bbx_output, global_pool
