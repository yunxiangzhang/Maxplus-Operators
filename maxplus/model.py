# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

TRAINSET_SIZE = 55000
TESTSET_SIZE = 10000
VALSET_SIZE = 5000
IMAGE_SIZE = 28
CHANNEL_SIZE = 1
NUM_CLASSES = 10
SEED = 111
BATCH_SIZE = 1650


def _activation_summary(tensor):
    """
    Create summaries for a given tensor.
    Args:
      tensor: a tf.Tensor.
    Returns:
      None.
    """
    tf.summary.histogram(tensor.op.name + '/activations', tensor)
    tf.summary.scalar(tensor.op.name + '/sparsity', tf.nn.zero_fraction(tensor))


def _variable_on_cpu(name, shape, initializer):
    """
    Create a variable on CPU memory.
    Args:
      name: name for the variable.
      shape: a list of integers.
      initializer: an initializer for the variable.
    Returns:
      A tf.Variable.
    """
    with tf.device('/cpu:0'):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=True)


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Create a variable with weight decay on CPU memory.
    Args:
      name: name for the variable.
      shape: a list of integers.
      stddev: standard deviation of a truncated Gaussian distribution.
      wd: L2 loss weight decay factor.
    Returns:
      A tf.Variable.
    """
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(mean=0.0, stddev=stddev, seed=SEED, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd / 3, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def maxplus(inputs, in_units, out_units, weights):
    """
    Implement maxplus layers.
    Args:
      inputs: input vector.
      in_units: number of input units.
      out_units: number of output units.
      weights: maxplus weight matrix.
    Returns:
      Output vector.
    """
    return tf.reduce_max(tf.add(tf.concat([tf.reshape(inputs, [-1, in_units, 1])] * out_units, axis=-1), weights), axis=-2, keepdims=False, name='maxplus')


def inference(images, stddev1, stddev2, weight_decay, keep, num_units):
    """
    Build a maxplus model.
    Args:
      images: input images.
      stddev1: standard deviation of a truncated Gaussian distribution.
      stddev2: standard deviation of a truncated Gaussian distribution.
      weight_decay: L2 loss weight decay factor.
      keep: keep ratio of dropout layers.
      num_units: number of units in the dense layer.
    Returns:
      A [batch_size, num_classes] float32 tf.Tensor representing the logits.
    """
    # Dense layer 1
    with tf.variable_scope('dense1', reuse=tf.AUTO_REUSE):
        weights = _variable_with_weight_decay('weights', [IMAGE_SIZE**2, num_units], stddev=stddev1, wd=weight_decay)
        dense = tf.matmul(images, weights, name='dense')
        tf.summary.image(name='dense_weights', tensor=tf.reshape(tf.transpose(weights),
                                                                 [num_units, IMAGE_SIZE, IMAGE_SIZE, 1]), max_outputs=num_units)
        _activation_summary(dense)
    # Maxplus layer 1
    with tf.variable_scope('maxplus1', reuse=tf.AUTO_REUSE):
        weights = _variable_with_weight_decay('weights', shape=[num_units, NUM_CLASSES], stddev=stddev2, wd=None)
        logits = maxplus(tf.nn.dropout(dense, keep_prob=keep, seed=SEED), num_units, NUM_CLASSES, weights)
        _activation_summary(logits)
    return logits


def inference_eval(images, weights, biases, indices):
    """
    Build a pruned model based on the filters selected by maxplus layer.
    Args:
      images: input images.
      weights: weight matrix.
      biases: bias vector.
      indices: a list of integers.
    Returns:
      A [batch_size, num_classes] float32 tf.Tensor representing the logits.
    """
    # Output layer
    with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
        temp = tf.transpose(tf.matmul(images, tf.transpose(weights)) + biases)
        logits = tf.transpose(tf.concat([tf.reshape(tf.reduce_max(temp[indices[i]:indices[i + 1]], axis=0), [1, -1])
                                         for i in range(NUM_CLASSES)], axis=0), name='logits')
        tf.add_to_collection('losses', tf.constant(0.0))
        _activation_summary(logits)
    return logits


def loss(logits, labels):
    """
    Build the objective function by summing the L2 weight decay term and the cross entropy loss.
    Args:
      logits: logits returned by inference().
      labels: ground truth.
    Returns:
      A scalar float32 tf.Tensor representing the total loss.
    """
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='cross_entropy')
        weight_decay = tf.add_n(tf.get_collection('losses'), name='weight_decay')
        total_loss = tf.add(cross_entropy, weight_decay, name='total_loss')
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('weight_decay', weight_decay)
        tf.summary.scalar('total_loss', total_loss)
    return cross_entropy


def accuracy(logits, labels):
    """
    Compute the accuracy.
    Args:
      logits: logits returned by inference().
      labels: ground truth.
    Returns:
      A scalar float32 tf.Tensor representing the accuracy.
    """
    with tf.name_scope('accuracy'):
        predictions = tf.argmax(logits, 1, name='predictions')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def train(total_loss, global_step, lr):
    """
    Train the model and create summaries for trainable variables.
    Args:
      total_loss: total loss returned by loss().
      global_step: a counter for finished training steps.
      lr: learning rate.
    Returns:
      train_op: an op for performing one training step.
    """
    with tf.name_scope('train_op'):
        tf.summary.scalar('learning_rate', lr)
        # Compute and apply gradients.
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # Create histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # Create histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
    return apply_gradient_op
