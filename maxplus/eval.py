# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import model
import os

DATA_DIR = os.path.join(os.path.abspath('..'), 'mnist')
EVAL_DIR = os.path.join(os.path.abspath('.'), 'eval')
MODEL_DIR = os.path.join(os.path.abspath('.'), 'model')
IMAGE_SIZE = model.IMAGE_SIZE
BATCH_SIZE = model.BATCH_SIZE
NUM_CLASSES = model.NUM_CLASSES
SEED = model.SEED


def read_active_filters(num_units, threshold):
    assert 0.0 <= threshold <= 1.0, "threshold should be chosen between [0.0, 1.0]!"
    with tf.Graph().as_default():
        # Restore the pre-trained variables.
        weights1 = model._variable_with_weight_decay('dense1/weights', shape=[IMAGE_SIZE**2, num_units], stddev=0.05, wd=None)
        weights2 = model._variable_with_weight_decay('maxplus1/weights', shape=[num_units, NUM_CLASSES], stddev=0.05, wd=None)
        to_restore = {}
        to_restore[weights1.op.name] = weights1
        to_restore[weights2.op.name] = weights2
        saver = tf.train.Saver(var_list=to_restore)

        # Select the active filters.
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            print("Restoring the model ...")
            saver.restore(sess, os.path.join(MODEL_DIR, "model.ckpt"))
            w1 = sess.run(weights1).transpose()
            w2 = sess.run(weights2).transpose()
            w2_max = np.max(w2, axis=-1)
            weights = []
            biases = []
            indices = [0]
            for i in range(NUM_CLASSES):
                for j, bias in enumerate(w2[i]):
                    if bias >= threshold * w2_max[i]:
                        weights.append(w1[j])
                        biases.append(bias)
                indices.append(len(biases))

    return np.array(weights), np.array(biases), indices


def eval(num_units, threshold):
    # Load the active filters
    w1, w2, indices = read_active_filters(num_units, threshold)

    with tf.Graph().as_default() as g:
        # Load the MNIST dataset
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)
        np.random.seed(seed=SEED)

        # Insert placeholders for input images and labels.
        with tf.device('/cpu:0'):
            images_train = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='train_images')
            labels_train = tf.placeholder(tf.int64, shape=[None], name='train_labels')
            images_val = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='validation_images')
            labels_val = tf.placeholder(tf.int64, shape=[None], name='validation_labels')
            images_test = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='test_images')
            labels_test = tf.placeholder(tf.int64, shape=[None], name='test_labels')

        # Build a Graph to perform the inference.
        weights = tf.constant(w1)
        biases = tf.constant(w2)
        logits_train = model.inference_eval(images_train, weights, biases, indices)
        logits_val = model.inference_eval(images_val, weights, biases, indices)
        logits_test = model.inference_eval(images_test, weights, biases, indices)

        # Compute the total loss.
        loss = model.loss(logits_train, labels_train)

        # Compute the accuracy
        accuracy_val = model.accuracy(logits_val, labels_val)
        accuracy_test = model.accuracy(logits_test, labels_test)

        # Write summaries to the disk
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(EVAL_DIR, g)

        # Demand GPU resources
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # Load the dataset
            val_images, val_labels = mnist.validation.images, mnist.validation.labels
            test_images, test_labels = mnist.test.images, mnist.test.labels
            batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)

            accuracy_val_value = 0.0
            accuracy_test_value = 0.0
            for i in range(10):
                if i < 5:
                    accuracy_val_value += sess.run(accuracy_val, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                        i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000]})
                accuracy_test_value += sess.run(accuracy_test, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                    i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000]})
            loss_value = sess.run(loss, feed_dict={images_train: batch_images, labels_train: batch_labels,
                                                   images_val: val_images[0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000]})
            accuracy_val_value /= 5.0
            accuracy_test_value /= 10.0

            summary_writer.add_summary(sess.run(summary_op, feed_dict={images_train: batch_images, labels_train: batch_labels,
                                                                       images_val: val_images[0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000]}))
            summary_writer.flush()

    return loss_value, accuracy_val_value, accuracy_test_value, indices[NUM_CLASSES]


def main(argv=None):
    if not tf.gfile.Exists(EVAL_DIR):
        tf.gfile.MakeDirs(EVAL_DIR)
    format_str = ('Restored model: num of filters = %d; loss = %.3f, val accuracy = %.3f, test accuracy = %.3f.')
    try:
        loss_value, accuracy_val_value, accuracy_test_value, num = eval(144, 0.6)
        print(format_str % (num, loss_value, accuracy_val_value, accuracy_test_value))
    except Exception as e:
        print(format_str % (-1, -1.0, -1.0, -1.0))


if __name__ == '__main__':
    tf.app.run()
