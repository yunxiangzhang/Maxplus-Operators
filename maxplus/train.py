# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import datetime
import model
import time
import os

DATA_DIR = os.path.join(os.path.abspath('..'), 'mnist')
TRAIN_DIR = os.path.join(os.path.abspath('.'), 'train')
MODEL_DIR = os.path.join(os.path.abspath('.'), 'model')
EARLY_STOPPING_STEPS = 300
DECAY_N_TIMES = 2
IMAGE_SIZE = model.IMAGE_SIZE
TRAINSET_SIZE = model.TRAINSET_SIZE
BATCH_SIZE = model.BATCH_SIZE
LOG_FREQ = 10
SEED = model.SEED


def train(initial_lr, stddev1, stddev2, weight_decay, keep, num_units, training=True):
    with tf.Graph().as_default() as g:
        # Load the MNIST dataset.
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)
        np.random.seed(seed=SEED)

        # Create a counter for finished training steps.
        global_step = tf.train.get_or_create_global_step()

        # Insert placeholders for input images and labels.
        with tf.device('/cpu:0'):
            images_train = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='train_images')
            labels_train = tf.placeholder(tf.int64, shape=[None], name='train_labels')
            images_val = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='validation_images')
            labels_val = tf.placeholder(tf.int64, shape=[None], name='validation_labels')
            images_test = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE**2], name='test_images')
            labels_test = tf.placeholder(tf.int64, shape=[None], name='test_labels')

        # Insert a placeholder for the learning rate.
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        # Build a Graph to perform the inference.
        logits_train = model.inference(images_train, stddev1, stddev2, weight_decay, keep, num_units)
        logits_val = model.inference(images_val, stddev1, stddev2, weight_decay, 1.0, num_units)
        logits_test = model.inference(images_test, stddev1, stddev2, weight_decay, 1.0, num_units)

        # Compute the total loss.
        loss = model.loss(logits_train, labels_train)

        # Compute the accuracy.
        accuracy_val = model.accuracy(logits_val, labels_val)
        accuracy_test = model.accuracy(logits_test, labels_test)

        # Perform one training step.
        train_op = model.train(loss, global_step, learning_rate)

        # Write summaries to the disk.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(TRAIN_DIR, g)

        # Demand GPU resources.
        gpu_options = tf.GPUOptions(allow_growth=True)

        # Save or restore the model.
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # Train a model.
            if training:
                # Initialize all variables.
                sess.run(tf.global_variables_initializer())

                step = 0
                best_step_val = -1
                best_accuracy_val = -1.
                final_accuracy_test = -1.
                early_stopping_counter = 0
                decay_counter = 0
                lr = initial_lr
                decay_steps = []
                decay_loss = []
                decay_accuracy = []
                start_time = time.time()

                # Load validation set and testing set.
                val_images, val_labels = mnist.validation.images, mnist.validation.labels
                test_images, test_labels = mnist.test.images, mnist.test.labels

                while True:
                    # Get one training batch.
                    batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)

                    if step % LOG_FREQ == 0:
                        accuracy_val_value = 0.0
                        accuracy_test_value = 0.0
                        for i in range(10):
                            if i < 5:
                                accuracy_val_value += sess.run(accuracy_val, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                                    i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000], learning_rate: lr})
                            accuracy_test_value += sess.run(accuracy_test, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                                i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000], learning_rate: lr})
                        loss_value = sess.run(loss, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[
                                              0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000], learning_rate: lr})
                        accuracy_val_value /= 5.0
                        accuracy_test_value /= 10.0

                        if accuracy_val_value > best_accuracy_val:
                            best_accuracy_val = accuracy_val_value
                            best_step_val = step
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1

                        duration = time.time() - start_time
                        examples_per_sec = LOG_FREQ * BATCH_SIZE / duration
                        sec_per_epoch = float((duration / LOG_FREQ) * (TRAINSET_SIZE / BATCH_SIZE))

                        format_str = ('%s: step %d, loss = %.3f, val accuracy = %.3f, test accuracy = %.3f (%.1f examples/sec; %.1f sec/epoch)')
                        date_time_now = datetime.datetime.now()
                        time_now = datetime.time(date_time_now.hour, date_time_now.minute, date_time_now.second)
                        print(format_str % (time_now, step, loss_value, accuracy_val_value, accuracy_test_value, examples_per_sec, sec_per_epoch))

                        summary_writer.add_summary(sess.run(summary_op, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[
                                                   0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000], learning_rate: lr}), step)
                        summary_writer.flush()
                        start_time = time.time()

                        if early_stopping_counter >= (EARLY_STOPPING_STEPS // LOG_FREQ):
                            early_stopping_counter = 0
                            decay_counter += 1
                            lr /= 5.0
                            decay_steps.append(step)
                            decay_loss.append(loss_value)
                            decay_accuracy.append(accuracy_val_value)
                            if decay_counter >= DECAY_N_TIMES:
                                final_accuracy_test = accuracy_test_value
                                break

                    # One training step.
                    sess.run(train_op, feed_dict={images_train: batch_images, labels_train: batch_labels,
                                                  images_val: val_images[0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000], learning_rate: lr})
                    step += 1

                # Save the model.
                saver.save(sess, os.path.join(MODEL_DIR, "model.ckpt"))

                return best_step_val, step, best_accuracy_val, final_accuracy_test, decay_steps, decay_loss, decay_accuracy

            # Restore a model.
            else:
                # Restore the model.
                print("Restoring the model ...")
                saver.restore(sess, os.path.join(MODEL_DIR, "model.ckpt"))

                val_images, val_labels = mnist.validation.images, mnist.validation.labels
                test_images, test_labels = mnist.test.images, mnist.test.labels
                batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)
                lr = initial_lr

                accuracy_val_value = 0.0
                accuracy_test_value = 0.0
                for i in range(10):
                    if i < 5:
                        accuracy_val_value += sess.run(accuracy_val, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                            i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000], learning_rate: lr})
                    accuracy_test_value += sess.run(accuracy_test, feed_dict={images_train: batch_images, labels_train: batch_labels, images_val: val_images[i * 1000:(
                        i + 1) * 1000], labels_val: val_labels[i * 1000:(i + 1) * 1000], images_test: test_images[i * 1000:(i + 1) * 1000], labels_test: test_labels[i * 1000:(i + 1) * 1000], learning_rate: lr})
                loss_value = sess.run(loss, feed_dict={images_train: batch_images, labels_train: batch_labels,
                                                       images_val: val_images[0:1000], labels_val: val_labels[0:1000], images_test: test_images[0:1000], labels_test: test_labels[0:1000], learning_rate: lr})
                accuracy_val_value /= 5.0
                accuracy_test_value /= 10.0

                return loss_value, accuracy_val_value, accuracy_test_value


def main(argv=None):
    training = True
    if not tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.MakeDirs(TRAIN_DIR)
    if not tf.gfile.Exists(MODEL_DIR):
        tf.gfile.MakeDirs(MODEL_DIR)
    if training:
        lr_list = [0.01]
        stddev1_list = [0.05]
        stddev2_list = [0.05]
        weight_decay_list = [0.0]
        keep_list = [0.9]
        num_units_list = [144]  # [24, 32, 48, 64, 100, 144]
        with open('training.txt', 'w', encoding="utf-8") as file:
            for lr in lr_list:
                for stddev1 in stddev1_list:
                    for stddev2 in stddev2_list:
                        for weight_decay in weight_decay_list:
                            for keep in keep_list:
                                for num_units in num_units_list:
                                    format_str = (
                                        "Parameters: initial lr = %.2f; stddev1 = %.2f; stddev2 = %.2f; weight_decay = %.1f; keep = %.2f; num units = %d - Performance: best val step = %d; best val accuracy = %.3f; final step = %d; final test accuracy = %.3f.")
                                    try:
                                        best_step_val, final_step, best_accuracy_val, final_accuracy_test, decay_steps, decay_loss, decay_accuracy = train(
                                            initial_lr=lr, stddev1=stddev1, stddev2=stddev2, weight_decay=weight_decay, keep=keep, num_units=num_units, training=training)
                                        print(format_str % (lr, stddev1, stddev2, weight_decay, keep, num_units,
                                                            best_step_val, best_accuracy_val, final_step, final_accuracy_test), file=file)
                                        # print('Decay steps:', decay_steps, file=file)
                                        # print('Decay loss:', decay_loss, file=file)
                                        # print('Decay accuracy:', decay_accuracy, file=file)
                                    except Exception as e:
                                        print(format_str % (lr, stddev1, stddev2, weight_decay, keep, num_units, -1, -1., -1, -1.), file=file)
    else:
        format_str = ('Restored model: loss = %.3f, val accuracy = %.3f, test accuracy = %.3f.')
        try:
            loss_value, accuracy_val_value, accuracy_test_value = train(0.01, 0.05, 0.05, 0.0, 0.55, 144, training)
            print(format_str % (loss_value, accuracy_val_value, accuracy_test_value))
        except Exception as e:
            print(format_str % (-1.0, -1.0, -1.0))


if __name__ == '__main__':
    tf.app.run()
