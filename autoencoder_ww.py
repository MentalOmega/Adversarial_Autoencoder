import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


class AE:
    def __init__(self, zdim):
        self.zdim = zdim
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()

    def cycle(self, x_input):
        latent_code = self.encoder_model(x_input)
        x_output = self.decoder_model(latent_code)
        return x_output

    def predict(self, decoder_input):
        x_output = self.decoder_model(decoder_input)
        return x_output

    def encoder(self):
        with tf.name_scope("encoder"):
            input = keras.layers.Input([784])
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(input)
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(fc)
            output = keras.layers.Dense(self.zdim)(fc)
            return keras.models.Model(input, output)

    def decoder(self):
        with tf.name_scope("decoder"):
            input = keras.layers.Input([self.zdim])
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(input)
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(fc)
            output = keras.layers.Dense(784, activation=tf.nn.sigmoid)(fc)
            return keras.models.Model(input, output)


zdim = 2
batch_size = 64
input_dim = 784
n_epochs = 100
from tensorflow.examples.tutorials.mnist import input_data


class dataset:
    def __init__(self, data):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (data)).repeat().batch(batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


import time


def train(train_model):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images, train_labels = train_images/255.0,train_labels/255.0
    test_images,test_labels = test_images/255.0,test_labels/255.0
    train_images_dataset = dataset(train_images)

    x_input = tf.placeholder(dtype=tf.float32, shape=[
                             batch_size, input_dim], name='Input')
    x_target = tf.placeholder(dtype=tf.float32, shape=[
                              batch_size, input_dim], name='Target')
    decoder_input = tf.placeholder(dtype=tf.float32, shape=[
                                   1, zdim], name='Decoder_input')

    autoencoder = AE(zdim)
    x_output = autoencoder.cycle(x_input)
    decoder_output = autoencoder.predict(decoder_input)

    input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    generated_images = tf.reshape(x_output, [-1, 28, 28, 1])
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images',
                     tensor=generated_images, max_outputs=10)

    cycle_loss = tf.reduce_mean(tf.square(x_target - x_output))
    tf.summary.scalar("loss", cycle_loss)
    summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer().minimize(cycle_loss)
    init = tf.global_variables_initializer()
    global_step = 0
    # Saving the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(train_images_dataset.iterator.initializer)
        logdir = r"./Results/Autoencoder/{}".format(
            time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime()))
        print(logdir)
        writer = tf.summary.FileWriter(logdir=logdir+"/Tensorboard")
        for v in tf.trainable_variables():
            print(v)
        if train_model:
            for epoch in range(n_epochs):
                n_batches = int(len(train_images) / batch_size)
                for _ in range(n_batches):
                    batch = sess.run(train_images_dataset.next_element)
                    batch = batch.reshape([-1, 28*28])
                    sess.run(optimizer, feed_dict={
                             x_input: batch, x_target: batch})
                    if _ % 50 == 0:
                        summary, batch_loss = sess.run([summary_op, cycle_loss], feed_dict={
                            x_input: batch, x_target: batch})
                        writer.add_summary(summary, global_step=global_step)
                        print("Loss: {}".format(batch_loss))
                        print("Epoch: {}, iteration: {}".format(epoch, _))
                    global_step += 1
                saver.save(sess, save_path=logdir+"/Saved_models",
                           global_step=global_step)
        else:
            pass


if __name__ == '__main__':
    train(train_model=True)