import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data


class AE:
    def __init__(self, zdim):
        self.zdim = zdim
        self.encoder_model = self.encoder()
        self.decoder_model = self.decoder()

    def cycle(self, x_input):
        x_input = tf.reshape(x_input, [-1, 28*28])
        latent_code = self.encoder_model(x_input)
        self.latent_code = latent_code
        x_output = self.decoder_model(latent_code)
        x_output = tf.reshape(x_output, [-1, 28, 28])
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
image_w = 28
image_h = 28
input_dim = 784
n_epochs = 100


class dataset:
    def __init__(self, data):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (data)).repeat().batch(batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


x_input = tf.placeholder(dtype=tf.float32, shape=[
    batch_size, image_w, image_h], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[
    batch_size, image_w, image_h], name='Target')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[
    1, zdim], name='Decoder_input')


def train(train_model):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images/255.0
    test_images = test_images/255.0
    train_images_dataset = dataset(train_images)

    autoencoder = AE(zdim)
    x_output = autoencoder.cycle(x_input)
    decoder_output = autoencoder.predict(decoder_input)

    input_images = tf.reshape(x_input, [-1, image_h, image_w, 1])
    generated_images = tf.reshape(x_output, [-1, image_h, image_w, 1])
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images',tensor=generated_images, max_outputs=10)


    cycle_loss = tf.reduce_mean(tf.square(x_target - x_output))
    tf.summary.scalar("loss", cycle_loss)
    summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer().minimize(cycle_loss)
    init = tf.global_variables_initializer()
    global_step = 0
    # Saving the model
    saver = tf.train.Saver()
    results_path = './Results/Autoencoder/'
    with tf.Session() as sess:
        sess.run(init)
        for v in tf.trainable_variables():
            print(v)
        print(autoencoder.decoder_model.variables)
        exit()
        if train_model:
            sess.run(train_images_dataset.iterator.initializer)
            logdir = results_path + \
                time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime())
            print(logdir)
            writer = tf.summary.FileWriter(logdir=logdir+"/Tensorboard/")
            for epoch in range(n_epochs):
                n_batches = int(len(train_images) / batch_size)
                for _ in range(n_batches):
                    batch = sess.run(train_images_dataset.next_element)
                    # batch = batch.reshape([-1, 28*28])
                    sess.run(optimizer, feed_dict={
                             x_input: batch, x_target: batch})
                    if _ % 50 == 0:
                        summary, batch_loss = sess.run([summary_op, cycle_loss], feed_dict={
                            x_input: batch, x_target: batch})
                        writer.add_summary(summary, global_step=global_step)
                        print("Loss: {}".format(batch_loss))
                        print("Epoch: {}, iteration: {}".format(epoch, _))
                    global_step += 1
                saver.save(sess, save_path=logdir+"/Saved_models/",
                           global_step=global_step)
        else:
            all_results = os.listdir(results_path)
            all_results.sort()
            print(all_results)
            plt.figure(figsize=(6, 6))
            n_batches = int(len(train_images) / batch_size)
            for i in range(n_batches):
                latent_code = sess.run(autoencoder.latent_code, feed_dict={
                    x_input: train_images[batch_size*i:batch_size*(i+1)]})
                plt.scatter(latent_code[:, 0],
                            latent_code[:, 1], c=train_labels[batch_size*i:batch_size*(i+1)])
            plt.colorbar()
            plt.show()

            saver.restore(sess, save_path=tf.train.latest_checkpoint(
                results_path + '/' + all_results[-1] + '/Saved_models/'))
            generate_image_grid(sess, op=decoder_output)


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    n = 10
    x_points = np.linspace(-1, 1, n)
    y_points = np.linspace(-1, 1, n)

    nx, ny = len(x_points), len(y_points)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        z = np.reshape(z, (1, 2))
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()


if __name__ == '__main__':
    train(train_model=False)
