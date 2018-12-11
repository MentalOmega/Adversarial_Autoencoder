import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from setting import *
results_path = './Results/Adversarial_Autoencoder'

class Autoencoder:
    def __init__(self):
        print("初始化自编码器")
        self.encoder_model = self.encoder()
        self.encoder_model.summary()
        self.decoder_model = self.decoder()
        self.decoder_model.summary()

    def cycle(self, x_input):
        x_input = tf.reshape(x_input, [-1, image_w*image_h])
        latent_code = self.encoder_model(x_input)
        self.latent_code = latent_code
        x_output = self.decoder_model(latent_code)
        x_output = tf.reshape(x_output, [-1, image_w, image_h])
        return x_output

    def predict(self, decoder_input):
        x_output = self.decoder_model(decoder_input)
        return x_output

    def encoder(self):
        with tf.name_scope("encoder"):
            input = keras.layers.Input([image_w*image_h])
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(input)
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(fc)
            output = keras.layers.Dense(z_dim)(fc)
            return keras.models.Model(input, output)

    def decoder(self):
        with tf.name_scope("decoder"):
            input = keras.layers.Input([z_dim])
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(input)
            fc = keras.layers.Dense(1000, activation=tf.nn.relu)(fc)
            output = keras.layers.Dense(image_w*image_h, activation=tf.nn.sigmoid)(fc)
            return keras.models.Model(input, output)



class Discriminator:
    def __init__(self):
        print("初始化判别器")
        self.model = self.build()
        self.model.summary()

    def __call__(self,input):
        return self.model(input)

    def build(self):
        input = keras.layers.Input([z_dim])
        fc = keras.layers.Dense(1000,activation="relu")(input)
        fc = keras.layers.Dense(1000,activation="relu")(fc)
        output = keras.layers.Dense(1)(fc)
        return keras.models.Model(input, output)

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
    1, z_dim], name='Decoder_input')
real_distribution = tf.placeholder(
    dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_Adversarial_Autoencoder". \
        format(time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime()), z_dim,
               learning_rate, batch_size, n_epochs, beta1)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

def train(train_model):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images/255.0
    test_images = test_images/255.0
    train_images_dataset = dataset(train_images)

    autoencoder = Autoencoder()
    discriminator = Discriminator()
    x_output = autoencoder.cycle(x_input)
    decoder_output = autoencoder.predict(decoder_input)

    input_images = tf.reshape(x_input, [-1, image_h, image_w, 1])
    generated_images = tf.reshape(x_output, [-1, image_h, image_w, 1])
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images',tensor=generated_images, max_outputs=10)

    cycle_loss = tf.reduce_mean(tf.square(x_target - x_output))
    d_real = discriminator(real_distribution)
    d_fake = discriminator(autoencoder.latent_code)
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(cycle_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=discriminator.model.variables)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=autoencoder.encoder_model.variables)

    init = tf.global_variables_initializer()

    # Tensorboard visualization
    tf.summary.scalar(name='Cycle Loss', tensor=cycle_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    summary_op = tf.summary.merge_all()
    global_step = 0
    # Saving the model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        if train_model:
            sess.run(train_images_dataset.iterator.initializer)
            tensorboard_path, saved_model_path, log_path = form_results()   
            writer = tf.summary.FileWriter(logdir=tensorboard_path)
            for epoch in range(n_epochs):
                n_batches = int(len(train_images) / batch_size)
                for _ in range(n_batches):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                    batch = sess.run(train_images_dataset.next_element)
                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch, x_target: batch})
                    sess.run(discriminator_optimizer,feed_dict={x_input: batch, x_target: batch, real_distribution: z_real_dist})
                    sess.run(generator_optimizer, feed_dict={x_input: batch, x_target: batch})
                    if _ % 50 == 0:
                        a_loss, d_loss, g_loss, summary = sess.run(
                            [cycle_loss, dc_loss, generator_loss, summary_op],
                            feed_dict={x_input: batch, x_target: batch,
                                       real_distribution: z_real_dist})
                        writer.add_summary(summary, global_step=global_step)
                        print("Epoch: {}, iteration: {}".format(epoch, _))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Loss: {}".format(d_loss))
                        print("Generator Loss: {}".format(g_loss))
                    global_step += 1
                saver.save(sess, save_path=saved_model_path, global_step=global_step)
        else:
            all_results = os.listdir(results_path)
            all_results.sort()
            print(all_results)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(
                results_path + '/' + all_results[-1] + '/Saved_models/'))

            generate_image_grid(sess, op=decoder_output)

            plt.figure(figsize=(6, 6))
            n_batches = int(len(train_images) / batch_size)
            for i in range(n_batches//10):
                latent_code = sess.run(autoencoder.latent_code, feed_dict={
                    x_input: train_images[batch_size*i:batch_size*(i+1)]})
                plt.scatter(latent_code[:, 0],
                            latent_code[:, 1], c=train_labels[batch_size*i:batch_size*(i+1)])
            plt.colorbar()
            plt.show()

            



def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    n = 10
    x_points = np.linspace(-20, 20, n)
    y_points = np.linspace(-20, 20, n)

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
