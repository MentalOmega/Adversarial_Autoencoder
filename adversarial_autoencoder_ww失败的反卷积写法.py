import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from tensorflow import keras


z_dim = 2
batch_size = 64
image_w = 28
image_h = 28
image_d = 1
input_dim = 784
n_epochs = 100
learning_rate = 0.001
beta1 = 0.9
results_path = './Results/Adversarial_Autoencoder'


class Dataset:
    '''
    处理数据，获取迭代器
    '''

    def __init__(self, data):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (data)).repeat().batch(batch_size).shuffle(buffer_size=128)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


class Autoencoder:

    initializer = tf.random_normal_initializer(0., 0.02)

    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.encoder_model = self.encoder()
        self.encoder_model.summary()
        self.decoder_model = self.decoder()
        self.decoder_model.summary()

    def cycle(self, x_input):
        x_input = tf.reshape(x_input, [-1, image_w,image_h,image_d])
        latent_code = self.encoder_model(x_input)
        self.latent_code = latent_code
        print(latent_code)
        x_output = self.decoder_model(self.latent_code)
        x_output = tf.image.resize_images(x_output, [28, 28])
        return x_output

    def predict(self, decoder_input):
        x_output = self.decoder_model(decoder_input)
        return x_output

    def encoder(self):
        with tf.name_scope("encoder"):
            input = keras.layers.Input([image_w, image_h, image_d])
            con = keras.layers.Conv2D(
                64, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(input)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2D(
                128, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)
            
            con = keras.layers.Conv2D(
                256, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2D(
                512, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2D(
                2, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)

            output = con
            # flatten = keras.layers.GlobalAveragePooling2D()(con)
            # output = keras.layers.Dense(self.z_dim)(flatten)
            return keras.models.Model(input, output)

    def decoder(self):
        with tf.name_scope("decoder"):
            input = keras.layers.Input([1,1,self.z_dim])
            con = keras.layers.Conv2DTranspose(
                64, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(input)  # 2*2*64
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2DTranspose(
                128, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)  # 4*4*128
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2DTranspose(
                256, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)  # 8*8*256
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            con = keras.layers.Conv2DTranspose(
                512, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con)  # 16*16*512
            con = keras.layers.BatchNormalization()(con)
            con = keras.layers.LeakyReLU()(con)

            output = keras.layers.Conv2DTranspose(
                1, [4, 4], strides=2, padding='same', kernel_initializer=self.initializer)(con) # 32*32*1
            return keras.models.Model(input, output)


# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_w,image_h], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[
                          batch_size, image_w,image_h], name='Target')
real_distribution = tf.placeholder(
    dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[
                               1, z_dim], name='Decoder_input')


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

class Discriminator:
    def __init__(self,z_dim):
        self.z_dim=z_dim
        self.model = self.build()
        self.model.summary()

    def __call__(self,input):
        input = tf.squeeze(input)
        return self.model(input)

    def build(self):
        input = keras.layers.Input([self.z_dim])
        fc = keras.layers.Dense(1000,activation="relu")(input)
        fc = keras.layers.Dense(1000,activation="relu")(fc)
        output = keras.layers.Dense(1)(fc)
        return keras.models.Model(input, output)


def train(train_model=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images/255.0
    test_images = test_images/255.0
    train_images_dataset = Dataset(train_images)

    autoencoder = Autoencoder(z_dim)
    discriminator = Discriminator(z_dim)
    x_output = autoencoder.cycle(x_input)
    decoder_output = autoencoder.predict(decoder_input)

    tf.summary.image(name='Input Images', tensor=x_input, max_outputs=10)
    tf.summary.image(name='Generated Images',
                     tensor=x_output, max_outputs=10)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Discrimminator Loss
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
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=discriminator.model.variables)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=autoencoder.encoder_model.variables)

    init = tf.global_variables_initializer()

    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.image(name='Input Images', tensor=x_input, max_outputs=10)
    tf.summary.image(name='Generated Images',tensor=x_output, max_outputs=10)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    global_step = 0
    with tf.Session() as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            sess.run(init)
            sess.run(train_images_dataset.iterator.initializer)
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
                            [autoencoder_loss, dc_loss, generator_loss, summary_op],
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
            pass 
            # # Get the latest results folder
            # all_results = os.listdir(results_path)
            # all_results.sort()
            # saver.restore(sess, save_path=tf.train.latest_checkpoint(
            #     results_path + '/' + all_results[-1] + '/Saved_models/'))
            # generate_image_grid(sess, op=decoder_image)


if __name__ == '__main__':
    train(train_model=True)
