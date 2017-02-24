"""
implement DCGAN with Tensorflow for MINST.
"""
import os
import tensorflow as tf


class Dcgan(object):
    @staticmethod
    def leaky_relu(x, leak=0.2, name="lrelu"):
        """
        https://github.com/tensorflow/tensorflow/issues/4079
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1.0 + leak)
            f2 = 0.5 * (1.0 - leak)
            return f1 * x + f2 * abs(x)

    @staticmethod
    def discriminator(source, reuse):
        """
        build the discriminator network.
        param 'source' is the input data in shape [-1, 64, 64, 3].
        param 'resue' is for sharing the network with generator.
        """
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        # # arXiv:1511.06434v2
        # build convolutional net to downsample.
        # no pooling layers.
        for layer_idx in xrange(4):
            # arXiv:1511.06434v2
            # in discriminator, use batch norm except input layer
            if layer_idx == 0:
                normalizer_fn = None
            else:
                normalizer_fn = tf.contrib.layers.batch_norm

            # arXiv:1511.06434v2
            # in discriminator, use LeakyReLU
            source = tf.contrib.layers.convolution2d(
                inputs=source,
                num_outputs=2 ** (7 + layer_idx),
                kernel_size=4,
                stride=2,
                padding='SAME',
                activation_fn=Dcgan.leaky_relu,
                normalizer_fn=normalizer_fn,
                weights_initializer=weights_initializer,
                scope='d_conv_{}'.format(layer_idx),
                reuse=reuse)

        # for fully connected layer
        source = tf.contrib.layers.flatten(source)

        # fully connected layer to binary classify
        source = tf.contrib.layers.fully_connected(
            inputs=source,
            num_outputs=1,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=weights_initializer,
            scope='d_out',
            reuse=reuse)

        return source

    @staticmethod
    def generator(seed):
        """
        build the generator network.
        """
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        # fully connected layer to upscale the seed for the input of
        # convolutional net.
        target = tf.contrib.layers.fully_connected(
            inputs=seed,
            num_outputs=4 * 4 * 1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='g_project')

        # reshape to images
        target = tf.reshape(target, [-1, 4, 4, 1024])

        # transpose convolution to upscale
        for layer_idx in xrange(4):
            if layer_idx == 3:
                num_outputs = 3

                # arXiv:1511.06434v2
                # use tanh in output layer
                activation_fn = tf.nn.tanh

                # arXiv:1511.06434v2
                # use batch norm except the output layer
                normalizer_fn = None
            else:
                num_outputs = 2 ** (9 - layer_idx)

                # arXiv:1511.06434v2
                # use ReLU
                activation_fn = tf.nn.relu

                # arXiv:1511.06434v2
                # use batch norm
                normalizer_fn = tf.contrib.layers.batch_norm

            target = tf.contrib.layers.convolution2d_transpose(
                inputs=target,
                num_outputs=num_outputs,
                kernel_size=5,
                stride=2,
                padding='SAME',
                activation_fn=activation_fn,
                normalizer_fn=normalizer_fn,
                weights_initializer=weights_initializer,
                scope='g_conv_t_{}'.format(layer_idx))

        return target

    def __init__(self, params):
        """
        build DCGAN in tensorflow.
        """
        generator_seed_size = params['generator_seed_size']

        # global step
        self._global_step = tf.get_variable(
            "gstep",
            [],
            trainable=False,
            initializer=tf.constant_initializer(0.0))

        # the input batch placeholder for the generator.
        self._seed = tf.placeholder(
            shape=[None, generator_seed_size], dtype=tf.float32)

        # the input batch placeholder (real data) for the discriminator.
        self._real = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)

        # build the generator to generate images from random seeds.
        self._generate_fake = Dcgan.generator(self._seed)

        # build the discriminator to judge the real data.
        discriminate_real = Dcgan.discriminator(self._real, False)

        # build the discriminator to judge the fake data.
        # judge both real and fake data with the same network (shared).
        discriminate_fake = Dcgan.discriminator(self._generate_fake, True)

        self._loss_discriminator = -tf.reduce_mean(
            tf.log(discriminate_real) + tf.log(1.0 - discriminate_fake))
        self._loss_generator = -tf.reduce_mean(tf.log(discriminate_fake))

        trainable_variables = tf.trainable_variables()

        # peek the trainable variables.
        # trainable_variables[:10] are for generator.
        # trainable_variables[10:] are for discriminator.
        # building 2 groups base on their names should be safer.
        for idx, variable in enumerate(trainable_variables):
            print idx, variable.name

        trainer_generator = tf.train.AdamOptimizer(
            learning_rate=0.0002, beta1=0.5)
        trainer_discriminator = tf.train.AdamOptimizer(
            learning_rate=0.0002, beta1=0.5)

        # minimize the loss to train the generator and discriminator.
        gradients_generator = trainer_generator.compute_gradients(
            self._loss_generator, trainable_variables[:10])
        gradients_discriminator = trainer_discriminator.compute_gradients(
            self._loss_discriminator, trainable_variables[10:])

        self._train_generator = trainer_generator.apply_gradients(
            gradients_generator, global_step=self._global_step)
        self._train_discriminator = trainer_discriminator.apply_gradients(
            gradients_discriminator)

        self._session = tf.Session()

        # restore check point
        if not os.path.isdir('./checkpoints/'):
            os.makedirs('./checkpoints/')

        self._checkpoint_source_path = \
            tf.train.latest_checkpoint('./checkpoints/')
        self._checkpoint_target_path = './checkpoints/model.ckpt'

        # dummy variable to set range of image summaries
        dummy = tf.get_variable(
            'dummy',
            [1, 64, 64, 3],
            initializer=tf.constant_initializer([1.0, -1.0]))

        if self._checkpoint_source_path is not None:
            saver = tf.train.Saver()
            saver.restore(self._session, self._checkpoint_source_path)
        else:
            self._session.run(tf.global_variables_initializer())

        self._summary_g = \
            tf.summary.scalar('generator loss', self._loss_generator)
        self._summary_d = \
            tf.summary.scalar('discriminator loss', self._loss_discriminator)

        # concat the dummy variable to force image summary to map pixel values
        # from [-1.0, +1.0] to [0, 255]
        temp = tf.concat(0, [self._generate_fake, dummy])

        self._summary_image = tf.summary.image(
            'generated image', temp, max_outputs=18)

        self._reporter = tf.summary.FileWriter(
            './tensorboard/', self._session.graph)

        # give up overlapped old data
        g_step = self._session.run(self._global_step)

        self._reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=g_step)

    def save_checkpoint(self):
        """
        """
        saver = tf.train.Saver()

        saver.save(self._session, self._checkpoint_target_path,
                   global_step=self._global_step)

    def train_discriminator(self, seed_sources, real_sources):
        """
        train the discriminator network with batch data.
        the generator use seed_sources to generate fake_sources for the
        discriminator.
        """
        fetch = [
            self._train_discriminator,
            self._loss_discriminator,
            self._global_step,
            self._summary_d]

        feeds = {self._seed: seed_sources, self._real: real_sources}

        _, loss, g_step, summary = self._session.run(fetch, feed_dict=feeds)

        self._reporter.add_summary(summary, g_step)

        return loss

    def train_generator(self, seed_sources):
        """
        train the generator network with batch data.
        """
        fetch = [
            self._train_generator,
            self._loss_generator,
            self._global_step,
            self._summary_g]

        feeds = {self._seed: seed_sources}

        _, loss, g_step, summary = self._session.run(fetch, feed_dict=feeds)

        self._reporter.add_summary(summary, g_step)

        return loss

    def generate(self, seed_sources):
        """
        generate fake images from the seeds.
        """
        fetch = [self._generate_fake, self._global_step, self._summary_image]
        feeds = {self._seed: seed_sources}

        fake, g_step, summary = self._session.run(fetch, feed_dict=feeds)

        self._reporter.add_summary(summary, g_step)

        return fake
