import numpy as np
import deepautoencoder.utils as utils
import tensorflow as tf
import os

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']
allowed_optimizers = ['gd','adam']


class StackedAutoEncoder:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses, allowed_optimizers
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert self.optimizer in allowed_optimizers, 'Incorrect optimizer given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, batch_size=100, print_step=50, optimizer='adam'):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.optimizer = optimizer
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []
        self.loss_history, self.accuracy_history = self.list_init(),self.list_init()
        self.model_path = os.getcwd()

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(
                    frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        '''

        :param x: m x p dataframe
        :return: trained weights and bias for the sdae
        '''
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                x = self.run(data_x=x, activation=self.activations[i],
                             data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[
                                 i], loss=self.loss,
                             batch_size=self.batch_size, lr=self.lr,
                             print_step=self.print_step,depth=i)
            else:
                temp = np.copy(x)
                x = self.run(data_x=self.add_noise(temp),
                             activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i],
                             epoch=self.epoch[
                                 i], loss=self.loss,
                             batch_size=self.batch_size,
                             lr=self.lr, print_step=self.print_step,depth=i)
            print('Layer {0}'.format(i + 1) + ' Weight Dimension: ' + str(self.weights[i].shape))

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run(self, data_x,depth, data_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):

        tf.reset_default_graph()

        #Variables and parameters for working with tensors
        input_dim = len(data_x[0])
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[
                            None, input_dim], name='x_')

        #Define the encoding/decoding weights and bias for the autoencoder layer
        encode = {'weights': tf.Variable(tf.truncated_normal(
            [input_dim, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim],dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim],dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}

        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        #Define the loss function and optimizer
        loss = self.cost(x_, decoded, loss)
        train_op = self.optimizers(lr, loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                loss_ = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                self.loss_history[depth].append(loss_)
                print('epoch {0}: global loss = {1}'.format(i, loss_))
                print(self.loss_history)
        self.loss_val = l
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})

    def cost(self,x_,decoded ,loss):
        if loss == 'rmse':
            return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            return -tf.reduce_mean(x_ * tf.log(decoded))

    def optimizers(self, lr, loss):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(lr).minimize(loss)
        else:
            return tf.train.GradientDescentOptimizer(lr).minimize(loss)

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

    def list_init(self):
        list = []
        for i in range(self.depth):
            list.append([])
        return list
