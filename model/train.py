import copy
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class Model:
    '''
    Requires all inputs to be float32
    '''

    def __init__(self,
                 optimizer,
                 enc_specs,
                 additional_specs,
                 config,
                 logger,
                 model=VAE):

        self.config = config
        self.logger = logger

        # model is initialized at self.set_data
        self.model = model
        self._enc_specs = enc_specs
        self._additional_specs = additional_specs
        self._optimizer = optimizer(learning_rate=self.config.lr, weight_decay=self.config.weight_decay)

        # initialize supporting functions
        self._BCE = tf.keras.losses.BinaryCrossentropy()
        self._mean_loss = tf.keras.metrics.Mean(name="train_loss")
        self._mean_MSE = tf.keras.metrics.MeanSquaredError(name="validation_mse")

        # anneal adjustment
        self._anneal = 1

    def _generator(self):
        return tf.data.Dataset.from_tensor_slices(self.data).shuffle(self._user_size).batch(self.config.batch_size)

    def _loss_function(self, output, y, mu, log_var):
        '''
        loss = CrossEntropy(output, y) - (anneal) * (KL_div)
        ! regularization needs to be added
        '''
        # cross entropy loss
        bce = self._BCE(y, output)

        # KL Divergence
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var), axis=1))

        return bce + self._anneal * kl

    def _save_checkpoint(self):
        # ! save checkpoint
        return 0

    @tf.function
    def _grad(self, X, y):
        with tf.GradientTape() as tape:
            output, mu, logvar = self.model(X)
            loss = self._loss_function(output, y, mu, logvar)
        grad = tape.gradient(loss, self.model.trainable_variables)
        return output, loss, grad

    def set_data(self, R:np.array):  # R : array(user - item)
        '''
        Requires data to be in float32 type
        '''
        self.data = tf.convert_to_tensor(R)
        self._user_size = R.shape[0]  # buffer_size = number of users
        self._item_size = R.shape[1]
        self._enc_specs[0]["input_dim"] = self._item_size
        self.model = self.model(self._enc_specs, self._additional_specs)
        self.model.build(R.shape)

    def train(self):

        self._train_losses = []
        self._train_mse = []

        self.logger.info("Start Training")

        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch}")

            self._mean_loss.reset_states()
            self._mean_MSE.reset_states()

            for batch_e, batch in enumerate(self._generator()):

                # compute gradient
                output, loss, grads = self._grad(batch, batch)

                # backprop
                self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # logging training info
                if not batch_e % self.config.logging_info_freq:
                    self.logger.info(f"Batch {batch_e}")
                    self._mean_loss(loss)
                    mean_loss = self._mean_loss.result().numpy()
                    self._mean_MSE(batch, output)
                    mean_MSE = self._mean_MSE.result().numpy()
                    self._train_losses.append(mean_loss)
                    self._train_mse.append(mean_MSE)

                    self.logger.info("Loss : {:0.6f} \t | Mean Loss: {:0.6f}".format(loss, mean_loss))
                    self.logger.info(f"Train MSE : {mean_MSE}")
