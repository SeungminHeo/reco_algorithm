import copy
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class ENC_LAYER(tf.keras.layers.Layer):
    def __init__(self, SPECS:List[dict], **kwargs):
        super(ENC_LAYER, self).__init__()
        '''
        SPECS given as follows
        layers_common = {"activation": "tanh", "regularizers": tf.keras.regularizers.l2}
        layer_1 = {**layers_common, "input_dim": 200, "units": 50}
        layer_2 = {**layers_common, "units": 50}
        SPECS = [layer_1, layer_2]
        '''
        self._specs = SPECS
        self._additional_specs = kwargs
        self._LAYERS = []

        if "dropout" in self._additional_specs.keys():
            self.dropout = tf.keras.layers.Dropout(self._additional_specs["dropout"])

        for spec in self._specs:
            self._LAYERS.append(tf.keras.layers.Dense(**spec))

    def _normalize(self, inputs, epsilon=1e-14):
        # below calculation from https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/nn_impl.py#L632-L668
        return inputs / tf.sqrt(tf.math.maximum(inputs ** 2, epsilon))

    def call(self, inputs):

        if "normalize_eps" in self._additional_specs.keys():
            outputs = self._normalize(inputs, self._additional_specs["normalize_eps"])
        else:
            outputs = inputs

        if "dropout" in self._additional_specs.keys():
            outputs = self.dropout(outputs)

        outputs = self._LAYERS[0](outputs)

        for layer in self._LAYERS[1:]:
            outputs = layer(outputs)

        last_layer_dim = self._specs[-1]["units"] // 2
        mu = outputs[:,:last_layer_dim]
        logvar = outputs[:,last_layer_dim:]

        return mu, logvar

class DEC_LAYER(ENC_LAYER):
    def __init__(self, SPECS:List[dict], **kwargs):
        super(DEC_LAYER, self).__init__(SPECS, **kwargs)

    def call(self, inputs):
        outputs = self._LAYERS[0](inputs)
        for layer in self._LAYERS[1:]:
            outputs = layer(outputs)
        return outputs

class VAE_STRT(tf.keras.Model):
    '''
    Model VAE structure
    
    Reference Paper: 
    Variational Auto-encoders for Collaborative Filtering
    https://arxiv.org/pdf/1802.05814.pdf
    '''
    def __init__(self,
                 enc_specs:List[dict],
                 additional_specs:dict,
                 vae_activation:str=None):
        super(VAE_STRT, self).__init__()

        self._enc_specs = enc_specs
        self._dec_specs = copy.deepcopy(enc_specs)[::-1]

        # last layer for mu and std
        self._enc_specs[-1]["units"] = self._enc_specs[-1]["units"] * 2

        # map encoder specs for decoder (reverse)
        self._dec_specs[0]["units"] = self._dec_specs[-1]["input_dim"]
        self._dec_specs[-1].pop("input_dim")
        self._dec_specs.append(self._dec_specs.pop(0))
        self._dec_specs[0]["input_dim"] = self._enc_specs[-1]["units"] // 2

        if vae_activation:
            self._dec_specs[-1]["activation"] = vae_activation

        self._ENCODER = ENC_LAYER(self._enc_specs, **additional_specs)

        if "dropout" in additional_specs.keys():
            additional_specs.pop("dropout")
        self._DECODER = DEC_LAYER(self._dec_specs, **additional_specs)

    @tf.function
    def _reparameterize(self, mu, logvar):
        std = tf.exp(logvar * 0.5)
        eps = tf.random.normal(std.shape)
        return tf.math.add(mu, tf.math.multiply(eps, std))

    def call(self, inputs):
        mu, logvar = self._ENCODER(inputs)
        z = self._reparameterize(mu, logvar)
        return self._DECODER(z), mu, logvar

class VAE:
    '''
    Requires all inputs to be float32
    '''
    def __init__(self,
                 optimizer,
                 enc_specs:List[dict],
                 additional_specs:dict,
                 model_config,
                 logger,
                 model=VAE_STRT):

        self.model_config = model_config
        self.logger = logger

        # model is initialized at self.set_data
        self.model = model
        self._enc_specs = enc_specs
        self._additional_specs = additional_specs
        self._optimizer = optimizer(learning_rate=self.model_config.lr, weight_decay=self.model_config.weight_decay)

        # initialize supporting functions
        self._BCE = tf.keras.losses.BinaryCrossentropy()
        self._mean_loss = tf.keras.metrics.Mean(name="train_loss")
        self._mean_MSE = tf.keras.metrics.MeanSquaredError(name="validation_mse")

        # anneal adjustment
        self._anneal = 1

    def _generator(self):
        return tf.data.Dataset.from_tensor_slices(self.data).shuffle(self._user_size).batch(self.model_config.batch_size)

    def _loss_function(self, output, y, mu, log_var):
        '''
        loss = CrossEntropy(output, y) - (anneal) * (KL_div)
        ! add regularization
        ! add adjustable anneal
        '''
        # cross entropy loss
        bce = self._BCE(y, output)

        # KL Divergence
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var), axis=1))

        return bce + self._anneal * kl

    def _save_checkpoint(self):
        # ! add checkpointing
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

        # ! train losses for loss function annealing and external plotting
        self._train_losses = []
        self._train_mse = []

        self.logger.info("Start Training")

        for epoch in range(self.model_config.epochs):
            self.logger.info(f"Epoch {epoch}")

            self._mean_loss.reset_states()
            self._mean_MSE.reset_states()

            for batch_e, batch in enumerate(self._generator()):

                # compute gradient
                output, loss, grads = self._grad(batch, batch)

                # backprop
                self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # logging training info
                if not batch_e % self.model_config.logging_info_freq:
                    self.logger.info(f"Batch {batch_e}")
                    self._mean_loss(loss)
                    mean_loss = self._mean_loss.result().numpy()
                    self._mean_MSE(batch, output)
                    mean_MSE = self._mean_MSE.result().numpy()
                    self._train_losses.append(mean_loss)
                    self._train_mse.append(mean_MSE)

                    self.logger.info("Loss : {:0.6f} \t | Mean Loss: {:0.6f}".format(loss, mean_loss))
                    self.logger.info(f"Train MSE : {mean_MSE}")

        self.logger.info("Finish Training")
