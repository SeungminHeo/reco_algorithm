import sys
import copy
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
sys.path.insert(1, '../evaluate')
from evaluate import Evaluate

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
            # outputs = self._normalize(inputs, self._additional_specs["normalize_eps"])
            outputs = tf.nn.l2_normalize(inputs, 1)
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

        if vae_activation is not None:
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
                 vae_activation=None,
                 anneal=1,
                 model=VAE_STRT):

        self.model_config = model_config
        self.logger = logger

        # model is initialized at self.set_data
        self.model = model
        self._enc_specs = enc_specs
        print("VAE ENC SPEC: ", enc_specs)
        self._additional_specs = additional_specs
        # self._optimizer = optimizer(learning_rate=self.model_config.lr, weight_decay=self.model_config.weight_decay)
        self._optimizer = optimizer(learning_rate=self.model_config.lr)

        self._eval = Evaluate(logger)

        # initialize supporting functions
        # self._BCE = tf.keras.losses.BinaryCrossentropy()
        self._mean_loss = tf.keras.metrics.Mean(name="train_loss")
        self._mean_MSE = tf.keras.metrics.MeanSquaredError(name="validation_mse")

        # anneal adjustment
        self._anneal = anneal

        self.vae_activation = vae_activation


    def _generator(self):
        batch_size = self.model_config.batch_size
        shuffle_idx = random.sample(list(range(0, self._user_size-1)), self._user_size-1)
        steps = self._user_size // batch_size + 1
        for i in range(steps):
            if i*batch_size+batch_size >= self._user_size:
                yield tf.convert_to_tensor(self.data[shuffle_idx[i*batch_size:]].toarray())
            else:
                yield tf.convert_to_tensor(self.data[shuffle_idx[i*batch_size:i*batch_size+batch_size]].toarray())
    
    def _valid_generator(self):
        # ! merge with _generator function
        input_data = self.valid_data_input
        label_data = self.valid_data_label
        batch_size = self.model_config.batch_size
        shuffle_idx = random.sample(list(range(0, input_data.shape[0]-1)), input_data.shape[0]-1)
        self.shuffle_idx = shuffle_idx
        steps = input_data.shape[0] // batch_size + 1
        for i in range(steps):
            if i*batch_size+batch_size > input_data.shape[0]:
                yield tf.convert_to_tensor(input_data[shuffle_idx[i*batch_size:]].toarray()), \
                      tf.convert_to_tensor(label_data[shuffle_idx[i*batch_size:]].toarray())
            elif i*batch_size+batch_size < input_data.shape[0]:
                yield tf.convert_to_tensor(input_data[shuffle_idx[i*batch_size:i*batch_size+batch_size]].toarray()), \
                      tf.convert_to_tensor(label_data[shuffle_idx[i*batch_size:i*batch_size+batch_size]].toarray())

    def _loss_function(self, output, y, mu, log_var, anneal=None):
        '''
        loss = CrossEntropy(output, y) - (anneal) * (KL_div)
        ! add adjustable anneal
        '''
        if anneal:
            self._anneal = anneal

        output_log_softmax = tf.nn.log_softmax(output)

        neg_ll = -tf.math.reduce_mean(tf.math.reduce_sum(output_log_softmax * y, axis=-1))
        # neg_ll = -tf.math.reduce_mean(tf.math.reduce_mean(output_log_softmax * y, axis=-1))

        # KL Divergence
        kl = -0.5* tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var), axis=1))

        return neg_ll + self._anneal * kl

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

    def set_data(self, R:sparse.csr_matrix, 
                       user_size:int,
                       item_size:int,
                       valid_data_input:sparse.csr_matrix=None,
                       valid_data_label:sparse.csr_matrix=None):  # R : array(user - item)
        '''
        Requires data to be in float32 type
        '''
        if valid_data_input is not None:
            assert valid_data_label is not None, "Must input labels for valid data"
            self.valid_data_input = valid_data_input
            self.valid_data_label = valid_data_label
        self.data = R
        self._user_size = user_size  # buffer_size = number of users
        self._item_size = item_size
        self._enc_specs[0]["input_dim"] = self._item_size
        self.model = self.model(self._enc_specs, self._additional_specs, self.vae_activation)
        # self.model.build((user_size, item_size))

    def train(self, validation=False):

        # ! train losses for loss function annealing and external plotting
        self._train_losses = []
        self._train_mse = []
        self._validation_scores = []

        self.logger.info("Start Training")

        for epoch in range(self.model_config.epochs):
            self.logger.info(f"Epoch {epoch}")
            self._mean_loss.reset_states()
            self._mean_MSE.reset_states()

            for e, batch in enumerate(self._generator()):
                output, loss, grads = self._grad(batch, batch)
                
                # backprop
                self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # logging training info
                if not e % self.model_config.logging_info_freq:

                    self.logger.info(f"Batch {e}")
                    self._mean_loss(loss)
                    mean_loss = self._mean_loss.result().numpy()
                    self._mean_MSE(batch, output)
                    mean_MSE = self._mean_MSE.result().numpy()
                    self._train_losses.append(mean_loss)
                    self._train_mse.append(mean_MSE)

                    self.logger.info("Loss : {:0.6f} \t | Mean Loss: {:0.6f}".format(loss, mean_loss))
                    self.logger.info(f"Train MSE : {mean_MSE}")
            
            if validation:
                validation_score = 0
                for valid_e, (valid_batch_input, valid_batch_label) in enumerate(self._valid_generator()):
                    output = self._recommend(valid_batch_input)
                    validation_score += self.validate(valid_batch_label, output)
                validation_score = validation_score / (valid_e+1)
                self.logger.info("Validation nDCG : {:0.6f}".format(validation_score))
                self._validation_scores.append(validation_score)
                
        self.logger.info("Finish Training")
        
        return self.model

    def validate(self, y, output):
        a = itertools.count()
        ndcg_list = np.apply_along_axis(lambda x: self.ndcg_np(y[next(a)], x), 1, output)
        return ndcg_list.mean()
    
    def ndcg_np(self, aaa, x):
        truth = np.where(aaa != 0)[0]
        self._eval.set_data(truth, x)
        return self._eval.ndcg()
    
    def inference(self, inp):
        output, _, _ = self.model(inp)
        return output

    def _recommend(self, inp, topk=100):
        seen = np.where(inp!=0)
        output, _, _ = self.model(inp)
        output = output.numpy()
        output[tuple(np.vstack(seen).tolist())] = -np.inf
        final_outputs = np.argsort(output)[:,-topk:][:,::-1]
        assert self._check_seen_removed(seen, final_outputs) == 0, "Seen not filtered"
        return final_output
    
    def recommend(self, inp, idx2user:dict, topk=100):
        seen = np.where(inp!=0)
        output, _, _ = self.model(inp)
        output = output.numpy()
        output[tuple(np.vstack(seen).tolist())] = -np.inf
        final_outputs = np.argsort(output)[:,-topk:][:,::-1]
        assert self._check_seen_removed(seen, final_outputs) == 0, "Seen not filtered"
        rec_pool = []
        for final_output in final_outputs:
            rec_pool.append(idx2user[final_output])
        return rec_pool
    
    @staticmethod
    def _check_seen_removed(a:np.array, b:np.array):
        a_aux = dict()
        for i,j in np.vstack(a).T.tolist():
            if i in a_aux.keys():
                a_aux[i].append(j)
            else:
                a_aux[i] = [j]
        a = itertools.count()
        return np.apply_along_axis(lambda x: bool(set(x).intersection(set(a_aux[next(a)]))), 1, b).sum()