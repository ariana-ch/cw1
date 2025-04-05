import keras
import numpy as np
import tensorflow as tf
import torch
from keras import Model
from keras import ops
from keras.api.metrics import Mean


class VAE(Model):

    def __init__(self, encoder, decoder, num_mc_samples=1, **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_metric = Mean(name='loss')
        self.nll_metric = Mean(name='nll')
        self.kl_metric = Mean(name='kl')
        self.pi = ops.array(np.pi)
        self.L = num_mc_samples

    def compute_losses(self, data):
        """
        This method should compute and return the loss, kl_loss and nll_loss.
        You should override this method as necessary in your implementations.
        """
        z_mean, z_log_var = self.encoder(data)
        kl_loss = 0.5 * ops.sum((ops.square(z_mean) + ops.exp(z_log_var) - 1 - z_log_var), axis=-1)
        kl_loss = ops.mean(kl_loss)

        epsilon = keras.random.normal(ops.shape(z_mean))
        z_std = ops.exp(0.5 * z_log_var)
        z_sample = z_mean + (z_std * epsilon)

        x_mean, x_log_std = self.decoder(z_sample)
        log_Z = 0.5 * ops.log(2 * self.pi)
        nll_loss = 0.5 * ops.square((data - x_mean) / ops.exp(x_log_std)) + x_log_std + log_Z
        nll_loss = ops.mean(ops.sum(nll_loss, axis=[-1, -2]))

        loss = kl_loss + nll_loss
        return loss, kl_loss, nll_loss

    def call(self, inputs):
        """
        This method should compute the approximate posterior using the encoder,
        and draw a single sample to pass through the decoder.
        You should override this method as necessary in your implementations.
        """
        z_mean, z_log_var = self.encoder(inputs)
        epsilon = keras.random.normal(ops.shape(z_mean))
        z_std = ops.exp(0.5 * z_log_var)
        z_sample = z_mean + (z_std * epsilon)
        return self.decoder(z_sample)

    def train_step(self, data):
        if keras.config.backend() == 'tensorflow':
            with tf.GradientTape() as tape:
                loss, kl_loss, nll_loss = self.compute_losses(data)
                loss = ops.mean(loss)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        else:
            assert keras.config.backend() == 'torch'
            self.zero_grad()
            loss, kl_loss, nll_loss = self.compute_losses(data)
            loss = ops.mean(loss)

            loss.backward()

            gradients = [v.value.grad for v in self.trainable_weights]
            with torch.no_grad():
                self.optimizer.apply(gradients, self.trainable_weights)

        self.loss_metric.update_state(loss)
        self.nll_metric.update_state(nll_loss)
        self.kl_metric.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        loss, kl_loss, nll_loss = self.compute_losses(data)
        loss = ops.mean(loss)
        self.loss_metric.update_state(loss)
        self.nll_metric.update_state(nll_loss)
        self.kl_metric.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_metric, self.nll_metric, self.kl_metric]