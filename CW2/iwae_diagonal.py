from CW2.base_vae import VAE
from CW2.vae_diagonal import VAEDiagonal
from keras.api.metrics import Mean
import numpy as np
import keras
from keras import ops
from keras.src.saving import register_keras_serializable

import keras
from keras import layers, ops
import numpy as np


class IWAEDiagonal(VAEDiagonal):

    def __init__(self, encoder, decoder, k, k_chunk_size=1, num_mc_samples=1,  **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples, **kwargs)
        self.k = k
        if k % k_chunk_size == 0:
            self.n = k // k_chunk_size
            self.k_chunk_size = k_chunk_size
        else:
            self.n = k
            self.k_chunk_size = 1

    # noinspection PyMethodOverride
    def _sample_z(self, z_mean, z_log_var, samples):
        '''
        Sample z using the mean and standard deviation output from the decoder and Gaussian
        noise epsilon ~ N(0, 1) (l dimensional)

            z = mu_q + std_q * epsilon
        Args:
            z_mean: The mean of z, obtained using the encoder
            z_log_var: The log variance of z, obtained using the decoder
            samples: number of samples to return

        Returns: An (samples, l) tensor, where l is the dimension of the latent space and
        samples is the number of samples required
        '''
        batch_size, latent_dim = ops.shape(z_mean)

        # Sample epsilon ~ N(0, 1)
        epsilon = keras.random.normal(shape=(samples, batch_size, latent_dim))

        # Reparameterisation trick
        z_std = ops.exp(0.5 * z_log_var)
        z_samples = z_mean + z_std * epsilon  # shape: (L, B, l)

        return z_samples

    def log_q_z_given_x(self, z_samples, z_mean, z_log_var):
        '''
        Compute the log likelihood for the posterior distribution of the latent
        space for fully factorised gaussian case - to be attached to the VAEDiagonal
        class

        Args:
            z_samples: (MC samples, batch_size, 2)
            z_mean: (batch_size, 2)
            z_log_var: (batch_size, 2), log_var of the posterior distribution

        Returns: (MC samples * batch_size, ), the log likelihood of z given x
        '''
        mc_samples, batch_size, latent_dim = keras.ops.shape(z_samples)  # mc_samples = self.L

        # mahalanobis distance
        z_centered = z_samples - keras.ops.expand_dims(z_mean, 0)
        z_log_var = keras.ops.expand_dims(z_log_var, 0)
        inv_var = keras.ops.exp(-z_log_var)
        maha = keras.ops.sum(z_centered ** 2 * inv_var, axis=-1)

        # log determinant
        log_det = keras.ops.sum(z_log_var, axis=-1)

        # log likelihood
        return -0.5 * (maha + log_det + latent_dim * keras.ops.log(2 * np.pi))

    def bce_loss(self, x_true, z_mean, z_log_var):
        '''
        Compute NLL loss for VAE

        Args:
            x_true: batch of input images
            z_mean: latent space mean - encoder output
            z_log_var: latent space log variance - encoder output

        Returns: scalar (float)
        '''
        # NLL/BCE Loss
        # Sample z \sim q(z|x)
        eps = 1e-7
        batch_size, latent_dim = ops.shape(z_mean)
        z_samples = self._sample_z(z_mean, z_log_var, self.L)  # (MC samples, batch, latent_dim)
        z_samples = ops.reshape(z_samples, (self.L * batch_size, latent_dim))  # (MC Samples * batch, latent_dim)

        # Decode: returns logits or probabilities
        x_pred = self.decoder(z_samples)  # (MC samples * batch, H, W, C)
        _, H, W, C = ops.shape(x_true)

        # Expand ground truth to match shape
        x_true = ops.expand_dims(x_true, axis=0)  # (1, batch, H, W, C)
        x_true = ops.repeat(x_true, self.L, axis=0)  # (MC samples, batch, H, W, C)
        x_true = ops.reshape(x_true, ops.shape(x_pred))

        # Clip probs to avoid log(0)
        x_pred = ops.clip(x_pred, eps, 1. - eps)

        bce_per_pixel = -(
            x_true * ops.log(x_pred) +
            (1. - x_true) * ops.log(1. - x_pred)
        )  # shape: (MC samples * batch, H, W, C)

        # Sum over pixel dimensions (H, W, C) -> total BCE per image
        bce_per_image = ops.sum(bce_per_pixel, axis=[-1, -2, -3])  # shape: (MC samples, batch)

        # Average over MC and batch
        nll_loss = ops.mean(bce_per_image)
        return nll_loss

    def kl_divergence(self, z_mean, z_log_var):
        '''
        Compute KL divergence analytically for VAE

        Args:
            z_mean: latent space mean - encoder output
            z_log_var: latent space log variance - encoder output

        Returns: scalar (float)
        '''
        # KL divergence term (analytical)
        kl_per_sample = 0.5 * ops.sum(
            ops.square(z_mean) + ops.exp(z_log_var) - 1 - z_log_var,
            axis=-1  # sum over latent dim
        )
        kl = ops.mean(kl_per_sample)  # scalar
        return kl

    def iwae_loss(self, x_true, z_mean, z_log_var):
        eps = 1e-7
        _, H, W, C = ops.shape(x_true)
        batch_size = ops.shape(x_true)[0]
        z_mean = ops.squeeze(z_mean, axis=0)
        z_log_var = ops.squeeze(z_log_var, axis=0)
        log_w = []
        for i in range(self.n):
            # Sample z: shape (k_chunk_size, batch_size, l)
            z_samples = self._sample_z(z_mean, z_log_var, samples=self.k_chunk_size)

            # Step 3: compute (MC samples, batch_size) image predictions using the decoder
            x_pred = self.decoder(ops.reshape(z_samples, (self.k_chunk_size * batch_size, 2)), training=True)
            x_pred = ops.reshape(x_pred, (self.k_chunk_size, batch_size, H, W, C))
            x_pred = ops.clip(x_pred, eps, 1 - eps)

            # log p(x|z)
            x_true = ops.expand_dims(x_true, axis=0)  # (1, batch_size, 28, 28, 1)
            log_p_x_given_z = ops.sum(x_true * ops.log(x_pred) + (1. - x_true) * ops.log(1. - x_pred), axis=[-1, -2, -3])

            # log p(z)
            log_p_z = -0.5 * ops.sum(z_samples ** 2 + ops.log(2 * np.pi), axis=-1)  # (k, B)

            # log q(z|x)
            log_q_z_given_x = self.log_q_z_given_x(z_samples, z_mean, z_log_var)
            log_w_chunk = log_p_x_given_z + log_p_z - log_q_z_given_x
            log_w.append(log_w_chunk)
        log_w = ops.concatenate(log_w)  # tensor with the (k samples per image in the batch, images in the batch)
        log_w = ops.logsumexp(log_w, axis=0) - ops.log(float(self.k))  # shape (batch_size,)
        return -ops.mean(log_w)

    def compute_losses(self, data):
        '''
        Compute the losses. Now needs to be IWAE NLL
        Args:
            data: a minibatch of the data

        Returns: the IWAE NLL
        '''
        # Encode the data -> image shape to latent dim
        z_mean, z_log_var = self.encoder(data)  # (batch, latent_dim)
        kl = self.kl_divergence()
        nll_loss = self.bce_loss(x_true=data, z_mean=z_mean, z_log_var=z_log_var)
        iwae_loss = self.iwae_loss()

        return iwae_loss, kl, nll_loss

