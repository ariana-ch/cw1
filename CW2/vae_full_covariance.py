from CW2.base_vae import VAE
from keras.api.metrics import Mean
import numpy as np
import keras
from keras import ops
from keras.src.saving import register_keras_serializable


class VAEFullCovariance(VAE):
    def __init__(self, encoder, decoder, num_mc_samples=1, **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples, **kwargs)

    def _build_cholesky_2d(self, components):
        '''
        Covert a (batch_size, 3) vector (raw Cholesky components obtained from the encoder) into
        batch_size Cholesky matrices and return a (batch_size, 2, 2) tensor
        Args:
            components: the (batch_size, 3) components [L_00_raw, L_10, L_11_raw]
            where _raw implies that they are not necessary positive.

        Returns: (batch_size, 2, 2) lower-triangular matrices with softplus diagonals.
        '''
        L_00_raw = components[:, 0]
        L_10 = components[:, 1]
        L_11_raw = components[:, 2]

        L_00 = ops.softplus(L_00_raw)
        L_11 = ops.softplus(L_11_raw)

        row1 = ops.stack([L_00, ops.zeros_like(L_00)], axis=-1)
        row2 = ops.stack([L_10, L_11], axis=-1)
        L = ops.stack([row1, row2], axis=1)
        return L

    def _sample_z(self, z_mean, cholesky_raw):
        '''
        Sample z using the mean and standard deviation output from the decoder and Gaussian
        noise epsilon ~ N(0, I) (l dimensional)

            z = mu_q + L_q * epsilon
        Args:
            z_mean: (batch_size, l) The mean of z, obtained using the encoder
            cholesky_raw: (batch_size, 3) The raw components of the Cholesky factor.

        Returns: An (L, 2) tensor, where l=2 is the dimension of the latent space and
        L is the number of Monte Carlo samples required
        '''
        batch_size, latent_dim = ops.shape(z_mean)
        epsilon = keras.random.normal(
            shape=(self.L, batch_size, 2, 1))  # (MC Samples, batch_size, 2, 1), last dim needed for column vectors

        # need MC samples copies of the Cholesky factor
        L_cholesky = self._build_cholesky_2d(cholesky_raw)  # (batch_size, 2, 2)
        L_cholesky = ops.expand_dims(L_cholesky, axis=0)  # (1, batch_size, 2, 2)
        L_cholesky = ops.repeat(L_cholesky, self.L, axis=0)  # (MC Samples, batch_size, 2, 2)

        # need MC samples copies of the mean *column* vector
        z = ops.expand_dims(z_mean, axis=-1)  # (batch_size, 2, 1) Column vectors
        z = ops.expand_dims(z, axis=0)  # (1, batch_size, 2, 1)
        z = ops.repeat(z, self.L, axis=0)  # (MC samples, batch_size, 2, 1)

        z = z + ops.matmul(L_cholesky, epsilon)  # (MC samples, batch_size, 2, 1)
        return ops.squeeze(z, axis=-1)  # (MC samples, batch_size, 2

    def call(self, inputs):
        """
        This method should compute the approximate posterior using the encoder,
        and draw a single sample to pass through the decoder.
        You should override this method as necessary in your implementations.
        """
        z_mean, cholesky_raw = self.encoder(inputs)  # (batch_size, l), (batch_size, l(l+1)/2) where l = 2
        assert ops.shape(z_mean)[-1] == 2

        L_cholesky = self._build_cholesky_2d(cholesky_raw)  # (batch_size, 2, 2)
        epsilon = keras.random.normal(ops.shape(z_mean))  # (batch_size, l)
        epsilon = ops.expand_dims(epsilon, -1)  # convert to a column vector
        z_mean = ops.expand_dims(z_mean, -1)  # convert to a column vector
        z_sample = z_mean + ops.matmul(L_cholesky, epsilon)  # (batch_size, 2, 1)
        z_sample = ops.squeeze(z_sample, axis=-1)
        return self.decoder(z_sample)

    def compute_losses(self, data):
        '''
        Compute the losses. The KL Loss term is the same as the VAE implementation but
        the reconstruction loss will now be the log likelihood of the Bernoulli distribution

        As with the base class, the encoder outputs the mean and log variance of z.
        The decoder outputs the probability $p$ of the Bernoulli distribution
        Args:
            data: a minibatch of the data

        Returns: the total loss, KL loss and negative log-likelihood loss (scalars)
        '''
        eps = 1e-7  # const used for numerical stability (logs)

        z_mean, cholesky_raw = self.encoder(data)  # (batch_size, 2), (batch_size, 3)
        batch_size, latent_dim = ops.shape(z_mean)
        assert latent_dim == 2

        L_cholesky = self._build_cholesky_2d(cholesky_raw)  # (batch_size, 2, 2)
        Sigma = ops.matmul(L_cholesky, ops.transpose(L_cholesky, [0, 2, 1]))  # (batch_size, 2, 2)

        # --------- KL divergence term (analytical) ------------
        # Compute trace of Sigma (sum of diagonal entries)
        trace = Sigma[:, 0, 0] + Sigma[:, 1, 1]  # (batch_size,)

        # Compute log(det(Sigma)) = 2 * (log L_00 + log L_11)
        log_L_00 = ops.log(L_cholesky[:, 0, 0] + eps)
        log_L_11 = ops.log(L_cholesky[:, 1, 1] + eps)
        log_det = 2 * (log_L_00 + log_L_11)

        # Compute squared norm of mean mu^T mu
        z_mean_sq = ops.sum(ops.square(z_mean), axis=-1)  # (batch_size,)

        # KL divergence term (analytical)
        kl_per_sample = 0.5 * (trace + z_mean_sq - log_det - latent_dim)
        kl_loss = ops.mean(kl_per_sample)  # scalar

        # -------- Negative Log Likelihood term (numerically) ----------

        # Sample z \sim q(z|x)
        # z_samples = self._sample_z(z_mean, z_log_var)  # (MC samples, batch, latent_dim) OLD CODE
        z_samples = self._sample_z(z_mean, cholesky_raw)
        z_samples = ops.reshape(z_samples, (self.L * batch_size, latent_dim))  # (MC Samples * batch, latent_dim)

        # Decode the z_samples
        x_pred = self.decoder(z_samples)  # (MC samples * batch, H, W, C)
        _, H, W, C = ops.shape(data)

        # Expand ground truth to match shape
        x_true = ops.expand_dims(data, axis=0)  # (1, batch, H, W, C)
        x_true = ops.repeat(x_true, self.L, axis=0)  # (MC samples, batch, H, W, C)
        x_true = ops.reshape(x_true, ops.shape(x_pred))

        # Clip probs to avoid log(0)
        x_pred = ops.clip(x_pred, eps, 1. - eps)

        # COMPUTE BCE Explicitly (using binary cross entropy loss from keras was leading to instabilities on M1 Max with Tensorflow
        bce_per_pixel = -(
            x_true * ops.log(x_pred) +
            (1. - x_true) * ops.log(1. - x_pred)
        )  # shape: (MC samples * batch, H, W, C)

        # Sum over pixel dimensions (H, W, C) -> total BCE per image
        bce_per_image = ops.sum(bce_per_pixel, axis=[-1, -2, -3])  # shape: (MC samples, batch)

        # Average over MC and batch
        nll_loss = ops.mean(bce_per_image)
        total_loss = kl_loss + nll_loss

        return total_loss, kl_loss, nll_loss