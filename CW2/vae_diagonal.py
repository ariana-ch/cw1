from CW2.base_vae import VAE
from keras.api.metrics import Mean
import numpy as np
import keras
from keras import ops
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class VAEDiagonal(VAE):

    def __init__(self, encoder, decoder, num_mc_samples=1, from_logits: bool = False, **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples, **kwargs)
        self.from_logits = from_logits


    def _sample_z(self, z_mean, z_log_var):
        '''
        Sample z using the mean and standard deviation output from the decoder and Gaussian
        noise epsilon ~ N(0, 1) (l dimensional)

            z = mu_q + std_q * epsilon
        Args:
            z_mean: The mean of z, obtained using the encoder
            z_log_var: The log variance of z, obtained using the decoder

        Returns: An (L, l) tensor, where l is the dimension of the latent space and
        L is the number of Monte Carlo samples required
        '''
        batch_size, latent_dim = ops.shape(z_mean)

        # Sample epsilon ~ N(0, 1)
        epsilon = keras.random.normal(shape=(self.L, batch_size, latent_dim))

        # Reparameterisation trick
        z_std = ops.exp(0.5 * z_log_var)
        z_samples = z_mean + z_std * epsilon  # shape: (L, B, l)

        return z_samples

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

        # Encode the data -> image shape to latent dim
        z_mean, z_log_var = self.encoder(data)  # (batch, latent_dim)
        batch_size, latent_dim = ops.shape(z_mean)

        # KL divergence term (analytical)
        kl_per_sample = 0.5 * ops.sum(
            ops.square(z_mean) + ops.exp(z_log_var) - 1 - z_log_var,
            axis=-1  # sum over latent dim
        )
        kl_loss = ops.mean(kl_per_sample)  # scalar

        # Sample z \sim q(z|x)
        z_samples = self._sample_z(z_mean, z_log_var)  # (MC samples, batch, latent_dim)
        z_samples = ops.reshape(z_samples, (self.L * batch_size, latent_dim)) # (MC Samples * batch, latent_dim)

        # Decode: returns logits or probabilities
        x_pred = self.decoder(z_samples)  # (MC samples * batch, H, W, C)
        _, H, W, C = ops.shape(data)


        # Expand ground truth to match shape
        x_true = ops.expand_dims(data, axis=0)  # (1, batch, H, W, C)
        x_true = ops.repeat(x_true, self.L, axis=0)  # (MC samples, batch, H, W, C)
        x_true = ops.reshape(x_true, ops.shape(x_pred))

        if self.from_logits:
            # --- BCE Loss ---
            # Use Keras' numerically stable BCE (handles logits internally)
            bce_per_pixel = keras.losses.binary_crossentropy(
                y_true=x_true,
                y_pred=x_pred,
                from_logits=True
            )  # shape: (L * B, H, W)
            # Sum over pixel dims (H, W) -> total BCE per image
            bce_per_image = ops.sum(bce_per_pixel, axis=[-1, -2])  # shape: (L * B,)

        else:
            # Clip probs to avoid log(0)
            x_pred = ops.clip(x_pred, eps, 1. - eps)

            # COMPUTE BCE Explicitly (using binary cross entropy loss from keras was leading to instabilities for some reason
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


@register_keras_serializable()
class VAEDiagonalOld(VAE):

    def __init__(self, encoder, decoder, num_mc_samples=1, **kwargs):
        """
        You should override this method as necessary in your implementations.
        """
        super().__init__(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples, **kwargs)


    def _sample_z(self, z_mean, z_log_var):
        '''
        Sample z using the mean and standard deviation output from the decoder and Gaussian
        noise epsilon ~ N(0, 1) (l dimensional)

            z = mu_q + std_q * epsilon
        Args:
            z_mean: The mean of z, obtained using the encoder
            z_log_var: The log variance of z, obtained using the decoder

        Returns: An (L, l) tensor, where l is the dimension of the latent space and
        L is the number of Monte Carlo samples required
        '''
        batch_size, latent_dim = ops.shape(z_mean)

        # Sample epsilon ~ N(0, 1)
        epsilon = keras.random.normal(shape=(self.L, batch_size, latent_dim))

        # Reparameterisation trick
        z_std = ops.exp(0.5 * z_log_var)
        z_samples = z_mean + z_std * epsilon  # shape: (L, B, l)

        return z_samples

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

        # Encode the data -> image shape to latent dim
        z_mean, z_log_var = self.encoder(data)  # (batch, latent_dim)
        batch_size, latent_dim = ops.shape(z_mean)

        # KL divergence term (analytical)
        kl_per_sample = 0.5 * ops.sum(
            ops.square(z_mean) + ops.exp(z_log_var) - 1 - z_log_var,
            axis=-1  # sum over latent dim
        )
        kl_loss = ops.mean(kl_per_sample)  # scalar

        # Sample z \sim q(z|x)
        z_samples = self._sample_z(z_mean, z_log_var)  # (MC samples, batch, latent_dim)
        z_samples = ops.reshape(z_samples, (self.L * batch_size, latent_dim)) # (MC Samples * batch, latent_dim)

        # Decode: returns probs in [0,1] from sigmoid
        x_prob = self.decoder(z_samples)  # (MC samples * batch, H, W, C)
        _, H, W, C = ops.shape(data)

        # Reshape into (MC Sample, batch, H, W, C)
        x_prob = ops.reshape(x_prob, (self.L, batch_size, H, W, C))

        # Expand ground truth to match shape
        x_true = ops.expand_dims(data, axis=0)  # (1, batch, H, W, C)
        x_true = ops.repeat(x_true, self.L, axis=0)  # (MC samples, batch, H, W, C)

        # Clip probs to avoid log(0)
        x_prob = ops.clip(x_prob, eps, 1. - eps)

        # COMPUTE BCE Explicitly (using binary cross entropy loss from keras was leading to instabilities for some reason
        # Step 1: Compute pixel-wise BCE (same shape)
        bce_per_pixel = -(
            x_true * ops.log(x_prob) +
            (1. - x_true) * ops.log(1. - x_prob)
        )  # shape: (MC samples, batch, H, W, C)

        # Step 2: Sum over pixel dimensions (H, W, C) -> total BCE per image
        bce_per_image = ops.sum(bce_per_pixel, axis=[-1, -2, -3])  # shape: (MC samples, batch)

        # Step 3: Mean over no of MC sample and batches
        nll_loss = ops.mean(bce_per_image)  # scalar

        # Total loss
        total_loss = kl_loss + nll_loss

        return total_loss, kl_loss, nll_loss
