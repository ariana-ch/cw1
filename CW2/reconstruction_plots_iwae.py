import os
os.environ['KERAS_BACKEND'] = 'torch'
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from keras import ops
import keras
print(keras.backend.backend())
from CW2.dataloader import get_datasets
from CW2.iwae_diagonal import IWAEDiagonal
from CW2.encoder_decoder import get_encoder_v8, get_decoder_v8


def get_trained_IWAEDiagonal():
    backend = keras.backend.backend()
    root = pathlib.Path(f'./{backend}/IWAEDiagonal')
    model_weights_path = root.joinpath('model.weights.h5')
    encoder = get_encoder_v8(image_shape=(28, 28, 1), latent_dim=2)
    decoder = get_decoder_v8(image_shape=(28, 28, 1), latent_dim=2)
    model = IWAEDiagonal(encoder=encoder, decoder=decoder, num_mc_samples=1, k=1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7))
    model(keras.random.normal(shape=(1, 28, 28, 1)))
    model.load_weights(model_weights_path, skip_mismatch=True)
    return model



def sample_z_IWAE(model, x_true, k):
    '''
    Return a single sample z, using the sampling algorithm 1.
    Args:
        model: A trained VAE model. Here it should be an instance of IWAEDiagonal from 2b
        x_true: an input image
        k: The number of importance samples to use

    Returns: the mean and log variance of the latent variable for the sample
    '''
    eps = 1e-7

    z_mean, z_log_var = model.encoder(x_true)  # (1, latent_dim)
    z_mean = ops.squeeze(z_mean, axis=0)  # (2,)
    z_log_var = ops.squeeze(z_log_var, axis=0)

    # Sample z_i ~ q(z|x) (could also use model._sample_z
    epsilon = keras.random.normal(shape=(k, 2))  # (k, 2)
    z_std = ops.exp(0.5 * z_log_var)  # (2,)
    z_samples = z_mean + epsilon * z_std  # (k, 2)

    # log p(z_i) ~ standard normal
    log_p_z = -0.5 * ops.sum(ops.square(z_samples), axis=-1)  # shape: (k,)

    # Decoder gives p(x|z) = Bernoulli(x | x_pred)
    x_pred = model.decoder(z_samples)  # (k, 28, 28, 1)
    x = ops.repeat(x_true, k, axis=0)  # (k, 28, 28, 1)

    x_pred = ops.clip(x_pred, eps, 1. - eps)
    log_p_x_given_z = ops.sum(x * ops.log(x_pred) + (1 - x) * ops.log(1 - x_pred), axis=[1, 2, 3])  # (k,)

    # log q(z|x) fully factorised Gaussian
    log_q_z_given_x = -0.5 * ops.sum(((z_samples - z_mean) ** 2) / ops.exp(z_log_var) + z_log_var + np.log(2 * np.pi),
                                     axis=-1)

    # Importance weights
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
    w = ops.exp(log_w - ops.max(log_w))  # for numerical stability
    w /= ops.sum(w)

    # Sample index j ~ categorical(w) - the keras categorical takes logits!!!!!
    j = keras.random.categorical(ops.expand_dims(log_w, axis=0), num_samples=1)
    # j = keras.random.categorical(ops.expand_dims(w, axis=0), num_samples=1)
    return z_samples[j[0]]


def plot_reconstructions(model, test_ds, num_images=5, k=50):
    batch = next(iter(test_ds))
    images = batch[:num_images]

    fig, axes = plt.subplots(num_images, 3, figsize=(9, 2.5 * num_images))

    for i, x in enumerate(images):
        # p_\theta(x|z), standard VAE
        x = ops.expand_dims(x, axis=0)
        z_mean, z_log_var = model.encoder(x)
        eps = keras.random.normal(shape=(1, 2))
        z = z_mean + ops.exp(0.5 * z_log_var) * eps
        recon_vae = model.decoder(z)  # (1, 28, 28, 1)

        # p_\theta(x|z) using IWAE Sample for q_\phi(z|x)
        z_iwae = sample_z_IWAE(model, x, k=k)
        recon_iwae = model.decoder(z_iwae)  # (1, 28, 28, 1)

        # Plotting
        axes[i, 0].imshow(x[0, ..., 0].detach().cpu(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(recon_vae[0, ..., 0].detach().cpu(), cmap='gray')
        axes[i, 1].set_title("VAE Reconstruction")
        axes[i, 2].imshow(recon_iwae[0, ..., 0].detach().cpu(), cmap='gray')
        axes[i, 2].set_title(f"IWAE Reconstruction (k={k})")

        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = get_trained_IWAEDiagonal()
    _, _, test_ds = get_datasets(500)
    plot_reconstructions(model=model, test_ds=test_ds, num_images=5, k=50)