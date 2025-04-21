import os
os.environ['KERAS_BACKEND'] = 'torch'
from keras import ops
import keras
import tensorflow as tf
import numpy as np
from CW2.iwae_diagonal import IWAEDiagonal
from CW2.vae_diagonal import VAEDiagonal
from CW2.encoder_decoder import get_decoder_v8, get_encoder_v8
import pathlib
import torch


def vae_loss(z_mean, z_log_var, decoder, data, *args, **kwargs):
    '''
    Helper function to compute vae loss
    '''
    eps = 1e-7
    epsilon = keras.random.normal(shape=z_mean.shape)
    z = z_mean + ops.exp(0.5 * z_log_var) * epsilon
    x_pred = decoder(z, training=False)
    x_pred = ops.clip(x_pred, eps, 1 - eps)
    recon_loss = ops.sum(data * ops.log(x_pred) + (1 - data) * ops.log(1 - x_pred), axis=[1, 2, 3])
    kl_div = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
    return -ops.mean(recon_loss - kl_div)


def iwae_loss(z_mean, z_log_var, decoder, data, k, *args, **kwargs):
    '''
    Helper function to compute iwae loss
    '''
    eps = 1e-7
    z_mean, z_log_var = ops.squeeze(z_mean, axis=0), ops.squeeze(z_log_var, axis=0)
    epsilon = keras.random.normal(shape=(k, z_mean.shape[-1]))
    z = z_mean + ops.exp(0.5 * z_log_var) * epsilon
    x_pred = decoder(z)
    data_tiled = ops.tile(data, [k, 1, 1, 1])
    x_pred = ops.clip(x_pred, eps, 1 - eps)
    log_p_x_given_z = ops.sum(data_tiled * ops.log(x_pred) + (1 - data_tiled) * ops.log(1 - x_pred),
                              axis=[1, 2, 3])
    log_p_z = -0.5 * ops.sum(ops.square(z), axis=-1)
    log_q_z_given_x = -0.5 * ops.sum(
        ((z - z_mean) ** 2) / ops.exp(z_log_var) + z_log_var + np.log(2 * np.pi), axis=-1)
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
    log_w_norm = log_w - ops.max(log_w)
    w = ops.exp(log_w_norm)
    w /= ops.sum(w)
    return -ops.sum(w * log_w)  # IWAE bound as expectation


def compute_snr(encoder, decoder, variables, objective_type, data,
                num_mc_samples, k=None):
    """
    Computes estimates for the signal-to-noise ratio (SNR) of gradients.

    Args:
        encoder: Keras model object for the encoder.
        decoder: Keras model object for the decoder.
        variables: List of Keras Variable objects (parameters for SNR).
        objective_type: String, either 'VAE' or 'IWAE'.
        data: A single data example tensor (1, H, W, C)
        num_mc_samples: number of MC samples (M) for expectation/std dev.
        k: number of importance samples for IWAE objective (required if objective_type='IWAE').

    Returns:
        A list of SNR Tensors, corresponding element-wise to the input `variables`.
    """
    if len(data.shape) == 3:
        data = ops.expand_dims(data, axis=0)
    if data.shape[0] != 1:
        raise ValueError(f'SNR is computed for one image. You provided a batch of {data.shape[0]} images')
    if objective_type == 'IWAE' and k is None:
        raise ValueError(f"Parameter 'k' is required for objective_type='IWAE'.")

    # Get parameters of q_phi(z|x). These remain fixed and we use them for MC
    z_mean, z_log_var = encoder(data, training=False)  # Shape: (1, latent_dim)
    gradients_list = [[] for _ in
                      variables]  # Store gradients for each variable. I am assuming these are only the trainable parameters
    loss_fn = vae_loss if objective_type == 'VAE' else iwae_loss

    for _ in range(num_mc_samples):
        if keras.config.backend() == 'tensorflow':
            with tf.GradientTape() as tape:
                tape.watch(variables)
                loss = loss_fn(z_mean, z_log_var, decoder, data, k)
                loss = ops.mean(loss)
            grads = tape.gradient(loss, variables)
        else:
            assert keras.config.backend() == 'torch'
            loss = loss_fn(z_mean, z_log_var, decoder, data, k)
            loss = ops.mean(loss)

            torch_params = [getattr(v, 'value', v) for v in variables]  # Extract tensors
            grads = list(torch.autograd.grad(
                outputs=loss,
                inputs=torch_params,
                retain_graph=True,
                allow_unused=True
            ))

        # Store gradients
        for i, g in enumerate(grads):
            if g is not None:
                gradients_list[i].append(g.detach().cpu())
    eps = 1e-7
    #  Compute Mean, Std Dev, and SNR for each variable
    snr_list = []
    for i, var_grads in enumerate(gradients_list):
        stacked_grads = ops.stack(var_grads, axis=0)  # (num_mc_samples, *var_shape)

        # Estimate Mean Gradient and Stdev
        mean_grad = ops.mean(stacked_grads, axis=0)
        variance_grad = ops.mean(ops.square(stacked_grads - mean_grad), axis=0)
        std_dev_grad = ops.sqrt(variance_grad)

        # Compute SNR = | E[Delta] / (sigma[Delta] + epsilon) |
        snr = ops.abs(mean_grad / (std_dev_grad + eps))
        snr_list.append(snr)
    return snr_list


def get_trained_VAEDiagonal():
    backend = keras.backend.backend()
    weight_path = pathlib.Path(f'./{backend}/VAEDiagonal_v8/model.weights.h5')
    model = VAEDiagonal(encoder=get_encoder_v8((28, 28, 1), 2), decoder=get_decoder_v8((28, 28, 1), 2))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7))
    model(keras.random.normal((1, 28, 28, 1)))
    model.load_weights(weight_path, skip_mismatch=True)
    return model


def get_trained_IWAEDiagonal():
    backend = keras.backend.backend()
    backend='torch'
    root = pathlib.Path(f'./{backend}/IWAEDiagonal')
    model_weights_path = root.joinpath('model.weights.h5')
    encoder = get_encoder_v8(image_shape=(28, 28, 1), latent_dim=2)
    decoder = get_decoder_v8(image_shape=(28, 28, 1), latent_dim=2)
    model = IWAEDiagonal(encoder=encoder, decoder=decoder, num_mc_samples=1, k=1)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7))
    model(keras.random.normal(shape=(1, 28, 28, 1)))
    model.load_weights(model_weights_path, skip_mismatch=True)
    return model


def flatten_snr(snr_list):
    snr_vec = ops.concatenate([v.flatten() for v in snr_list]).flatten()
    return snr_vec.detach().cpu()


def compute_snr_for_batch(model, variables, objective_type, data_batch, num_mc_samples, k=None):
    snr_list = [ops.zeros(v.shape) for v in variables]
    cnt = 0
    encoder = model.encoder
    decoder = model.decoder
    for data in data_batch:
        snr = compute_snr(encoder=encoder, decoder=decoder, variables=variables, objective_type=objective_type,
                          data=data, num_mc_samples=num_mc_samples, k=k)
        for j, snr_val in enumerate(snr):
            snr_list[j] += snr_val
        cnt += 1
    if cnt > 1:
        snr_list = [total_snr / cnt for total_snr in snr_list]
    return flatten_snr(snr_list)



if __name__ == '__main__':
    from CW2.dataloader import get_datasets

    train_ds, _, _ = get_datasets()
    vae_model = get_trained_VAEDiagonal()
    iwae_model = get_trained_IWAEDiagonal()

    vae_vars = vae_model.trainable_variables
    iwae_vars = iwae_model.trainable_variables
    x = next(iter(train_ds))[0, ...]
    vae_snr = compute_snr(encoder=vae_model.encoder, decoder=vae_model.decoder, objective_type='VAE',
                          data=x, num_mc_samples=20, variables=vae_vars)
    iwae_snr = compute_snr(encoder=vae_model.encoder, decoder=vae_model.decoder, objective_type='VAE',
                           data=x, num_mc_samples=20, variables=iwae_vars)
