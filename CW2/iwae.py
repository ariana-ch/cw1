import os
import torch

os.environ['KERAS_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import ops
import numpy as np
from CW2.dataloader import get_datasets
from CW2.encoder_decoder import * #get_encoder_v9, get_encoder_full_covariance, get_decoder_v9, get_decoder_full_covariance
import keras
from CW2.vae_full_covariance import VAEFullCovariance
from CW2.vae_diagonal import VAEDiagonal
import pathlib


def get_trained_VAEDiagonal():
    backend = keras.backend.backend()
    weight_path = pathlib.Path(f'./{backend}/VAEDiagonal_v8/model.weights.h5')
    model = VAEDiagonal(encoder=get_encoder_v8((28, 28, 1), 2), decoder=get_decoder_v8((28, 28, 1), 2))
    model(keras.random.normal((1, 28, 28, 1)))
    model.load_weights('/Users/thx1138/Documents/MLDS/Term5/DeepLearningAssessment/CW2/torch/VAEDiagonal_v8/model.weights.h5')
    return model


def get_trained_VAEFullCovariance():
    backend = keras.backend.backend()
    weight_path = pathlib.Path(f'./{backend}/VAEFullCovariance/model.weights.h5')
    model = VAEFullCovariance(encoder=get_encoder_full_covariance((28, 28, 1), 2),
                              decoder=get_decoder_full_covariance((28, 28, 1), 2))
    model.compile()
    model(keras.random.normal((1, 28, 28, 1)))
    model.load_weights(weight_path)
    return model



def _log_q_z_given_x_diag(self, z, z_mean, z_log_var):
    '''
    Compute the log likelihood for the posterior distribution of the latent
    space for fully factorised gaussian case - to be attached to the VAEDiagonal
    class

    Args:
        z: (MC samples, batch_size, 2)
        z_mean: (batch_size, 2)
        z_log_var: (batch_size, 2), log_var of the posterior distribution

    Returns: (MC samples * batch_size, ), the log likelihood of z given x
    '''
    mc_samples, batch_size, latent_dim = keras.ops.shape(z)  # mc_samples = self.L

    # mahalanobis distance
    z_centered = z - keras.ops.expand_dims(z_mean, 0)
    z_log_var = keras.ops.expand_dims(z_log_var, 0)
    inv_var = keras.ops.exp(-z_log_var)
    maha = keras.ops.sum(z_centered ** 2 * inv_var, axis=-1)

    # log determinant
    log_det = keras.ops.sum(z_log_var, axis=-1)

    # log likelihood
    log_q = -0.5 * (maha + log_det + latent_dim * keras.ops.log(2 * np.pi))
    return keras.ops.reshape(log_q, (mc_samples * batch_size,))


def _log_q_z_given_x_full(self, z, z_mean, cholesky_params):
    '''
    Compute the log likelihood for the posterior distribution of the latent
    space for full covariance model - to be attached to the VAEFullCovariance
    class

    Args:
        z: (MC samples, batch_size, 2)
        z_mean: (batch_size, 2)
        cholesky_params: (batch_size, 3), raw components of the Cholesky factor

    Returns: (MC samples * batch_size, ), the log likelihood of z given x
    '''
    eps = 1e-7 # numerical stability
    mc_samples, batch_size, latent_dim = keras.ops.shape(z) # mc_samples = self.L

    L = self._build_cholesky_2d(cholesky_params)  # (batch_size, 2, 2)

    # mahalanobis distance
    z_centered = z - keras.ops.expand_dims(z_mean, 0)
    z_centered = keras.ops.expand_dims(z_centered, -1)  # (MC samples, batch_size, 2, 1)
    L_exp = keras.ops.expand_dims(L, 0)  # (1, batch_size, 2, 2)
    x = keras.ops.linalg.solve_triangular(L_exp, z_centered, lower=True)
    x = keras.ops.squeeze(x, axis=-1)  # (MC samples, batch_size, 2)
    maha = keras.ops.sum(x ** 2, axis=-1)  # (MC samples, batch_size)

    # log determinant
    log_L_00 = ops.log(L[:, 0, 0] + eps)
    log_L_11 = ops.log(L[:, 1, 1] + eps)
    log_det = 2 * (log_L_00 + log_L_11) # (batch_size,)
    log_det = ops.expand_dims(log_det, axis=0) # (batch_size, 1)

    # log likelihood
    log_q = -0.5 * (maha + log_det + latent_dim * keras.ops.log(2 * np.pi))
    return keras.ops.reshape(log_q, (mc_samples * batch_size,))


def importance_sampling_nnl(model, dataset, k=5000, k_batch = 25):
    batches = k//k_batch
    assert k % k_batch == 0

    sum_log_W_i = 0
    total_samples = 0

#
# def compute_iwae_nll(model, dataset, k=5000):
#     """
#     Memory-efficient IWAE estimate.
#     Uses model._sample_z to get a single sample at a time — supports richer posteriors.
#     """
#     sum_log_iw = 0.0
#     total_samples = 0
#
#     for x in dataset:
#         x = ops.convert_to_tensor(x) # in case I am using torch backend.
#         batch_size = ops.shape(x)[0]
#         z_mean, z_logvar = model.encoder(x, training=False)
#
#         log_ws = []
#
#         for _ in range(k):
#             z_k = model._sample_z(z_mean, z_logvar)  # shape (1, batch_size, 2), one sample per x
#             if keras.backend.backend() == "torch":
#                 import torch
#                 with torch.no_grad():
#                     x_recon_k = model.decoder(ops.squeeze(z_k, axis=0), training=False)
#                 torch.mps.empty_cache()
#                 print(torch.mps.current_allocated_memory())
#             else:
#                 x_recon_k = model.decoder(ops.squeeze(z_k, axis=0), training=False)  # shape (B, H, W, C)
#
#             x_recon_k = ops.clip(x_recon_k, 1e-7, 1 - 1e-7)
#
#             # Log p(x|z)
#             bce = -(x * ops.log(x_recon_k) + (1 - x) * ops.log(1. - x_recon_k))
#             log_px_given_z = ops.sum(bce, axis=[-1, -2, -3])  # (B,)
#
#             # Log p(z)
#             log_pz = -0.5 * ops.sum(z_k ** 2 + ops.log(2 * np.pi), axis=-1)  # (B,)
#
#             # Log q(z|x) from model (shape must be (B,))
#             log_qz_given_x = model.log_q_z_given_x(z_k, z_mean, z_logvar) #[0]  # shape (B,)
#
#             log_ws.append(log_px_given_z + log_pz - log_qz_given_x)  # (B,)
#
#         log_ws = ops.stack(log_ws, axis=0)  # shape (k, B)
#         log_iw = ops.logsumexp(log_ws, axis=0) - ops.log(float(k))  # shape (B,)
#
#         sum_log_iw += ops.sum(log_iw)
#         total_samples += ops.shape(log_iw)[0]
#
#     return float(-sum_log_iw / total_samples)

#
# def compute_iwae_nll(model, dataset, k=5000):
#     """
#     Compute the IWAE-based negative log-likelihood estimate for a dataset.
#
#     Args:
#         model: VAE model (must implement encoder, decoder, _sample_z, log_q_z_given_x)
#         dataset: batched dataset of images
#         k: number of importance samples per input
#
#     Returns:
#         Estimated average negative log-likelihood (scalar)
#     """
#     # log_weights_all = [] # can't hold them in memory - blows up. Need a streaming version
#
#     sum_log_iw = 0.0
#     total_samples = 0
#
#     model.L = k # set the number of MC samples
#
#     for x in dataset:
#         batch_size = ops.shape(x)[0]
#         latent_dim = model.encoder.output[0].shape[-1]
#
#         batch_size = keras.ops.shape(x)[0]
#         latent_dim = model.encoder.output[0].shape[-1]
#
#         # Encode input -> z_mean + variance parameter
#         encoder_outputs = model.encoder(x)
#         z_mean = encoder_outputs[0]
#
#         # Sample z ~ q(z|x), shape (k, B, l)
#         z_samples = model._sample_z(z_mean, encoder_outputs[1])
#
#         # Decoder works batch-wise: expects shape (B, D)
#         # z is (k, batch_size, 2) - decode each one at lot [1, :, :] at a time and stack outputs manually
#         if keras.backend.backend() == "torch":
#             x_recon_list = []
#             import torch
#             with torch.no_grad():
#                 for z_k in ops.unstack(z_samples, axis=0):
#                     x_recon_list.append(model.decoder(ops.squeeze(z_k, axis=0), training=False))
#                     torch.mps.empty_cache()
#                 print(torch.mps.current_allocated_memory())
#             x_recon = ops.stack(x_recon_list, axis=0)  # (k, batch_size, H, W, C)
#         else:
#             x_recon = model.decoder(ops.squeeze(z_k, axis=0), training=False)  # shape (B, H, W, C)
#
#         # Expand and tile x to match shape (k, batch_size, H, W, C)
#         x_true = ops.expand_dims(x, 0)
#         x_true = ops.repeat(x_true, k, axis=0)
#
#         # Compute log p(x|z) — Bernoulli log-likelihood
#         x_recon = ops.clip(x_recon, 1e-7, 1 - 1e-7)
#         log_px_given_z = -(
#             x_true * ops.log(x_recon) + (1 - x_true) * ops.log(1 - x_recon)
#         )
#         log_px_given_z = ops.sum(log_px_given_z, axis=[-1, -2, -3])  # (k, batch_size)
#
#         # Compute log p(z)
#         log_pz = -0.5 * ops.sum(z_samples ** 2 + ops.log(2 * np.pi), axis=-1)  # (k, batch_size)
#
#         # Compute log q(z|x)
#         log_qz_given_x = model.log_q_z_given_x(z_samples, encoder_outputs[0], encoder_outputs[1])  # (k, batch_size)
#
#         # IWAE weights
#         log_w = log_px_given_z + log_pz - log_qz_given_x  # (k, batch_size)
#         log_iw = ops.logsumexp(log_w, axis=0) - ops.log(float(k))  # (batch_size,)
#
#         sum_log_iw += ops.sum(log_iw)
#         total_samples += ops.shape(log_iw)[0]
#
#     return float(-sum_log_iw / total_samples)
#
#
# import keras
# from keras import ops
# import numpy as np


def compute_iwae_nll(model, dataset, k=5000):
    """
    Memory-safe IWAE negative log-likelihood estimator for Keras 3 + PyTorch (MPS-safe).
    """

    backend = keras.backend.backend()
    sum_log_iw = 0.0
    total_samples = 0

    for x in dataset:
        x = ops.convert_to_tensor(x)  # ensure correct backend
        z_mean, z_logvar = model.encoder(x)  # shape (B, D)
        B = ops.shape(x)[0]

        s = None  # max log-w across k
        sum_exp = None  # sum exp(log-w - s)

        for _ in range(k):
            # Sample a single z_k
            z_k = model._sample_z(z_mean, z_logvar)  # (B, D)

            # Decode under torch.no_grad to suppress autograd graph building
            if backend == "torch":
                import torch
                with torch.no_grad():
                    x_recon_k = model.decoder(ops.squeeze(z_k, axis=0), training=False)
                    torch.mps.empty_cache()
            else:
                x_recon_k = model.decoder(ops.squeeze(z_k, axis=0), training=False)

            x_recon_k = ops.clip(x_recon_k, 1e-7, 1 - 1e-7)

            # Log p(x|z_k) under Bernoulli decoder
            # one = ops.ones_like(x)
            log_px_given_z = -ops.sum(
                x * ops.log(x_recon_k) + (1. - x) * ops.log(1. - x_recon_k),
                axis=[-1, -2, -3]
            )  # (B,)

            # Log p(z)
            log_pz = -0.5 * ops.sum(z_k ** 2 + ops.log(2 * np.pi), axis=-1)

            # Log q(z|x)
            log_qz_given_x = model.log_q_z_given_x(z_k, z_mean, z_logvar)[0]

            log_w = log_px_given_z + log_pz - log_qz_given_x  # (B,)

            # Streaming log-sum-exp
            if s is None:
                s = log_w
                sum_exp = ops.ones_like(log_w)
            else:
                s_new = ops.maximum(s, log_w)
                sum_exp = ops.exp(s - s_new) * sum_exp + ops.exp(log_w - s_new)
                s = s_new

        log_iw = s + ops.log(sum_exp) - ops.log(float(k))  # shape (B,)
        sum_log_iw += ops.sum(log_iw)
        total_samples += ops.shape(log_iw)[0]

    return float(-sum_log_iw / total_samples)


def log_q_z_given_x_diag_covariance(self, z_samples, z_mean, z_log_var):
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


def log_q_z_given_x_full_covariance(self, z_sample, z_mean, cholesky_params):
    '''
    Compute the log likelihood for the posterior distribution of the latent
    space for full covariance model - to be attached to the VAEFullCovariance
    class

    Args:
        z_sample: (MC samples, batch_size, 2)
        z_mean: (batch_size, 2)
        cholesky_params: (batch_size, 3), raw components of the Cholesky factor

    Returns: (MC samples * batch_size, ), the log likelihood of z given x
    '''
    eps = 1e-7  # numerical stability
    mc_samples, batch_size, latent_dim = keras.ops.shape(z_sample)  # mc_samples = self.L

    L = self._build_cholesky_2d(cholesky_params)  # (batch_size, 2, 2)

    # mahalanobis distance
    z_centered = z_sample - keras.ops.expand_dims(z_mean, 0)
    z_centered = keras.ops.expand_dims(z_centered, -1)  # (MC samples, batch_size, 2, 1)
    L_exp = keras.ops.expand_dims(L, 0)  # (1, batch_size, 2, 2)
    x = keras.ops.linalg.solve_triangular(L_exp, z_centered, lower=True)
    x = keras.ops.squeeze(x, axis=-1)  # (MC samples, batch_size, 2)
    maha = keras.ops.sum(x ** 2, axis=-1)  # (MC samples, batch_size)

    # log determinant
    log_L_00 = ops.log(L[:, 0, 0] + eps)
    log_L_11 = ops.log(L[:, 1, 1] + eps)
    log_det = 2 * (log_L_00 + log_L_11)  # (batch_size,)
    log_det = ops.expand_dims(log_det, axis=0)  # (batch_size, 1)

    # log likelihood
    return -0.5 * (maha + log_det + latent_dim * keras.ops.log(2 * np.pi))


def log_p_x_given_z(x_pred, x_true):
    '''
    Compute the log likelihood for the posterior distribution of
    the reconstructed images (Bernoulli)
    Args:
        x_pred: reconstructed images (MC samples, batch_size, 28, 28, 1
        x_true: input images (batch_size, 28, 28, 1)

    Returns: (batch_size,) Binary cross entropy loss per prediction
    '''
    eps = 1e-7
    x_pred = ops.clip(x_pred, eps, 1 - eps)  # for numerical stability
    x_true = ops.expand_dims(x_true, axis=0)  # (1, batch_size, 28, 28, 1)

    # Log p(x|z_k) under Bernoulli decoder
    ll_per_image = ops.sum(x_true * ops.log(x_pred) + (1. - x_true) *
                           ops.log(1. - x_pred), axis=[-1, -2, -3])  # (MC samples, batch_size)
    return ll_per_image


def log_p_z_prior(z_samples):
    '''
    Compute the log likelihood of the prior for the latent space (normal
    distribution)

    Args:
        z_samples: (MC samples, batch_size, 2)

    Returns: log(p(z)) where p(z) ~ N(0, 1)

    '''
    return -0.5 * ops.sum(z_samples ** 2 + ops.log(2 * np.pi), axis=-1)



def compute_IWAE_loss(model, dataset, k=5000, k_partition = 25):
    '''

    Args:
        model:
        dataset:
        k:
        k_partition:

    Returns:

    '''

    running_total = 0
    sample_count = 0

    model.L = k_partition
    batches = len(dataset)
    for i, x in enumerate(dataset):
        log_ws = []

        print(f'[{i+1}/{batches}]')
        # Step 1: encode the images
        encoder_output = model.encoder(x, training=False)
        for j in range(k//k_partition):
            print(f'[{i + 1}/{batches}] [{j+1}/{k// k_partition}]')
            # Step 2: sample model.L MC samples
            z_samples = model._sample_z(encoder_output[0], encoder_output[1])

            # Step 3: compute (MC samples, batch_size) image predictions using the decoder
            x_pred = model.decoder(ops.reshape(z_samples, (k_partition * len(x), 2)), training=False)
            x_pred = ops.reshape(x_pred, (k_partition, len(x), 28, 28, 1))

            # compute w_i
            x_given_z = log_p_x_given_z(x_pred=x_pred, x_true=x)
            z_prior = log_p_z_prior(z_samples=z_samples)
            z_given_x = model.log_q_z_given_x(z_samples, encoder_output[0], encoder_output[1])
            log_ws.append(x_given_z + z_prior - z_given_x)  # List of l_ws of shape (k, batch_size)
        log_ws = ops.concatenate(log_ws) # tensor with the k samples per image in the batch, images in the batch
        log_wi = ops.logsumexp(log_ws, axis=0) - ops.log(float(k))  # shape (batch_size,)
        running_total += ops.sum(log_wi)
        sample_count += x.shape[0]
    return -float(running_total / sample_count)


if __name__ == '__main__':
    import tensorflow_datasets as tfds
    from CW2.dataloader import prepare_dataset
    import types

    # VAEFullCovariance.log_q_z_given_x = _log_q_z_given_x_full
    # VAEDiagonal.log_q_z_given_x = _log_q_z_given_x_diag

    train_ds = tfds.load('binarized_mnist', data_dir='data', split='test')
    train_dl = prepare_dataset(train_ds, batch_size=2000, shuffle=False)

    model = get_trained_VAEDiagonal()
    model.log_q_z_given_x = types.MethodType(log_q_z_given_x_diag_covariance, model)
    with torch.inference_mode():
        print(compute_IWAE_loss(model=model, dataset=train_dl, k=100 , k_partition=25))
