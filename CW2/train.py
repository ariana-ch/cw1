import os
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'torch'
import json
import pathlib
from CW2.vae_full_covariance import VAEFullCovariance
from CW2.vae_diagonal import VAEDiagonal
from CW2.dataloader import get_datasets
import keras
from keras.api.callbacks import EarlyStopping
from CW2.encoder_decoder import *

batch_size = 500
image_shape = (28, 28, 1)


def trainVAEDiagonal(encoder_getter, decoder_getter, model_class, batch_size: int = 500, latent_dim: int=2, num_mc_samples: int = 1):
    '''
    Helper function to facilitate train

    Args:
        encoder: A keras model that will serve as the encoder
        decoder: A keras model that will serve as the decoder
        batch_size: The batch size
        embedding_dim: The dimension of the latent space
        num_mc_samples: The number of MC samples to generate when computing the
        negative log likelihood.

    Returns: a trained model and the history.
    '''

    train, val, test = get_datasets(batch_size=batch_size)

    encoder = encoder_getter(image_shape=image_shape, latent_dim=latent_dim)
    decoder = decoder_getter(image_shape=image_shape, latent_dim=latent_dim)
    model = model_class(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, run_eagerly=keras.backend.backend() == 'tensorflow')
    model(keras.random.normal((1, 28, 28, 1))) # force build the model
    early_stopping = EarlyStopping(patience=10)
    history = model.fit(train, validation_data=val, epochs=1000, callbacks=[early_stopping])
    return history, model


def train_all_VAEDiagonal():
    backend = keras.backend.backend()
    fig, ax = plt.subplots()
    _, _, test_data = get_datasets(batch_size)
    latent_dim = 2
    import CW2.encoder_decoder as ed
    for i in range(9):
        root = pathlib.Path(f'./{backend}/VAEDiagonal_v{i}')
        root.mkdir(parents=True, exist_ok=True)
        encoder = getattr(ed, f'get_encoder_v{i}')
        decoder = getattr(ed, f'get_decoder_v{i}')

        summary_path = root.joinpath('summary.txt')
        history_path = root.joinpath('history.json')
        model_weights_path = root.joinpath('model.weights.h5')
        with open(summary_path, 'w') as f:
            encoder(image_shape, latent_dim).summary(print_fn=lambda x: f.write(x + '\n'))
            decoder(image_shape, latent_dim).summary(print_fn=lambda x: f.write(x + '\n'))
        history, model = trainVAEDiagonal(encoder_getter=encoder, decoder_getter=decoder, model_class=VAEDiagonal, batch_size=batch_size)
        model.save_weights(model_weights_path, overwrite=True)
        with open(history_path, 'w') as f:
            json.dump(history.history, f)


def train_VAEFullCovariance():
    backend = keras.backend.backend()

    root = pathlib.Path(f'./{backend}/VAEFullCovariance')
    root.mkdir(parents=True, exist_ok=True)

    history_path = root.joinpath('history.json')
    model_weights_path = root.joinpath('model.weights.h5')
    train_ds, val_ds, test_ds = get_datasets(batch_size=batch_size)
    encoder = get_encoder_full_covariance(image_shape=(28, 28, 1), latent_dim=2)
    decoder = get_decoder_full_covariance(image_shape=(28, 28, 1), latent_dim=2)
    model = VAEFullCovariance(encoder=encoder, decoder=decoder, num_mc_samples=1)
    model(keras.random.normal(shape=(1, 28, 28, 1)))
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, run_eagerly=keras.backend.backend() == 'tensorflow')
    early_stopping = EarlyStopping(patience=10)
    history = model.fit(train_ds, validation_data=val_ds, epochs=1000, callbacks=[early_stopping])

    model.save_weights(model_weights_path)
    with open(history_path, 'w') as f:
        json.dump(history.history, f)


if __name__ == '__main__':
    # train_VAEFullCovariance()
    train_all_VAEDiagonal()

    # BATCH_SIZE = 500
    # backend = keras.backend.backend()
    #
    # path_full_cov = pathlib.Path(f'./{backend}/VAEFullCovariance_{BATCH_SIZE}/history.json')
    # path_diag = pathlib.Path(f'./{backend}/VAEDiagonal_{BATCH_SIZE}/history.json')
    # with open(path_full_cov, 'r') as f:
    #     full_cov_history = json.load(f)
    # with open(path_diag, 'r') as f:
    #     diag_history = json.load(f)
    # plt.plot(full_cov_history['val_loss'], linewidth=0.8, label='Full Covariance')
    # plt.plot(diag_history['val_loss'], linewidth=0.8, label='Factorised Gaussian')
    # plt.legend()
    # plt.show()
