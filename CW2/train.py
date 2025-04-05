from CW2.vae_diagonal import VAEDiagonal
from CW2.encoder_decoder import get_encoder, get_decoder
from CW2.dataloader import get_datasets
from keras import ops
import keras
from keras.api.callbacks import EarlyStopping


def train(encoder_getter, decoder_getter, model_class, batch_size: int = 64, latent_dim: int=2, num_mc_samples: int = 1):
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

    train, val, test = get_datasets(batch_size=64)
    for data in train.take(1):
        image_shape = ops.shape(data)[1:]

    encoder = get_encoder(image_shape=image_shape, latent_dim=latent_dim)
    decoder = get_decoder(image_shape=image_shape, latent_dim=latent_dim)
    model = model_class(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    model.compile(optimizer='adam')
# encoder = get_decoder()
# vae = VAE(encoder, decoder, name='vae')
# # Compile and fit the Model
#
#
    early_stopping = EarlyStopping(patience=10)
#
# vae.compile(optimizer='adam')
    history = model.fit(train, validation_data=val, epochs=10, callbacks=[early_stopping])
    return history, model


if __name__ == '__main__':
    print(image_shape)