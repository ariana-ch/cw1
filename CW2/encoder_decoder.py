# Define the decoder
from keras import ops
from keras.api.models import Model
from keras.api.layers import Input, Dense, Reshape, Flatten
from typing import Tuple

def get_decoder(image_shape: Tuple[int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(200, activation='relu')(inputs)
    h = Dense(img_h * img_w)(h)
    h = Reshape((img_h, img_w))(h)
    outputs = ops.sigmoid(h)

    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder(image_shape: Tuple[int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w) = image_shape
    inputs = Input(shape=(img_h, img_w))
    h = Flatten()(inputs)
    h = Dense(200, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)

    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')

if __name__ == '__main__':
    decoder = get_decoder((112, 112), 2)
    print(decoder.summary())
    encoder = get_encoder((112, 112), 2)
    print(encoder.summary())