# Define the decoder
import keras.src.layers
from keras import ops
from keras.api.models import Model
from keras.api.layers import (Input, Dense, Reshape, Flatten, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization,
                              ReLU, UpSampling2D)
from typing import Tuple

from keras.src.ops import BatchNorm

from CW1.models import ACTIVATION


def get_decoder_v1(image_shape: Tuple[int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(200, activation='relu')(inputs)
    h = Dense(img_h * img_w * img_c)(h)
    h = Reshape((img_h, img_w, img_c))(h)
    outputs = ops.sigmoid(h)

    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v1(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(img_h, img_w, img_c))
    h = Flatten()(inputs)
    h = Dense(200, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)

    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v2(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(7 * 7 * 64, activation='relu')(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(h)  # 7 → 14
    outputs = Conv2DTranspose(1, 3, strides=2, activation='sigmoid', padding='same')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v2(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''

    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation='relu', padding='same')(inputs)  # 28 → 14
    h = Conv2D(64, 3, strides=2, activation='relu', padding='same')(h)  # 14 → 7
    h = Flatten()(h)
    h = Dense(64, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')




if __name__ == '__main__':
    decoder = get_decoder_v2((28, 28, 1), 2)
    print(decoder.summary())
    encoder = get_encoder_v2((28, 28, 1), 2)
    print(encoder.summary())