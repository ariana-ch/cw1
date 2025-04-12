# Define the encoder-decoder pairs
import keras.src.layers
from keras import ops
from keras.api.models import Model
from keras.api.layers import (Input, Dense, Reshape, Flatten, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization,
                              ReLU, UpSampling2D)
from typing import Tuple

REGULARIZER = keras.regularizers.L2()
INITIALIZER_RELU = keras.initializers.HeUniform()
INITIALIZER_SIGMOID = keras.initializers.GlorotNormal()


def get_decoder_v0(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(img_h * img_w * img_c, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_SIGMOID,
              activation='sigmoid')(h)
    outputs = Reshape((img_h, img_w, img_c))(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v0(image_shape: Tuple[int, int, int], latent_dim: int):
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
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)

    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


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
    h = Dense(64, activation='leaky_relu', kernel_regularizer=REGULARIZER,
              kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(128, activation='leaky_relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='leaky_relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(img_h * img_w * img_c, activation='sigmoid', kernel_initializer=INITIALIZER_SIGMOID)(h)
    output = Reshape((img_h, img_w, img_c))(h)
    return Model(inputs=inputs, outputs=output, name='decoder')


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
    h = Dense(256, activation='leaky_relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='leaky_relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(64, activation='leaky_relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER)(h)
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
    # (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    # h = Dense(64, activation='relu')(inputs)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(
        inputs)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(h)
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
    h = Conv2D(32, 3, strides=2, activation='relu', padding='same')(inputs)
    h = Conv2D(64, 3, strides=2, activation='relu', padding='same')(h)
    h = Flatten()(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v3(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    # (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    # h = Dense(64, activation='relu')(inputs)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(
        inputs)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 2, strides=2, activation='relu', padding='same')(h)
    outputs = Conv2DTranspose(1, 2, strides=2, activation='sigmoid', padding='same')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v3(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''

    inputs = Input(shape=image_shape)
    h = Conv2D(32, 2, strides=2, activation='relu', padding='same')(inputs)
    h = Conv2D(64, 2, strides=2, activation='relu', padding='same')(h)
    h = Flatten()(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v4(image_shape: Tuple[int, int, int], latent_dim: int):
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
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
                        kernel_initializer=INITIALIZER_RELU)(h)
    outputs = Conv2DTranspose(1, 3, strides=2, activation='sigmoid', padding='same', kernel_regularizer=REGULARIZER,
                              kernel_initializer=INITIALIZER_SIGMOID)(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v4(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''

    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Conv2D(64, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)
    h = Flatten()(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v5(image_shape: Tuple[int, int, int], latent_dim: int):
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
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
                        kernel_initializer=INITIALIZER_RELU)(h)
    outputs = Conv2DTranspose(1, 3, strides=2, activation='sigmoid', padding='same', kernel_regularizer=REGULARIZER,
                              kernel_initializer=INITIALIZER_SIGMOID)(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v5(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Conv2D(64, 3, strides=2, activation='relu', padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v6(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2DTranspose(1, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_SIGMOID)(h)
    h = BatchNormalization()(h)
    outputs = keras.layers.Activation('sigmoid')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v6(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2D(64, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v7(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu', kernel_initializer=INITIALIZER_RELU, kernel_regularizer=REGULARIZER)(inputs)
    h = Dense(128, activation='relu', kernel_initializer=INITIALIZER_RELU, kernel_regularizer=REGULARIZER)(h)
    h = Dense(256, activation='relu', kernel_initializer=INITIALIZER_RELU, kernel_regularizer=REGULARIZER)(h)
    h = Dense(7 * 7 * 64, activation='relu', kernel_initializer=INITIALIZER_RELU, kernel_regularizer=REGULARIZER)(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=1, activation=None, use_bias=False, padding='same',
                        kernel_initializer=INITIALIZER_RELU, kernel_regularizer=REGULARIZER)(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = UpSampling2D(2)(h)
    h = Conv2DTranspose(1, 3, strides=1, activation=None, use_bias=False, padding='same',
                        kernel_initializer=INITIALIZER_SIGMOID, kernel_regularizer=REGULARIZER)(h)
    h = BatchNormalization()(h)
    h = UpSampling2D(2)(h)
    outputs = keras.layers.Activation('sigmoid')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v7(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''

    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=1, activation=None, use_bias=False, padding='same')(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = MaxPool2D(2)(h)
    h = Conv2D(64, 3, strides=1, activation=None, use_bias=False, padding='same')(h)  # 14 → 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = MaxPool2D(2)(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(128, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v8(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(4 * 4 * 128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Reshape((4, 4, 128))(h)
    h = Conv2DTranspose(64, 2, strides=2, activation=None, use_bias=False, padding='same', output_padding=1,
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2DTranspose(32, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2DTranspose(1, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_SIGMOID)(h)
    h = BatchNormalization()(h)
    outputs = keras.layers.Activation('sigmoid')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v8(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)  # 28 -> 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2D(64, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)  # 14 -> 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2D(128, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)  # 14 -> 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_decoder_v9(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    # output size for Conv2D transpose:
    # out = (in − 1) x stride − 2 x padding + kernel_size + output_padding
    inputs = Input(shape=(latent_dim,))
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(inputs)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(512, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2DTranspose(1, 3, strides=2, activation=None, use_bias=False, padding='same',
                        kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_SIGMOID)(h)
    h = BatchNormalization()(h)
    outputs = keras.layers.Activation('sigmoid')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v9(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)  # 28 -> 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2D(64, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)  # 14 -> 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(512, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')


def get_encoder_full_covariance(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Updated version of our final encoder architecture (v9). The head of the
    model has been modified to output the mean vector and the l(l+1)/2 components of the
    lower triangular matrix L

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    inputs = Input(shape=image_shape)
    h = Conv2D(32, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(inputs)  # 28 -> 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2D(64, 3, strides=2, activation=None, use_bias=False, padding='same', kernel_regularizer=REGULARIZER,
               kernel_initializer=INITIALIZER_RELU)(h)  # 14 -> 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(512, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(256, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    h = Dense(128, activation='relu', kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)

    # UPDATED CODE:
    # h = Dense(2 * latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    # z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    z_mean = Dense(latent_dim, kernel_regularizer=REGULARIZER, kernel_initializer=INITIALIZER_RELU)(h)
    # lower triangular matrix has l (l+1)/2 components. For l=2 => 3 components
    L = Dense(int(latent_dim * (latent_dim + 1) / 2), kernel_regularizer=REGULARIZER,
              kernel_initializer=INITIALIZER_RELU)(h)
    L = BatchNormalization()(L)
    return Model(inputs=inputs, outputs=[z_mean, L], name='encoder')


def get_decoder_full_covariance(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Define this function for consistency. Simply returns our previous encoder for
    model 9.
    '''
    return get_decoder_v9(image_shape=image_shape, latent_dim=latent_dim)