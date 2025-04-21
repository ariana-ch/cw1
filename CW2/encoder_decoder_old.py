# Define the decoder
import keras.src.layers
from keras import ops
from keras.api.models import Model
from keras.api.layers import (Input, Dense, Reshape, Flatten, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization,
                              ReLU, UpSampling2D, AvgPool2D, Activation, Dropout)
from typing import Tuple



def get_decoder_v0(image_shape: Tuple[int, int], latent_dim: int):
    '''
    Get the decoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    # h = Dense(128, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2(0.01))(inputs)
    h = Dense(64, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2(0.01))(inputs)
    h = Dense(128, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2(0.01))(inputs)
    h = Dense(256, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2(0.01))(inputs)
    h = Dense(img_h * img_w * img_c)(h)
    h = Reshape((img_h, img_w, img_c))(h)
    outputs = ops.sigmoid(h)

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
    # h = Dense(512, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Dense(256, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Dense(128, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Dense(64, activation='leaky_relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Dense(2 * latent_dim)(h)
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
    h = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.L2())(inputs)
    # h = Dropout(0.5)(h)
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
    h = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.L2())(h)
    # h = Dropout(0.5)(h)
    h = Dense(2 * latent_dim, kernel_regularizer=keras.regularizers.L2())(h)
    h = BatchNormalization()(h)
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
    # h = Dense(64, activation='relu')(inputs)
    h = Dense(7 * 7 * 64, activation='relu')(inputs)
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
    # h = Dense(64, activation='relu')(h)
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
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2())(inputs)
    h = Dense(7 * 7 * 64, activation='relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Reshape((7, 7, 64))(h)
    h = Conv2DTranspose(32, 2, strides=2, activation='relu', padding='same', kernel_regularizer=keras.regularizers.L2())(h)  # 7 → 14
    outputs = Conv2DTranspose(1, 2, strides=2, activation='sigmoid', padding='same', kernel_regularizer=keras.regularizers.L2())(h)
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
    h = Conv2D(32, 2, strides=2, activation='relu', padding='same', kernel_regularizer=keras.regularizers.L2())(inputs)  # 28 → 14
    h = Conv2D(64, 2, strides=2, activation='relu', padding='same', kernel_regularizer=keras.regularizers.L2())(h)  # 14 → 7
    h = Flatten()(h)
    h = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2())(h)
    h = Dense(2 * latent_dim, kernel_regularizer=keras.regularizers.L2())(h)
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
    h = Dense(256, activation='relu')(inputs)
    h = Dense(14 * 14 * 64, activation='relu')(h)
    h = Reshape((14, 14, 64))(h)
    h = Conv2DTranspose(32, 3, strides=1, activation='relu', padding='same')(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = UpSampling2D(2)(h)
    outputs = Conv2DTranspose(1, 3, strides=1, activation='sigmoid', padding='same')(h)

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
    h = Conv2D(32, 3, strides=1, activation='relu', padding='same')(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = MaxPool2D(2)(h)
    h = Conv2D(64, 3, strides=1, activation='relu', padding='same')(h)  # 14 → 7
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
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
    h = Dense(128, activation='relu')(inputs)
    h = Dense(14 * 14 * 64, activation='relu')(h)
    h = Reshape((14, 14, 64))(h)
    h = Conv2DTranspose(32, 3, strides=1, activation=None, padding='same')(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = UpSampling2D(2)(h)
    h = Conv2DTranspose(1, 3, strides=1, activation=None, padding='same')(h)
    h = BatchNormalization()(h)
    outputs = Activation('sigmoid')(h)
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
    h = Conv2D(32, 3, strides=1, activation=None, padding='same')(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = MaxPool2D(2)(h)
    h = Conv2D(64, 3, strides=1, activation=None, padding='same')(h)  # 14 → 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(128, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
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
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(256, activation='relu')(h)
    h = Dense(14 * 14 * 32, activation='relu')(h)
    h = Reshape((14, 14, 32))(h)
    h = Conv2DTranspose(16, 3, strides=1, activation=None, use_bias=False, padding='same')(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = UpSampling2D(2)(h)
    h = Conv2DTranspose(1, 3, strides=1, activation=None, use_bias=False, padding='same')(h)
    h = BatchNormalization()(h)
    # h = Flatten()()
    # h = ReLU()(h)
    # h = UpSampling2D(2)(h)
    outputs = Activation('sigmoid')(h)
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
    h = Conv2D(16, 3, strides=1, activation=None, use_bias=False, padding='same')(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = MaxPool2D(2)(h)
    h = Conv2D(32, 3, strides=1, activation=None, use_bias=False, padding='same')(h)  # 14 → 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    # h = MaxPool2D(2)(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    h = Dense(2 * latent_dim)(h)
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
    (img_h, img_w, img_c) = image_shape
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(128, activation='relu')(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(7 * 7 * 32, activation='relu')(h)
    h = Reshape((7, 7, 32))(h)
    h = Conv2DTranspose(16, 3, strides=2, activation=None, use_bias=False, padding='same')(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    # h = UpSampling2D(2)(h)
    h = Conv2DTranspose(1, 3, strides=2, activation=None, use_bias=False, padding='same')(h)
    h = BatchNormalization()(h)
    # h = Flatten()()
    # h = ReLU()(h)
    # h = UpSampling2D(2)(h)
    outputs = Activation('sigmoid')(h)
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
    h = Conv2D(16, 3, strides=2, activation=None, use_bias=False, padding='same')(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    # h = MaxPool2D(2)(h)
    h = Conv2D(32, 3, strides=2, activation=None, use_bias=False, padding='same')(h)  # 14 → 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    # h = MaxPool2D(2)(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(128, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
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
    (img_h, img_w, img_c) = image_shape
    initializer = keras.initializers.LecunNormal()
    inputs = Input(shape=(latent_dim,))
    h = Dense(64, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(inputs)
    h = Dense(128, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(h)
    h = Dense(256, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(h)
    h = Dense(14 * 14 * 32, activation='relu', kernel_initializer=initializer)(h)
    h = Reshape((14, 14, 32))(h)
    h = Conv2DTranspose(16, 2, strides=1, use_bias=None, activation=None, padding='same', output_padding=1, kernel_initializer=initializer)(h)  # 7 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Conv2DTranspose(1, 2, strides=2, activation=None, use_bias=False, padding='same', kernel_initializer=initializer)(h)
    h = BatchNormalization()(h)
    outputs = Activation('sigmoid')(h)
    # outputs = Activation('sigmoid')(h)
    return Model(inputs=inputs, outputs=outputs, name='decoder')


def get_encoder_v8(image_shape: Tuple[int, int, int], latent_dim: int):
    '''
    Get the encoder model

    Args:
        image_shape: The shape of the images - output tensors
        latent_dim: The latent dimension - dim of the inputs

    Returns: A keras model
    '''
    initializer = keras.initializers.LecunNormal()
    inputs = Input(shape=image_shape)
    h = Conv2D(16, 2, strides=2, activation=None, use_bias=False, padding='same', kernel_initializer=initializer)(inputs)  # 28 → 14
    h = BatchNormalization()(h)
    h = ReLU()(h)

    h = Conv2D(32, 2, strides=1, activation=None, use_bias=False, padding='same', kernel_initializer=initializer)(h)  # 14 → 7
    h = BatchNormalization()(h)
    h = ReLU()(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(h)
    h = Dense(128, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(h)
    h = Dense(64, activation='relu', kernel_regularizer='l2', kernel_initializer=initializer)(h)
    h = Dense(2 * latent_dim)(h)
    z_mean, z_log_var = ops.split(h, indices_or_sections=2, axis=-1)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var], name='encoder')



if __name__ == '__main__':
    decoder = get_decoder_v8((28, 28, 1), 2)
    print(decoder.summary())
    encoder = get_encoder_v8((28, 28, 1), 2)
    print(encoder.summary())