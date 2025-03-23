import tensorflow as tf
import keras
ACTIVATION = 'swish'


def padding(inputs, kernel_size):  # copied from EfficientNet because mine was getting the number wrong.
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """
    img_dim = 1
    input_size = inputs.shape[img_dim: (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def InvResBlock(activation=ACTIVATION, drop_rate=0.0, name='', filters_in=32, filters_out=16, kernel_size=3,
                strides=1, expand_ratio=1, se_ratio=0.0):
    '''
    Builds an inverted residual block with optional squeeze-and-excitation.

    :param activation: Activation function to use, e.g., 'relu', 'swish'. Default is ACTIVATION.
    :param drop_rate: Dropout rate for the dropout layer. Default is 0.0.
    :param name: Name prefix for the block. Default is an empty string.
    :param filters_in: Number of input filters (channels). Default is 32.
    :param filters_out: Number of output filters (channels). Default is 16.
    :param kernel_size: Size of the convolution kernel. Default is 3.
    :param strides: Stride of the convolution. Default is 1.
    :param expand_ratio: Expansion ratio for the block. Default is 1. (Must be INTEGER!!)
    :param se_ratio: Squeeze-and-excitation ratio. Default is 0.0.

    :return: A model that represents the inverted residual block. Chose this implemenation to be able to view
    the model summary in detail.
    '''
    name = f'{name.strip("_")}_' if name else ''

    inputs = keras.layers.Input(shape=(None, None, filters_in), name=f'{name}input')

    # Track current tensor
    x = inputs

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = keras.layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
                                kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                                name=f'{name}expansion_Conv2D')(x)
        x = keras.layers.BatchNormalization(name=f'{name}expansion_BN')(x)
        x = keras.layers.Activation(activation, name=f'{name}expand_activation')(x)

    # Depthwise Convolution
    if strides == 2:
        # manually add padding because we want the output dim to be the same and padding = same with stride = 2 would halve the output dim
        x = keras.layers.ZeroPadding2D(padding=padding(x, kernel_size), name=f'{name}DWConv_zeropadding')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding=conv_pad, use_bias=False,
                                     depthwise_initializer=keras.initializers.VarianceScaling(scale=2.0),
                                     name=f'{name}DWConv')(x)
    x = keras.layers.BatchNormalization(name=f'{name}DWConv_BN')(x)
    x = keras.layers.Activation(activation, name=f'{name}DWConv_activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = keras.layers.GlobalAveragePooling2D(name=f'{name}SE_squeeze')(x)
        se = keras.layers.Reshape((1, 1, filters), name=f'{name}SE_reshape')(se)
        se = keras.layers.Conv2D(filters_se, kernel_size=1, padding='same', activation=activation,
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                                 name=f'{name}SE_reduce')(se)
        se = keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation='sigmoid',
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                                 name=f'{name}SE_expand')(se)
        x = keras.layers.multiply([x, se], name=f'{name}SE_excite')

    # Projection phase
    x = keras.layers.Conv2D(filters_out, kernel_size=1, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0),
                            name=f'{name}project_conv')(x)
    x = keras.layers.BatchNormalization(name=f'{name}project_BN')(x)

    # Add skip connection
    if strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=f'{name}drop')(x)
        x = keras.layers.add([x, inputs], name=f'{name}add')
    return keras.models.Model(inputs, x, name=name.strip('_'))


def MBConv(filters_in=3, filters_out=16, expand_ratio=1.0, kernel_size=3, strides=1, drop_rate=0.0, starting_block=0,
           se_ratio=0.0, activation='swish', name='', num_blocks=1):
    '''
    Builds a mobile inverted residual block with multiple inverted residual blocks.

    :param filters_in: Number of input channels.
    :param filters_out: Number of output channels after each block.
    :param expand_ratio: Expansion factor for the block.
    :param kernel_size: Size of the convolution kernel.
    :param strides: Stride of the convolution.
    :param drop_rate: Drop rate factor (for each block, the actual drop rate is computed dynamically.
                      Deeper layers have higher drop rates).
    :param starting_block: The current inverted residual block number (used to compute the drop rate).
    :param se_ratio: Squeeze-and-excitation ratio. Default is 0.0.
    :param activation: Activation function to use. Default is 'swish' but light model uses 'relu6'.
    :param name: Name for the block.
    :param num_blocks: Number of inverted residual blocks to stack.
    :return: A Keras Model representing the mobile inverted residual block.
    '''
    name = f'{name.strip("_")}_' if name else ''

    inputs = keras.layers.Input(shape=(None, None, filters_in), name=f'{name}input')

    # Track current tensor
    x = inputs
    for i in range(num_blocks):
        block_strides = strides if i == 0 else 1
        filters_in = filters_in if i == 0 else filters_out
        drop_rate = drop_rate * (starting_block + i) / 16
        x = InvResBlock(activation=activation, drop_rate=drop_rate, name=f'{name}InvResNet{i + 1}',
                        filters_in=filters_in, filters_out=filters_out, kernel_size=kernel_size,
                        strides=block_strides, expand_ratio=expand_ratio, se_ratio=se_ratio
                        )(x)
    return keras.models.Model(inputs, x, name=name.strip('_'))


def EfficientNet(input_shape, embedding_dim=128, drop_connect_rate=0.2, depth_divisor=8, activation=ACTIVATION):
    def round_filters(filters, divisor=depth_divisor):
        '''Round number of filters based on depth multiplier.'''
        new_filters = max(
            divisor, int(filters + divisor / 2) // divisor * divisor
        )
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    inputs = keras.layers.Input(shape=input_shape, name='Input')

    # Stem
    x = keras.layers.Normalization(name='stem_normalization')(inputs)
    x = keras.layers.ZeroPadding2D(padding=padding(x, 3), name='stem_zeropadding')(x)
    x = keras.layers.Conv2D(round_filters(32), kernel_size=3, strides=2, padding='valid', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0), name='stem_conv')(x)
    x = keras.layers.BatchNormalization(name='stem_BN')(x)
    x = keras.layers.Activation(activation, name="stem_activation")(x)

    # MBConv Blocks (mobile inverted bottleneck convolutions)
    # Main architecture: 16 inverted residual blocks which form 7 mobile inverted bottleneck convolutions (MBConv)
    x = MBConv(filters_in=x.shape[-1], filters_out=16, name='MBConv1', drop_rate=drop_connect_rate, starting_block=0,
               kernel_size=3,
               expand_ratio=1, strides=1, se_ratio=0.25, num_blocks=1)(x)

    x = MBConv(filters_in=16, name='MBConv2', drop_rate=drop_connect_rate, starting_block=1, kernel_size=3,
               filters_out=24, expand_ratio=6, strides=2, se_ratio=0.25, num_blocks=2)(x)

    x = MBConv(filters_in=24, name='MBConv3', drop_rate=drop_connect_rate, starting_block=3, kernel_size=5,
               filters_out=40, expand_ratio=6, strides=2, se_ratio=0.25, num_blocks=2)(x)

    x = MBConv(filters_in=40, name='MBConv4', drop_rate=drop_connect_rate, starting_block=5, kernel_size=3,
               filters_out=80, expand_ratio=6, strides=2, se_ratio=0.25, num_blocks=3)(x)

    x = MBConv(filters_in=80, name='MBConv5', drop_rate=drop_connect_rate, starting_block=8, kernel_size=5,
               filters_out=112, expand_ratio=6, strides=2, se_ratio=0.25, num_blocks=3)(x)

    x = MBConv(filters_in=112, name='MBConv6', drop_rate=drop_connect_rate, starting_block=11, kernel_size=5,
               filters_out=192, expand_ratio=6, strides=2, se_ratio=0.25, num_blocks=4)(x)

    x = MBConv(filters_in=192, name='MBConv7', drop_rate=drop_connect_rate, starting_block=15, kernel_size=5,
               filters_out=320, expand_ratio=6, strides=1, se_ratio=0.25, num_blocks=1)(x)

    # Top/head
    x = keras.layers.Conv2D(round_filters(1280), kernel_size=1, padding='same', use_bias=False,
                            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0), name='head_conv')(x)
    x = keras.layers.BatchNormalization(name='head_BN')(x)
    x = keras.layers.Activation(activation, name='head_activation')(x)

    # flatten and embedding
    x = keras.layers.Dropout(0.2, name='head_dropout')(x)
    x = keras.layers.Flatten(name='embedding_flatten')(x)
    x = keras.layers.Dense(embedding_dim, name='embedding_fc')(x)

    # L2 normalization for embedding vectors
    outputs = keras.layers.UnitNormalization(name='embedding_normalisation')(x)

    return keras.models.Model(inputs, outputs, name='EfficientNet0_Replication')


def SimpleEmbeddingNet(input_shape=(112, 112, 3), embedding_dim=128, activation='swish', pooling='max', no_blocks=4,
                       top='flatten'):
    '''
    Simple keras model which progressively does dimensionality reduction via a series of convolutions and pooling layers.

    Args:
        input_shape: Tuple specifying the input shape (height, width, channels).
        embedding_dim: Dimensionality of the embedding vector.
        activation: Activation function to use in the model.
        pooling: Type of pooling to use, either 'max' or 'avg'.
        no_blocks: Number of blocks to use in the model.
        top: Type of top layer to use, either 'flatten' or 'pooling'.

    Returns:
        A Keras Sequential model.
    '''
    initialiser = keras.initializers.VarianceScaling(scale=0.2)
    pooling = keras.layers.MaxPool2D if pooling == 'max' else keras.layers.AvgPool2D

    channels = 32
    model = keras.Sequential([keras.layers.Input(shape=input_shape)],
                             name=f'SimpleEmbeddingNet{no_blocks}_{top.title()}Top')
    for _ in range(min(2, no_blocks)):
        model.add(keras.layers.Conv2D(channels, kernel_size=5, strides=1, padding='same', activation=None,
                                      kernel_initializer=initialiser, use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation))
        model.add(pooling(2))
        channels *= 2
    for _ in range(max(0, no_blocks - 2)):
        model.add(keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding='same', activation=None,
                                      kernel_initializer=initialiser, use_bias=False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation))
        model.add(pooling(2))
        channels *= 2

    # head/top
    if top == 'flatten':
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(embedding_dim, use_bias=False, kernel_initializer=initialiser,
                               kernel_regularizer=keras.regularizers.l2(1e-4)))
        model.add(keras.layers.BatchNormalization())
    else:
        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Dense(embedding_dim, use_bias=False, kernel_initializer=initialiser,
                               kernel_regularizer=keras.regularizers.l2(1e-4)))
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.UnitNormalization())
    return model


def SimpleEmbeddingNetV2(input_shape=(112, 112, 3), embedding_dim=128, activation='swish', pooling='max', no_blocks=4,
                         top='flatten'):
    '''
    Simple keras model which progressively does dimensionality reduction via a series of convolutions with stride = 2.

    Args:
        input_shape: Tuple specifying the input shape (height, width, channels).
        embedding_dim: Dimensionality of the embedding vector.
        activation: Activation function to use in the model.
        pooling: Type of pooling to use, either 'max' or 'avg'.
        no_blocks: Number of blocks to use in the model.
        top: Type of top layer to use, either 'flatten' or 'pooling'.
    '''

    initialiser = keras.initializers.VarianceScaling(2.0)
    pooling = keras.layers.MaxPool2D if pooling == 'max' else keras.layers.AvgPool2D
    inputs = keras.Input(shape=input_shape)

    # Block 1: 112 -> 56
    x = keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, kernel_initializer=initialiser)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    channels = 64
    for i in range(max(0, no_blocks - 1)):
        x = keras.layers.Conv2D(channels, 3, strides=2, padding='same', use_bias=False, kernel_initializer=initialiser)(
            x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)
        channels *= 2

        # head/top
    if top == 'flatten':
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(embedding_dim, use_bias=False, kernel_initializer=initialiser,
                               kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.BatchNormalization()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(embedding_dim, use_bias=False, kernel_initializer=initialiser,
                               kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.UnitNormalization()(x)
    model = keras.Model(inputs, outputs, name=f'SimpleEmbeddingV2_{no_blocks}_{top.title()}Top')
    return model

def EfficientNetPretrained(input_shape, embedding_dim=128, freeze_weights: bool = True):
    '''
    Use the pretrained EfficientNet0 model and adapt it to learn our embeddings
    '''
    base_model = keras.applications.efficientnet.EfficientNetB0(
        weights='imagenet',
        include_top=False,  # Exclude the classification head
        input_shape=input_shape,
        pooling=None)

    base_model.trainable = not freeze_weights

    inputs = keras.layers.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_weights)
    # pooling and embedding
     # flatten and embedding
    x = keras.layers.Dropout(0.2, name='head_dropout')(x)
    x = keras.layers.Flatten(name='embedding_flatten')(x)
    x = keras.layers.Dense(embedding_dim, name='embedding_fc')(x)

    # L2 normalization for embedding vectors
    outputs = keras.layers.UnitNormalization(name='embedding_normalisation')(x)
    return keras.models.Model(inputs, outputs, name='EfficientNet0_Pretrained')
