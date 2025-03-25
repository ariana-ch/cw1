
from CW1.models import SimpleEmbeddingNet, SimpleEmbeddingNetV2, EfficientNet, EfficientNetPretrained
from CW1.helpers import get_path


def get_simple_embedding_5_flat_top2(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop2'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embedding_5_flat_top3(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop3'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embedding_5_flat_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model

def get_simple_embedding_5_pooling_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='pooling', no_blocks=5)
    name = 'SimpleEmbeddingNet5_PoolingTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embedding_6_flat_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='flatten', no_blocks=6)
    name = 'SimpleEmbeddingNet6_FlattenTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embedding_4_pooling_top_avg_pooling(best: bool = True):
    # Worse - no improvement and slow training despite the smaller model
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNet(top='pooling', no_blocks=4, pooling='avg')
    name = 'SimpleEmbeddingNet4_PoolingTop_AvgPooling'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model

def get_simple_embeddingV2_5_flat_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNetV2(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNetV2_5_FlattenTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embeddingV2_6_flat_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNetV2(top='flatten', no_blocks=6)
    name = 'SimpleEmbeddingNetV2_6_FlattenTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_simple_embeddingV2_5_pooling_top(best: bool = True):
    version = 'best' if best else 'final'
    model = SimpleEmbeddingNetV2(top='pooling', no_blocks=5)
    name = 'SimpleEmbeddingNetV2_5_PoolingTop'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_efficient_net(best: bool = True):
    version = 'best' if best else 'final'
    # Very bad performance - needs more data/training?
    model = EfficientNet((112, 112, 3), activation='swish')
    name = 'EfficientNet'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model


def get_efficient_net_pretrained(best: bool = True):
    version = 'best' if best else 'final'
    model = EfficientNetPretrained((112, 112, 3), freeze_weights=True)
    name = 'EfficientNetPretrained'
    name = f'{name}_{version}'
    path = get_path(name=name, directory='models', fmt='weights.h5', mkdir=False)
    model.load_weights(path)
    return name, model

