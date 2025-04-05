import tensorflow_datasets as tfds
import tensorflow as tf
from enum import Enum
from tree import flatten


def prepare_dataset(ds, batch_size: int = 64, shuffle: bool = False):
    '''
    Prepare the dataset `ds` for training or evaluation.
    Args:
        ds: A tensorflow PrefetchDataset
        batch_size: The batch size to partition the dataset into
        shuffle: If True, the dataset is shuffled

    Returns: Tensorflow DataSet
    '''
    ds = ds.map(lambda x: tf.cast(x['image'], 'float32')) # originally tf.uint8
    if shuffle:
        ds = ds.shuffle(buffer_size=len(ds))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_datasets(batch_size: int = 64):
    '''
    Load and preprocess the MNIST dataset(s)
    Args:
        batch_size: The batch size

    Returns: Tensorflow dataset or tuple of Tensorflow datasets
    '''
    train, val, test = tfds.load('binarized_mnist', data_dir='data', split=['train', 'validation', 'test'])
    train = prepare_dataset(train, batch_size=batch_size, shuffle=True)
    val = prepare_dataset(val, batch_size=batch_size, shuffle=False)
    test = prepare_dataset(test, batch_size=batch_size, shuffle=False)
    return train, val, test


def test_get_dataset():
    train, val, test = get_datasets()
    print(len(train))
    print(len(val))
    print(len(test))
    print(type(train))
    for ds in train.take(1):
        print(ds.shape)
        print(ds.dtype)


if __name__ == '__main__':
    test_get_dataset()
