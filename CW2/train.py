import json
import os
os.environ['KERAS_BACKEND'] = 'torch'
from CW2.vae_diagonal import VAEDiagonal
from CW2.encoder_decoder import get_encoder_v0 as get_encoder, get_decoder_v0 as get_decoder
from CW2.dataloader import get_datasets
from keras import ops
import keras
from keras.api.callbacks import EarlyStopping


def train(encoder_getter, decoder_getter, model_class, batch_size: int = 1024, latent_dim: int=2, num_mc_samples: int = 1):
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
    for data in train.take(1):
        image_shape = ops.shape(data)[1:]

    encoder = encoder_getter(image_shape=image_shape, latent_dim=latent_dim)
    decoder = decoder_getter(image_shape=image_shape, latent_dim=latent_dim)
    model = model_class(encoder=encoder, decoder=decoder, num_mc_samples=num_mc_samples)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    # optimizer = keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.001, epsilon=1e-6)
    model.compile(optimizer=optimizer, run_eagerly=keras.backend.backend() == 'tensorflow')
    _ = model(data) # force build the model
    early_stopping = EarlyStopping(patience=10)
    history = model.fit(train, validation_data=val, epochs=1000, callbacks=[early_stopping])
    return history, model


def do_train():
    import pathlib
    batch_size = 256
    import CW2.encoder_decoder as ed
    import matplotlib.pyplot as plt
    import json
    backend = keras.backend.backend()
    fig, ax = plt.subplots()
    _, _, test_data = get_datasets(batch_size)
    img_shape = (28, 28, 1)
    latent_dim = 2
    plots = pathlib.Path(f'./{backend}/validation_{batch_size}.png')
    for i in [9]:
        root = pathlib.Path(f'./{backend}/version_{str(i)}_{str(batch_size)}')
        root.mkdir(parents=True, exist_ok=True)
        encoder = getattr(ed, f'get_encoder_v{i}')
        decoder = getattr(ed, f'get_decoder_v{i}')

        summary_path = root.joinpath('summary.txt')
        history_path = root.joinpath('history.json')
        model_weights_path = root.joinpath('model.weights.h5')
        with open(summary_path, 'w') as f:
            encoder(img_shape, latent_dim).summary(print_fn=lambda x: f.write(x + '\n'))
            decoder(img_shape, latent_dim).summary(print_fn=lambda x: f.write(x + '\n'))
        history, model = train(encoder_getter=encoder, decoder_getter=decoder, model_class=VAEDiagonal, batch_size=batch_size)
        test_loss = model.evaluate(test_data)[0]
        ax.plot(history.history['val_loss'], label=f'Model {i}: {test_loss:.1f}')
        ax.legend()
        ax.set_title('Validation Loss')
        fig.savefig(plots)
        model.save_weights(model_weights_path, overwrite=True)

        with open(history_path, 'w') as f:
            json.dump(history.history, f)

def do_test():
    batch_size = 500
    import pathlib
    import matplotlib.pyplot as plt
    backend = keras.backend.backend()

    _, _, test_batch = get_datasets(batch_size)
    for i in range(9):
        import CW2.encoder_decoder as ed
        history_path = pathlib.Path(f'./{backend}/version_{str(i)}_{str(batch_size)}/history.json')
        path = pathlib.Path(f'./{backend}/version_{str(i)}_{str(batch_size)}/model.weights.h5')
        encoder = getattr(ed, f'get_encoder_v{i}')((28, 28, 1), 2)
        decoder = getattr(ed, f'get_decoder_v{i}')((28, 28, 1), 2)
        model = VAEDiagonal(encoder=encoder, decoder=decoder, num_mc_samples=1)
        model.build((None, 28, 28, 1))
        model.load_weights(path)
        model.compile()
        res = model.evaluate(test_batch)[0]
        print(f'version_{str(i)}_{str(batch_size)}: {res:.2f}')
        with open(history_path, 'r') as f:
            history = json.load(f)
        plt.plot(history['val_loss'], linewidth=1, label=f'Model {i}: {res:.2f}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    do_test()