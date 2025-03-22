
import pandas as pd
from pathlib import Path
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from CW1.dataloading import get_dataloader
from CW1.models import SimpleEmbeddingNetV2, SimpleEmbeddingNet, EfficientNet, EfficientNetPretrained
from CW1.helpers import ExponentialDecaySchedule, hard_negative, batch_all, batch_hard, circle_loss


# Helper functions

from IPython.display import clear_output

def plot(loss_values, title):
    clear_output()
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label='Training loss', color='C0', linestyle='-')
    plt.xlabel('Batch Number')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.show()


# Helper functions

from IPython.display import clear_output

def plot(loss_values, title):
    clear_output()
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label='Training loss', color='C0', linestyle='-')
    plt.xlabel('Batch Number')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.show()

def circle_loss_scheduled(epoch, anchor_embeddings, positive_embeddings, negative_embeddings,
                          base_m=0.2, base_gamma=64, step_size=5):
    '''
    TensorFlow-friendly helper function to dynamically update the parameters in the circle loss.
    '''
    # Cast constants as tensors for safety (optional, but recommended)
    base_m = tf.constant(base_m, dtype=tf.float32)
    base_gamma = tf.constant(base_gamma, dtype=tf.float32)
    step_size = tf.constant(step_size, dtype=tf.int32)

    # Compute increments using TensorFlow operations
    steps = tf.math.floordiv(epoch, step_size)

    # m = base_m + steps * 0.02
    m_increment = tf.multiply(tf.cast(steps, tf.float32), 0.02)
    m = tf.add(base_m, m_increment)

    # gamma = base_gamma * (1 + steps * 0.5)
    gamma_increment = tf.multiply(tf.cast(steps, tf.float32), 0.5)
    gamma = tf.multiply(base_gamma, tf.add(1.0, gamma_increment))

    return circle_loss(anchor_embeddings, positive_embeddings, negative_embeddings, m, gamma)


def recall_at_k(embeddings, labels, k=1):
    """
    Compute Recall@K. Use to monitor the training instead of the loss function
    """
    sims = keras.ops.matmul(embeddings, keras.ops.transpose(embeddings))
    sims = sims - keras.ops.eye(keras.ops.shape(sims)[0]) * 1e9  # Exclude self-similarity

    top_k = keras.ops.top_k(sims, k=k).indices
    labels = keras.ops.expand_dims(labels, axis=1)
    top_k_labels = keras.ops.take(labels, top_k)

    positive_matches = keras.ops.equal(top_k_labels, labels)
    recall = keras.ops.mean(keras.ops.any(positive_matches, axis=1), axis=0)
    return recall


@tf.function(reduce_retracing=True)
def training_step(batch, model, optimizer, epoch):
    with tf.GradientTape() as tape:
        batch_labels, batch_images = batch
        embeddings = model(batch_images, training=True)
        # if epoch < EPOCH_THRESHOLD:
        #     triplets = hard_negative(embeddings=embeddings, labels=batch_labels)
        # else:
        #     triplets = batch_hard(embeddings=embeddings, labels=batch_labels)
        triplets = hard_negative(embeddings, batch_labels)
        anchor_embeddings = keras.ops.take(embeddings, triplets[:, 0], axis=0)
        positive_embeddings = keras.ops.take(embeddings, triplets[:, 1], axis=0)
        negative_embeddings = keras.ops.take(embeddings, triplets[:, 2], axis=0)

        loss = circle_loss_scheduled(epoch, anchor_embeddings, positive_embeddings, negative_embeddings)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, embeddings, batch_labels


@tf.function(reduce_retracing=True)
def validation_step(batch, model):
    batch_labels, batch_images = batch
    embeddings = model(batch_images, training=False)
    triplets = batch_all(embeddings=embeddings, labels=batch_labels)
    anchor_embeddings = keras.ops.take(embeddings, triplets[:, 0], axis=0)
    positive_embeddings = keras.ops.take(embeddings, triplets[:, 1], axis=0)
    negative_embeddings = keras.ops.take(embeddings, triplets[:, 2], axis=0)

    loss = circle_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
    recall = recall_at_k(embeddings, batch_labels, k=1)
    return loss, recall


def get_path(name, directory, fmt, postfix = ''):
    path = Path(f'./{directory}/{name}{postfix}.{fmt}')
    if path.exists():
        if postfix != '': postfix += 1
        else: postfix = 0
        return get_path(name=name, directory=directory, fmt=fmt, postfix=postfix)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

no_people = 50
no_photos = 10
epochs = 30
patience = 4


def train(model, name):
    log_path = get_path(name=name, directory='logs', fmt='csv')
    chkpt_path = get_path(name=name, directory='models', fmt='weights.h5')
    val_batch = next(iter(get_dataloader(shuffle=False, augment=False)))

    lr_schedule = ExponentialDecaySchedule(initial_lr=1e-3, t0=100, t1=20000, final_factor=0.01)
    optimiser = keras.optimizers.Adam(learning_rate=lr_schedule)


    patience_counter = 0
    step = 0
    best_recall_at_1 = 0.0

    losses = []
    recalls = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_dataloader = get_dataloader(no_people=no_people, no_photos=no_photos, shuffle=True, augment=True)

        epoch_losses = []
        epoch_recalls = []

        minibatch_losses = []
        minibatch_recalls = []

        for i, batch in enumerate(train_dataloader):
            step += 1
            epoch_tensor = tf.convert_to_tensor(epoch, dtype=tf.int32)
            loss_value, embeddings, labels = training_step(epoch=epoch_tensor,
                                                           batch=batch, model=model, optimizer=optimiser)
            recall_value = recall_at_k(embeddings, labels, k=1)
            loss_value, recall_value = keras.ops.convert_to_numpy(loss_value), keras.ops.convert_to_numpy(recall_value)
            minibatch_losses.append(loss_value)
            minibatch_recalls.append(recall_value)
            epoch_losses.append(loss_value)
            epoch_recalls.append(recall_value)
            losses.append(loss_value)
            recalls.append(recall_value)

            if i % 20 == 0:
                elapsed = time.time() - start_time
                elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed))

                running_loss = np.mean(minibatch_losses)
                running_recall = np.mean(minibatch_recalls)

                print(
                    f'\t[{epoch}: {i + 1}/{len(train_dataloader)} {elapsed}] Minibatch Loss: {running_loss:.4f}, Minibatch Recall@1: {running_recall:.4f}')
                minibatch_losses = []
                minibatch_recalls = []

        val_loss, val_recall = validation_step(val_batch, model)

        print(
            f'\tValidation Recall@1: {keras.ops.convert_to_numpy(val_recall):.4f}, Validation Loss: {keras.ops.convert_to_numpy(val_loss):.4f}')

        if val_recall > best_recall_at_1:
            best_recall_at_1 = val_recall
            patience_counter = 0
            print(
                f'\t[{epoch}/{epochs}] New Best Recall@1: {keras.ops.convert_to_numpy(best_recall_at_1):.4f}. Saving model...')
            model.save_weights(chkpt_path, overwrite=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f'\t[{epoch}/{epochs}] Early stopping triggered after {patience_counter} epochs without improvement.')
                break
    print('---------Training Completed------------')

    df = pd.DataFrame(dict(losses=losses, recalls=recalls))
    df.to_csv(log_path, index=False)
    fig = plt.figure(figsize=(14, 10))
    plt.plot(df.recalls, linewidth=1, color='C0', label='Recall')
    plt.plot(df.losses, linewidth=1, color='C1', label='Circle Loss')
    plt.title(f'{chkpt_path.name}')
    plt.legend()
    plt.show()

def train_simple_embedding_5_flat_top():
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop'
    train(model=model, name=name)

if __name__ == '__main__':
    train_simple_embedding_5_flat_top()