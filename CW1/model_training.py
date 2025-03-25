
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import time


from CW1.dataloaders import get_dataloader
from CW1.models import SimpleEmbeddingNetV2, SimpleEmbeddingNet, EfficientNet, EfficientNetPretrained
from CW1.helpers import ExponentialDecaySchedule, circle_loss_scheduled, get_path, circle_loss, recall_at_k
from CW1.triplet_mining import hard_negative, batch_all, batch_hard


# Common parameters:
# no_people = 50
# no_photos = 10
no_people = 32
no_photos = 8
epochs = 100
patience = 15
batch_length = (8000 - 32) // no_people # all IDs - 32 we hold for the validation data divided by the number of "batches" which have


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
        # loss = circle_loss(anchor_embeddings, positive_embeddings, negative_embeddings, m=0.25, gamma=64)
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


def train(model, name):
    print(f'\n\n---------------------Training {name}---------------------')
    log_path = get_path(name=name, directory='logs', fmt='csv')
    chkpt_path_best = get_path(name=f'{name}_best', directory='models', fmt='weights.h5')
    chkpt_path_final = get_path(name=f'{name}_final', directory='models', fmt='weights.h5')
    val_batch = next(iter(get_dataloader(shuffle=False, augment=False)))

    # start at epoch 10 and decay until epoch 30
    lr_schedule = ExponentialDecaySchedule(initial_lr=1e-3, t0=int(batch_length*10), t1=int(batch_length*40), final_factor=0.01)
    optimiser = keras.optimizers.Adam(learning_rate=lr_schedule)

    step = 0
    patience_counter = 0
    best_recall_at_1 = 0.0

    losses = []
    recalls = []
    epoch_val_recall = []
    epoch_val_loss = []
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
        epoch_val_loss.append(keras.ops.convert_to_numpy(val_loss))
        epoch_val_recall.append(keras.ops.convert_to_numpy(val_recall))
        epoch_array = np.repeat(np.arange(1, 1 + epoch), batch_length)

        df = pd.DataFrame(dict(epoch=epoch_array, training_losses=losses,
                               training_recalls=recalls,
                               validation_loss_epoch_end=np.repeat(epoch_val_loss, batch_length),
                               validation_recall_epoch_end=np.repeat(epoch_val_recall, batch_length)))

        df.to_csv(log_path, index=False)
        # fig = plt.figure(figsize=(14, 10))
        # plt.plot(df.training_recalls, linewidth=1, color='C0', label='Recall')
        # plt.plot(df.training_losses, linewidth=1, color='C1', label='Circle Loss')
        # plt.title(f'{name}')
        # plt.legend()
        # fig.savefig(get_path(name=name, directory='plots', fmt='png').as_posix())
        # plt.close(fig)

        print(f'\tValidation Recall@1: {keras.ops.convert_to_numpy(val_recall):.4f}, Validation Loss: {keras.ops.convert_to_numpy(val_loss):.4f}')

        if val_recall > best_recall_at_1:
            best_recall_at_1 = val_recall
            patience_counter = 0
            print(f'\t[{epoch}/{epochs}] New Best Recall@1: {keras.ops.convert_to_numpy(best_recall_at_1):.4f}. Saving model...')
            model.save_weights(chkpt_path_best, overwrite=True)
            if time.time() - start_time > 2 * 60 * 60: # Force early stopping after 1 hour(s)
                print(f'\t[{epoch}/{epochs}] Early stopping triggered after {(time.time() - start_time)/3600:.1f} hours.')
                model.save_weights(chkpt_path_final, overwrite=True)
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\t[{epoch}/{epochs}] Early stopping triggered after {patience_counter} epochs without improvement.')
                model.save_weights(chkpt_path_final, overwrite=True)
                break
        if time.time() - start_time > 3 * 60 * 60: # Force early stopping after 1.5 hour(s)
            print(f'\t[{epoch}/{epochs}] Early stopping triggered after {(time.time() - start_time)/3600:.1f} hours.')
            model.save_weights(chkpt_path_final, overwrite=True)
            break
    print('---------Training Completed------------')

    
    df = pd.DataFrame(dict(epoch=epoch_array, training_losses=losses,
                            training_recalls=recalls,
                            validation_loss_epoch_end=np.repeat(epoch_val_loss, batch_length),
                            validation_recall_epoch_end=np.repeat(epoch_val_recall, batch_length)))
    df.to_csv(log_path, index=False)
    # fig = plt.figure(figsize=(14, 10))
    # plt.plot(df.recalls, linewidth=1, color='C0', label='Recall')
    # plt.plot(df.losses, linewidth=1, color='C1', label='Circle Loss')
    # plt.title(f'{chkpt_path.name}')
    # plt.legend()
    # fig.savefig(get_path(name=name, directory='plots', fmt='png').as_posix())


def train_simple_embedding_5_flat_top2():
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop2'
    train(model=model, name=name)

def train_simple_embedding_5_flat_top3():
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop3'
    train(model=model, name=name)

def train_simple_embedding_5_flat_top():
    model = SimpleEmbeddingNet(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNet5_FlattenTop'
    train(model=model, name=name)

def train_simple_embedding_5_pooling_top():
    model = SimpleEmbeddingNet(top='pooling', no_blocks=5)
    name = 'SimpleEmbeddingNet5_PoolingTop'
    train(model=model, name=name)

def train_simple_embedding_6_flat_top():
    model = SimpleEmbeddingNet(top='flatten', no_blocks=6)
    name = 'SimpleEmbeddingNet6_FlattenTop'
    train(model=model, name=name)

def train_simple_embedding_4_pooling_top_avg_pooling():
    # Worse - no improvement and slow training despite the smaller model
    model = SimpleEmbeddingNet(top='pooling', no_blocks=4, pooling='avg')
    name = 'SimpleEmbeddingNet4_PoolingTop_AvgPooling'
    train(model=model, name=name)

def train_simple_embeddingV2_5_flat_top():
    model = SimpleEmbeddingNetV2(top='flatten', no_blocks=5)
    name = 'SimpleEmbeddingNetV2_5_FlattenTop'
    train(model=model, name=name)

def train_simple_embeddingV2_6_flat_top():
    model = SimpleEmbeddingNetV2(top='flatten', no_blocks=6)
    name = 'SimpleEmbeddingNetV2_6_FlattenTop'
    train(model=model, name=name)

def train_simple_embeddingV2_5_pooling_top():
    model = SimpleEmbeddingNetV2(top='pooling', no_blocks=5)
    name = 'SimpleEmbeddingNetV2_5_PoolingTop'
    train(model=model, name=name)

def train_efficient_net():
    # Very bad performance - needs more data/training?
    model = EfficientNet((112, 112, 3), activation='swish')
    name = 'EfficientNet'
    train(model=model, name=name)

def train_efficient_net_pretrained():
    model = EfficientNetPretrained((112, 112, 3), freeze_weights=True)
    name = 'EfficientNetPretrained'
    train(model=model, name=name)


if __name__ == '__main__':
    train_simple_embedding_5_flat_top3()