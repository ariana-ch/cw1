# Torch DataLoaders appear to be slightly
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from typing import Optional
# You will need the following imports for this assessment. You can make additional imports when you need them
import os
from functools import partial
from pathlib import Path
import keras
import matplotlib.pyplot as plt
import numpy as np
from CW1.models.models import SimpleEmbeddingNetV2, ExponentialDecaySchedule
print(keras.backend.backend())


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, no_photos, img_transforms: Optional[list], augment: bool = False, test: bool = False):
        '''
         Custom dataset class to sample an identity and return no_photos of that identity
        :param no_photos: Integer, the number of images to return per sample
        :param img_transforms: optional list of image transformations
        :param augment: Whether the images should be transformed. Note that at test time
        any transforms should be deterministic
        :para test: If test is False, the identities are selected from the first 1768 (=8000-32) of the identities and they are
        shuffled before being drawn. If true then the last 32 identities are used
        '''
        self.root_dir = Path('./data/casia-webface/')
        self.no_photos = no_photos
        if img_transforms:
            self.extra_transforms = img_transforms
        else:
            self.extra_transforms = []

        default_img_gen_transf = [transforms.RandomHorizontalFlip(p=0.5),  # Flipping images
                                  transforms.Pad(10),  # Zero padding (common in person re-ID)
                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                                         hue=0.1),  # Color variations
                                  transforms.RandomPerspective(distortion_scale=0.5,
                                                               p=0.3),
                                  transforms.Resize((112, 112))]
        self.default_transforms = [transforms.ToTensor(),
                                   # -> torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                   transforms.ConvertImageDtype(torch.float32),
                                   transforms.Resize((112, 112))]
        self.transforms = transforms.Compose(self.default_transforms + self.extra_transforms)
        self.img_generating_transforms = transforms.Compose(
            self.default_transforms + (default_img_gen_transf or self.extra_transforms))
        self.augment = augment or img_transforms
        self.__set_labels(test=test)

    def __set_labels(self, test: bool):
        labels = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        if test:
            self.labels = labels[-32:]
        else:
            labels = labels[:-32]
            np.random.shuffle(labels)
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        identity = self.labels[idx]
        img_paths = [os.path.join(self.root_dir, identity, x) for x in
                     os.listdir(os.path.join(self.root_dir, identity))]

        if len(img_paths) >= self.no_photos:
            img_paths = np.random.choice(img_paths, size=self.no_photos, replace=False)
            extra = 0
        else:
            extra = self.no_photos - len(img_paths)

        img_list = []
        for img_path in img_paths:
            img = Image.open(img_path)  # already in RGB here.
            img_tensor = self.transforms(img)
            img_list.append(img_tensor)

        if extra:
            for i in range(extra):
                img_path = np.random.choice(img_paths)
                img = Image.open(img_path)
                img_tensor = self.img_generating_transforms(img)
                img_list.append(img_tensor)

        img_tensor = torch.stack(img_list, dim=0)  # shape N, C, H, W
        img_tensor = img_tensor.permute(0, 2, 3, 1)
        label_tensor = torch.from_numpy(np.repeat(idx, self.no_photos).astype(np.int32))
        return label_tensor, img_tensor


def collate_fn(batch):
    '''
    Custom collate function to be used with the dataloader
    :param batch: batch of data
    :return: collated batch
    '''
    labels, images = zip(*batch)
    labels = torch.cat(labels, dim=0)
    images = torch.cat(images, dim=0)
    return labels, images


# overwrite the previous method - not clean code but this is a notebook so ...
def get_dataloader(no_people: int = 32, no_photos: int = 4, img_transforms: Optional[list] = None,
                   shuffle: bool = False, augment: bool = False, collate_fn=collate_fn, **kwargs):
    '''
        Function to get the dataloader for the dataset

    :param no_people: Number of people to sample
    :param no_photos: Number of photos to return per person
    :param img_transforms: an optional list of transformations to be applied to the image (after it is converted to a tensor so ensure compatibility)
    :param shuffle: If true, samples are shuffled
    :param augment: If True the images are augmented
    :param collate_fn: Collate function. This is used to transform the shape of the batch. By default, given our dataset, the batches would consist of a 2d tuple with dimensions:
        (no_people, no_photos) <-- labels
        (no_people, no_photos, 112, 112, 3) <-- images
    both are fine since we are anyway writing a custom training loop, but we added the collate function to be in line with the exercise
    :return: the dataloader to be used for training the model
    '''
    dataset = ImageDataset(no_photos=no_photos, img_transforms=img_transforms, augment=augment, test=not shuffle)
    return torch.utils.data.DataLoader(dataset, batch_size=no_people, shuffle=shuffle, drop_last=True,
                                       collate_fn=collate_fn, **kwargs)


def scaled_cosine_similarity(a, b):
    '''
    Computes the scaled cosine similarity between two (batches) of vectors, row-wise
    :param a: Tensor of shape (N, d) (l2 normalised embeddings)
    :param b: Tensor of shape (N, d) (l2 normalised embeddings)
    :return: Tensor of shape (N,) containing scaled cosine similarities between each pair.
    '''
    cos_sim = keras.ops.sum(a * b, axis=-1)
    return (cos_sim + 1) / 2


def circle_loss(anchor_embeddings, positive_embeddings, negative_embeddings,
                m=0.2, gamma=256):
    '''
    Computes the Circle Loss as described in the assignment, using Keras ops.

    :param anchor_embeddings: Tensor of shape (N, d), l2-normalised embeddings for anchor images.
    :param positive_embeddings: Tensor of shape (N, d), l2-normalised embeddings for positive images.
    :param negative_embeddings: Tensor of shape (N, d), l2-normalised embeddings for negative images.
    :param m: Margin parameter, typically between 0 and 1.
    :param gamma: Scaling hyperparameter in the circle loss.
    :return: Scalar tensor representing the mean circle loss over the batch.
    '''
    anchor_embeddings = keras.ops.normalize(anchor_embeddings, axis=-1)
    positive_embeddings = keras.ops.normalize(positive_embeddings, axis=-1)
    negative_embeddings = keras.ops.normalize(negative_embeddings, axis=-1)

    # Similarities (scaled cosine similarity between corresponding rows)
    sp = scaled_cosine_similarity(anchor_embeddings, positive_embeddings)
    sn = scaled_cosine_similarity(anchor_embeddings, negative_embeddings)

    # Compute alpha and delta terms
    alpha_p = keras.ops.relu(1 + m - sp)  # ensures min=0
    alpha_n = keras.ops.relu(sn + m)
    delta_p = 1 - m
    delta_n = m

    logit_p = - alpha_p * (sp - delta_p)
    logit_n = alpha_n * (sn - delta_n)
    logits = logit_p + logit_n
    loss = (1.0 / gamma) * keras.ops.softplus(logits * gamma)
    # Return the mean loss over the batch
    return keras.ops.mean(loss)
import tensorflow as tf


@tf.function
def scaled_cosine_similarity_matrix(embeddings):
    '''
    Computes the scaled cosine similarity matrix for all pairs in the batch. (inspired by the sklearn library)

    :param embeddings: Tensor of shape (N, d), assumed to be L2-normalised.
    :return: Tensor of shape (N, N), scaled cosine similarities between each pair.
    '''
    sim_matrix = keras.ops.matmul(embeddings, keras.ops.transpose(embeddings))
    scaled_similarity = (sim_matrix + 1.0) / 2.0
    return scaled_similarity


@tf.function
def batch_all(embeddings, labels, *args, **kwargs):
    '''
    Returns all valid triplets from the batch.

    Valid triplet:
      - anchor and positive share the same label (excluding self-pair)
      - negative has a different label

    :param embeddings: Tensor of shape (N, d), L2-normalised.
    :param labels: Tensor of shape (N,), class labels.
    :return: Tensor of (anchor_idx, positive_idx, negative_idx) triplets.
    '''
    labels = keras.ops.array(labels)
    N = labels.shape[0]

    triplets = tf.TensorArray(dtype=tf.int32, size=N * (N - 4) * 3, dynamic_size=True)
    triplet_count = tf.constant(0)

    def loop_body(anchor_idx, triplets, triplet_count):
        anchor_label = labels[anchor_idx]
        positive = keras.ops.where((labels == anchor_label) & (keras.ops.arange(N) != anchor_idx))[0]
        negative = keras.ops.where(labels != anchor_label)[0]

        def positive_loop_body(pos_idx_index, triplets, triplet_count):
            pos_idx = positive[pos_idx_index]

            def negative_loop_body(neg_idx_index, triplets, triplet_count):
                neg_idx = negative[neg_idx_index]
                triplets = triplets.write(triplet_count, tf.stack([anchor_idx, pos_idx, neg_idx]))
                triplet_count += 1
                return neg_idx_index + 1, triplets, triplet_count

            _, triplets, triplet_count = tf.while_loop(
                cond=lambda neg_idx_index, *_: neg_idx_index < keras.ops.shape(negative)[0],
                body=negative_loop_body,
                loop_vars=[tf.constant(0), triplets, triplet_count]
            )

            return pos_idx_index + 1, triplets, triplet_count

        _, triplets, triplet_count = tf.while_loop(
            cond=lambda pos_idx_index, *_: pos_idx_index < keras.ops.shape(positive)[0],
            body=positive_loop_body,
            loop_vars=[tf.constant(0), triplets, triplet_count]
        )
        return anchor_idx + 1, triplets, triplet_count

    _, triplets, triplet_count = tf.while_loop(cond=lambda anchor_idx, *_: anchor_idx < N,
                                               body=loop_body,
                                               loop_vars=[tf.constant(0), triplets, triplet_count])
    stacked_triplets = triplets.stack()
    valid_triplets = tf.slice(stacked_triplets, [0, 0], [triplet_count, 3])
    return valid_triplets


@tf.function
def semi_hard(embeddings, labels, margin=0.01):
    """Selects semi-hard triplets using tf.while_loop."""
    labels = keras.ops.array(labels)
    sim_matrix = scaled_cosine_similarity_matrix(embeddings)
    N = labels.shape[0]
    triplets = tf.TensorArray(dtype=tf.int32, size=N * (N - 4) * 3, dynamic_size=True)
    triplet_count = tf.constant(0)

    def loop_body(anchor_idx, triplets, triplet_count):
        anchor_label = labels[anchor_idx]
        positive = keras.ops.where((labels == anchor_label) & (keras.ops.arange(N) != anchor_idx))[0]
        negative = keras.ops.where(labels != anchor_label)[0]
        pos_sims = keras.ops.take(sim_matrix[anchor_idx, :], indices=positive)
        neg_sims = keras.ops.take(sim_matrix[anchor_idx, :], indices=negative)

        def positive_loop_body(i, triplets, triplet_count):
            pos_idx = positive[i]
            s_ap = pos_sims[i]
            condition = keras.ops.where((neg_sims > s_ap) & (neg_sims < s_ap + margin))[0]

            def negative_loop_body(j, triplets, triplet_count):
                triplets = triplets.write(triplet_count, tf.stack([anchor_idx, pos_idx, negative[j]]))
                triplet_count += 1
                return j + 1, triplets, triplet_count

            _, triplets, triplet_count = tf.while_loop(
                cond=lambda j, *_: j < keras.ops.shape(condition)[0],
                body=negative_loop_body,
                loop_vars=[tf.constant(0), triplets, triplet_count]
            )
            return i + 1, triplets, triplet_count

        _, triplets, triplet_count = tf.while_loop(
            cond=lambda i, *_: i < keras.ops.shape(positive)[0],
            body=positive_loop_body,
            loop_vars=[tf.constant(0), triplets, triplet_count]
        )

        return anchor_idx + 1, triplets, triplet_count

    _, triplets, triplet_count = tf.while_loop(cond=lambda anchor_idx, *_: anchor_idx < N, body=loop_body,
                                               loop_vars=[tf.constant(0), triplets, triplet_count])
    stacked_triplets = triplets.stack()
    valid_triplets = tf.slice(stacked_triplets, [0, 0], [triplet_count, 3])
    return valid_triplets


@tf.function
def batch_hard(embeddings, labels, *args, **kwargs):
    '''
    Selects the hardest positive and hardest negative for each anchor.

    From Hermans et al. (2017):
      - Hardest positive: min similarity (hardest to pull closer)
      - Hardest negative: max similarity (easiest to push away)

    :param embeddings: Tensor of shape (N, d), L2-normalised.
    :param labels: Tensor of shape (N,), class labels.
    :return: Tensor of (anchor_idx, positive_idx, negative_idx) triplets.
    '''
    labels = keras.ops.array(labels)
    sim_matrix = scaled_cosine_similarity_matrix(embeddings)
    N = labels.shape[0]
    triplets = tf.TensorArray(dtype=tf.int32, size=N * 3 * (N - 4), dynamic_size=True)
    triplet_count = tf.constant(0)

    def loop_body(anchor_idx, triplets, triplet_count):
        anchor_label = labels[anchor_idx]
        positive = keras.ops.where((labels == anchor_label) & (keras.ops.arange(N) != anchor_idx))[0]
        negative = keras.ops.where(labels != anchor_label)[0]
        pos_sims = keras.ops.take(sim_matrix[anchor_idx, :], indices=positive)
        neg_sims = keras.ops.take(sim_matrix[anchor_idx, :], indices=negative)
        hard_pos_idx = positive[keras.ops.argmin(pos_sims)]
        hard_neg_idx = negative[keras.ops.argmax(neg_sims)]
        triplets = triplets.write(triplet_count, tf.stack([anchor_idx, hard_pos_idx, hard_neg_idx]))
        triplet_count += 1
        return anchor_idx + 1, triplets, triplet_count

    _, triplets, triplet_count = tf.while_loop(cond=lambda anchor_idx, *_: anchor_idx < N, body=loop_body,
                                               loop_vars=[tf.constant(0), triplets, triplet_count])
    stacked_triplets = triplets.stack()
    valid_triplets = tf.slice(stacked_triplets, [0, 0], [triplet_count, 3])
    return valid_triplets


@tf.function
def hard_negative(embeddings, labels, *args, **kwargs):
    '''
    Uses all positive pairs for an anchor, and selects the hardest negative.

    - Equation (Hard Negative): argmax_n(similarity(a, n))
    - From FaceNet (Schroff et al., 2015)

    :param embeddings: Tensor of shape (N, d), L2-normalised.
    :param labels: Tensor of shape (N,), class labels.
    :return: Tensor of (anchor_idx, positive_idx, negative_idx) triplets.
    '''
    labels = keras.ops.array(labels)
    sim_matrix = scaled_cosine_similarity_matrix(embeddings)
    N = labels.shape[0]
    triplets = tf.TensorArray(dtype=tf.int32, size=N * (N - 4) * 3, dynamic_size=True)
    triplet_count = tf.constant(0)

    def loop_body(anchor_idx, triplets, triplet_count):
        anchor_label = labels[anchor_idx]
        positive = keras.ops.where((labels == anchor_label) & (keras.ops.arange(N) != anchor_idx))[0]
        negative = keras.ops.where(labels != anchor_label)[0]

        neg_sims = keras.ops.take(sim_matrix[anchor_idx, :], indices=negative)
        neg_idx = negative[keras.ops.argmax(neg_sims)]

        def positive_loop_body(pos_idx_index, triplets, triplet_count):
            pos_idx = positive[pos_idx_index]
            triplets = triplets.write(triplet_count, tf.stack([anchor_idx, pos_idx, neg_idx]))
            triplet_count += 1
            return pos_idx_index + 1, triplets, triplet_count

        _, triplets, triplet_count = tf.while_loop(
            cond=lambda pos_idx_index, *_: pos_idx_index < keras.ops.shape(positive)[0],
            body=positive_loop_body,
            loop_vars=[tf.constant(0), triplets, triplet_count]
        )

        return anchor_idx + 1, triplets, triplet_count

    _, triplets, triplet_count = tf.while_loop(cond=lambda anchor_idx, *_: anchor_idx < N, body=loop_body,
                                               loop_vars=[tf.constant(0), triplets, triplet_count])
    stacked_triplets = triplets.stack()
    valid_triplets = tf.slice(stacked_triplets, [0, 0], [triplet_count, 3])
    return valid_triplets


def get_triplets(batch, model, method: str = 'semi_hard', **kwargs):
    '''
    Function that returns a list of triplet indices, as per specs above, using the method selected
    :param batch: A batch of images
    :param model: The model used to get the image embeddings
    :param method: The mining method to use
    :return: A list of tuples, each of length 3 with (a_idx, p_idx, n_idx)
    '''
    embeddings = model(batch[1])
    labels = batch[0]
    fn = dict(semi_hard=semi_hard, batch_hard=batch_hard, batch_all=batch_all, hard_negative=hard_negative)[method]
    res = fn(embeddings=embeddings, labels=labels, **kwargs)
    return list(map(tuple, res.numpy().tolist()))

def get_path(name, directory, fmt, postfix = ''):
    path = Path(f'./{directory}/{name}{postfix}.{fmt}')
    if path.exists():
        if postfix != '': postfix += 1
        else: postfix = 0
        return get_path(name=name, directory=directory, fmt=fmt, postfix=postfix)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

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

def circle_loss_scheduler(epoch, base_m=0.2, base_gamma=64, step_size=5):
    '''
    Helper function to dynamically update the parameters in the circle loss.
    As the training progresses, we can increase the values of m and gamma
    '''
    m = base_m + (epoch // step_size) * 0.02
    gamma = base_gamma * (1 + (epoch // step_size) * 0.5)
    return partial(circle_loss, m=m, gamma=gamma)

def triplet_function_scheduler(epoch, embeddings, labels):
    if epoch < 1:
        return semi_hard(embeddings=embeddings, labels=labels)
    elif epoch < 2:
        return hard_negative(embeddings=embeddings, labels=labels)
    else:
        return batch_hard(embeddings=embeddings, labels=labels)


if __name__ == '__main__':
    import json

    patience_counter = 0
    name = 'SimpleEmbeddingNetV2'
    log_path = get_path(name=name, directory='logs', fmt='csv')
    chkpt_path = get_path(name=name, directory='models', fmt='weights.h5')
    meta_path = get_path(name=name, directory='meta', fmt='txt')
    val_batch = next(iter(get_dataloader(shuffle=False, augment=False)))

    lr_schedule = ExponentialDecaySchedule(initial_lr=0.5 * 1e-3, t0=100, t1=20000, final_factor=0.001)
    optimiser = keras.optimizers.Adam(learning_rate=lr_schedule)
    triplet_fn = hard_negative
    no_people = 50
    no_photos = 5
    epochs = 20
    patience = 3
    model = SimpleEmbeddingNetV2(activation='swish')

    meta_data = dict(model=model.name, patience=patience, epochs=epochs, triplet_fn=triplet_fn.__name__,
                     optimiser=optimiser.name, learning_rate=optimiser.learning_rate.numpy().tolist(),
                     loss_fn_kwargs=dict(mu='default', gamma='default'), model_path=chkpt_path.name,
                     learning_schedule=lr_schedule.__dict__, circle_loss_scheduler='default')
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f)


    @tf.function  # (reduce_retracing=True)
    def training_step(batch, model, optimizer, triplet_fn, loss_fn):
        with tf.GradientTape() as tape:
            batch_labels, batch_images = batch
            embeddings = model(batch_images, training=True)
            triplets = triplet_fn(embeddings=embeddings, labels=batch_labels)

            anchor_embeddings = keras.ops.take(embeddings, triplets[:, 0], axis=0)
            positive_embeddings = keras.ops.take(embeddings, triplets[:, 1], axis=0)
            negative_embeddings = keras.ops.take(embeddings, triplets[:, 2], axis=0)

            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, embeddings, batch_labels


    @tf.function
    def validation_step(batch, model, loss_fn):
        batch_labels, batch_images = batch
        embeddings = model(batch_images, training=False)

        triplets = triplet_fn(embeddings=embeddings, labels=batch_labels)

        anchor_embeddings = keras.ops.take(embeddings, triplets[:, 0], axis=0)
        positive_embeddings = keras.ops.take(embeddings, triplets[:, 1], axis=0)
        negative_embeddings = keras.ops.take(embeddings, triplets[:, 2], axis=0)

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        recall = recall_at_k(embeddings, batch_labels, k=1)
        return loss, recall


    step = 0

    best_loss = float('inf')
    best_recall_at_1 = 0.0

    losses = []
    recalls = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        loss_fn = circle_loss_scheduler(epoch)

        if epoch > 10:
            no_people += 20
            no_photos += 5
            # triplet_fn = batch_hard

        train_dataloader = get_dataloader(no_people=no_people, no_photos=no_photos, shuffle=True, augment=True)

        epoch_losses = []
        epoch_recalls = []

        minibatch_losses = []
        minibatch_recalls = []

        for i, batch in enumerate(train_dataloader):
            step += 1
            loss_value, embeddings, labels = training_step(
                batch=batch, model=model, optimizer=optimiser,
                triplet_fn=triplet_fn, loss_fn=loss_fn)

            recall_value = recall_at_k(embeddings, labels, k=1)

            loss_value, recall_value = keras.ops.convert_to_numpy(loss_value), keras.ops.convert_to_numpy(recall_value)
            minibatch_losses.append(loss_value)
            minibatch_recalls.append(recall_value)
            epoch_losses.append(loss_value)
            epoch_recalls.append(recall_value)
            losses.append(loss_value)
            recalls.append(recall_value)

            if i % 10 == 0:
                running_loss = np.mean(minibatch_losses)
                running_recall = np.mean(minibatch_recalls)

                print(
                    f'\t[{epoch}: {i + 1}/{len(train_dataloader)}] Minibatch Loss: {running_loss:.4f}, Minibatch Recall@1: {running_recall:.4f}')
                minibatch_losses = []
                minibatch_recalls = []

        val_loss, val_recall = validation_step(val_batch, model, loss_fn)

        print(
            f'\tValidation Recall@1: {keras.ops.convert_to_numpy(val_recall):.4f}, Validation Loss: {keras.ops.convert_to_numpy(val_loss):.4f}')

        if val_recall > best_recall_at_1:
            best_recall_at_1 = val_recall
            patience_counter = 0
            print(
                f'\t[{epoch}/{epochs}] New Best Recall@1: {keras.ops.convert_to_numpy(best_recall_at_1):.4f}. Saving model...')
            # model.save_weights(chkpt_path, overwrite=True)
        else:
            patience_counter += 1
            no_photos += 5
            no_people += 5
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

    # losses

    # losses