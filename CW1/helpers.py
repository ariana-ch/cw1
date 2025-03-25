from pathlib import Path
import keras
import tensorflow as tf


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



def circle_loss_scheduled(epoch, anchor_embeddings, positive_embeddings, negative_embeddings,
                          base_m=0.2, base_gamma=164, step_size=10):
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
    m = tf.minimum(m, tf.constant(1.0, dtype=tf.float32))
    # gamma = base_gamma * (1 + steps * 0.5)
    gamma_increment = tf.multiply(tf.cast(steps, tf.float32), 0.5)
    gamma = tf.multiply(base_gamma, tf.add(1.0, gamma_increment))
    return circle_loss(anchor_embeddings, positive_embeddings, negative_embeddings, m, gamma)


# This is adapted from the paper Hermans, A., Beyer, L., & Leibe, B. (2017). In Defense of the Triplet Loss for Person Re-Identification. ArXiv, abs/1703.07737.
class ExponentialDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-3, t0=15000, t1=25000, final_factor=0.001):
        super().__init__()
        self.initial_lr = initial_lr
        self.t0 = t0
        self.t1 = t1
        self.final_factor = final_factor

    def __call__(self, step):
        # Convert step to float for division ops
        step = tf.cast(step, tf.float32)

        # Phase 1: before t0, lr is constant
        phase1 = tf.less_equal(step, self.t0)
        lr_phase1 = self.initial_lr

        # Phase 2: t0 < step <= t1, exponential decay
        decay_exponent = (step - self.t0) / (self.t1 - self.t0)
        lr_phase2 = self.initial_lr * tf.pow(self.final_factor, decay_exponent)

        # Phase 3: after t1, lr stays at final value
        phase3 = tf.greater(step, self.t1)
        lr_phase3 = self.initial_lr * self.final_factor

        # Select learning rate based on current phase
        return tf.where(phase1, lr_phase1,
                        tf.where(phase3, lr_phase3, lr_phase2))


def get_path(name, directory, fmt, postfix = '', overwrite: bool = True, mkdir: bool = True):
    path = Path(f'./{directory}/{name}{postfix}.{fmt}')
    if path.exists() and not overwrite:
        if postfix != '': postfix += 1
        else: postfix = 0
        return get_path(name=name, directory=directory, fmt=fmt, postfix=postfix)
    if mkdir:
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
