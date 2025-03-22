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