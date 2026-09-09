"""Low-level Carlini-Wagner L2 attack utility.

Wraps ART's CarliniL2Method + TensorFlowV2Classifier. Kept separate from
attack_metric.py for symmetry with pgd_attack.py.

CW is an optimization-based attack, much slower per sample than PGD — callers
typically subsample the test set before invoking.
"""

import numpy as np
import tensorflow as tf


def _wrap_sigmoid_as_softmax(sigmoid_model, input_shape=None):
    """Convert a single-logit sigmoid Keras model into a 2-class softmax-like
    model outputting [1-p, p]. Needed so ART's multiclass CW can attack
    binary-output models without mutating them.

    input_shape: tuple of ints (e.g. (13,) for Adult). Must be supplied for
    Keras 3, where freshly loaded Sequential models do not expose .input/.output
    until the model has been called at least once. Pass x.shape[1:] from the
    caller.
    """
    if input_shape is not None:
        # Keras 3 compatible: build functional wrapper via a symbolic Input call.
        inp = tf.keras.Input(shape=input_shape)
        p = sigmoid_model(inp)  # symbolic forward pass — builds the graph
    else:
        # Legacy fallback for already-called models (Keras 2 style).
        inp = sigmoid_model.input
        p = sigmoid_model.output  # (N, 1)
    p0 = tf.keras.layers.Lambda(lambda t: 1.0 - t)(p)
    stacked = tf.keras.layers.Concatenate(axis=-1)([p0, p])  # (N, 2)
    return tf.keras.Model(inputs=inp, outputs=stacked)


def _make_tf2_classifier(model, nb_classes, clip_values, input_shape):
    """Build an ART TensorFlowV2Classifier around a Keras model. ART requires a
    loss_object even if the attack overrides it internally."""
    from art.estimators.classification import TensorFlowV2Classifier

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return TensorFlowV2Classifier(
        model=model,
        nb_classes=nb_classes,
        input_shape=input_shape,
        loss_object=loss_object,
        clip_values=clip_values,
    )


def cw_l2_attack(
    model,
    x,
    y,
    nb_classes,
    clip_values,
    *,
    confidence=0.0,
    learning_rate=0.01,
    binary_search_steps=10,
    max_iter=10,
    initial_const=0.01,
    batch_size=128,
):
    """Generate CW-L2 adversarial examples.

    Args:
        model: Keras model producing class probabilities (shape (N, nb_classes)).
               For sigmoid-binary models, wrap with _wrap_sigmoid_as_softmax first.
        x: inputs, shape (N, ...). Will be coerced to float32.
        y: ground-truth labels as integers, shape (N,). Used as the class to move
           prediction AWAY from (untargeted attack).
        nb_classes: number of output classes.
        clip_values: (low, high) bounds for the input space.

    Returns:
        numpy array of adversarial examples, same shape/dtype as input x.
    """
    from art.attacks.evasion import CarliniL2Method

    classifier = _make_tf2_classifier(
        model=model,
        nb_classes=nb_classes,
        clip_values=clip_values,
        input_shape=tuple(x.shape[1:]),
    )
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        targeted=False,
        learning_rate=learning_rate,
        binary_search_steps=binary_search_steps,
        max_iter=max_iter,
        initial_const=initial_const,
        batch_size=batch_size,
        verbose=False,
    )
    x_np = np.asarray(x, dtype=np.float32)
    y_int = np.asarray(y).astype(np.int32).flatten()
    y_onehot = tf.keras.utils.to_categorical(y_int, num_classes=nb_classes).astype(np.float32)
    return attack.generate(x=x_np, y=y_onehot)
