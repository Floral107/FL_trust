import weakref
import tensorflow as tf

# Cache of compiled PGD-loop runners, keyed by model object. The in-loop res
# scoring calls pgd_attack ~600x/round on the SAME reused model object (only its
# weights change via set_weights, which does not invalidate a traced graph), so
# caching the compiled loop here turns a per-op eager loop into a single fused
# graph that compiles once and is reused. Identical math, ~10-50x faster on CPU.
_RUNNER_CACHE = weakref.WeakKeyDictionary()  # model -> {key: tf.function}


def _build_loss(loss_name, from_logits):
    if isinstance(loss_name, str):
        if loss_name == "binary_crossentropy":
            return tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        elif loss_name == "sparse_categorical_crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        elif loss_name == "categorical_crossentropy":
            return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
        else:
            return tf.keras.losses.get(loss_name)
    return loss_name


def _get_runner(model, loss_obj, num_iter, targeted, from_logits):
    """Get-or-build a compiled PGD iteration loop bound to `model`.

    Cached per (model, loss-class, from_logits, num_iter, targeted). The loop body
    is exactly the eager loop below — same op order — so with random_start=False
    the output is bit-identical to the original eager implementation.
    `reduce_retracing=True` lets the single trace serve the varying batch sizes
    (clean-correct subset differs per coalition) without endless retracing.
    """
    by_model = _RUNNER_CACHE.get(model)
    if by_model is None:
        by_model = {}
        try:
            _RUNNER_CACHE[model] = by_model
        except TypeError:
            by_model = None  # not weak-referenceable -> compile per call (rare)
    key = (loss_obj.__class__.__name__, bool(from_logits), int(num_iter), bool(targeted))
    if by_model is not None and key in by_model:
        return by_model[key]

    factor = -1.0 if targeted else 1.0
    n = int(num_iter)

    @tf.function(reduce_retracing=True)
    def _runner(x_orig, x_adv, y_tensor, alpha, eps, cmin, cmax):
        for _ in range(n):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = model(x_adv, training=False)
                loss = loss_obj(y_tensor, logits)
            grad = tape.gradient(loss, x_adv)
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
            x_adv = x_adv + factor * alpha * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x_orig - eps, x_orig + eps)
            x_adv = tf.clip_by_value(x_adv, cmin, cmax)
        return x_adv

    if by_model is not None:
        by_model[key] = _runner
    return _runner


def pgd_attack(
    model,
    x,
    y,
    epsilon=0.03,
    alpha=0.007,
    num_iter=40,
    clip_min=0.0,
    clip_max=1.0,
    targeted=False,
    random_start=True,
    loss_name="sparse_categorical_crossentropy",
    from_logits=True,
):
    """
    Simple L_inf PGD (TF2) attack.

    Behaviour is identical to the original eager implementation; the per-iteration
    loop is run inside a cached `@tf.function` (compiled once per model) so the
    hundreds of in-loop scoring calls do not pay per-op eager dispatch overhead.

    Args:
        model: tf.keras.Model. Should return logits if from_logits=True, otherwise probabilities.
        x: input tensor or numpy array of shape (batch, ...), dtype float32.
        y: labels (integers for sparse losses, or one-hot for categorical). Make sure dtype matches loss.
        epsilon: maximum L_inf perturbation.
        alpha: step size per iteration.
        num_iter: number of iterations.
        clip_min, clip_max: input value range.
        targeted: if True, performs a targeted attack (moves towards target labels in y).
        random_start: if True, start from x + uniform noise in [-eps, eps].
        loss_name: string name of loss or a tf.keras.losses.Loss instance.
        from_logits: whether model outputs logits (True) or probabilities (False).

    Returns:
        x_adv tensor (same dtype/shape as x).
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y)
    loss_obj = _build_loss(loss_name, from_logits)

    # Start point (kept eager so the RNG stream matches the original exactly).
    if random_start:
        noise = tf.random.uniform(shape=tf.shape(x), minval=-epsilon, maxval=epsilon, dtype=tf.float32)
        x_adv = tf.clip_by_value(x + noise, clip_min, clip_max)
    else:
        x_adv = tf.identity(x)

    runner = _get_runner(model, loss_obj, num_iter, targeted, from_logits)
    x_adv = runner(
        x, x_adv, y_tensor,
        tf.constant(alpha, tf.float32), tf.constant(epsilon, tf.float32),
        tf.constant(clip_min, tf.float32), tf.constant(clip_max, tf.float32),
    )
    return x_adv
