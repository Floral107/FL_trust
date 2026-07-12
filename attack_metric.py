import tensorflow as tf
import numpy as np
from cw_attack import cw_l2_attack, _wrap_sigmoid_as_softmax
# NOTE: imdb's process_text (USE embedder) is imported LAZILY inside the text
# functions — a top-level import breaks hosts that only deploy the *noniid*
# app dirs (e.g. the celeba eval box has no imdb/ package).

# Per-dataset PGD evaluation parameters (MAIN.tex, Experiments setup):
#   ADULT  : eps=0.3,    alpha=0.007,  40 iters
#   CelebA : eps=0.03,   alpha=0.007,  40 iters
#   IMDB   : eps=0.005, alpha=0.00125, 10 iters (attack in USE-embedding space;
#            eps recalibrated 2026-07-08 on the converged LR=1e-3 baseline: the
#            old 0.0015 left BL res~=clean acc (0.87), 0.005 -> BL res 0.47,
#            0.01 -> 0.08. See imdb_eps_curve.txt / advisor comment 5.)
# These are the values the article reports; callers that omit epsilon/alpha/
# num_iter get them resolved from this table. Explicit arguments still win.
PGD_EVAL_PARAMS = {
    "adult": (0.3, 0.007, 40), "adultnoniid": (0.3, 0.007, 40),
    "celeba": (0.01, 0.0025, 40), "celebanoniid": (0.01, 0.0025, 40),  # eps lowered 0.03->0.01 (0.03 saturated res to 0; 0.01 gives res~0.5)
    "cifar": (0.03, 0.007, 40), "cifarnoniid": (0.03, 0.007, 40),
    "imdb": (0.005, 0.00125, 10), "imdbnoniid": (0.005, 0.00125, 10),
}


def calculate_pgd_score(model, x_test, dataset_name, y_test,
                        epsilon=None, alpha=None, num_iter=None, batch_size=128):
    """
    Modular PGD robustness evaluation replicating ART behavior,
    with dataset-specific implementations.

    epsilon/alpha/num_iter default to the per-dataset PGD_EVAL_PARAMS above
    (the parameters reported in MAIN.tex) when not given explicitly.

    IMPORTANT:
    We now measure robustness ONLY on samples that are correctly
    classified on clean inputs.
    """
    dataset_name = dataset_name.lower()
    d_eps, d_alpha, d_iter = PGD_EVAL_PARAMS.get(dataset_name, (0.03, 0.007, 40))
    epsilon = d_eps if epsilon is None else epsilon
    alpha = d_alpha if alpha is None else alpha
    num_iter = d_iter if num_iter is None else num_iter
    dispatch = {
        "cifar": calculate_pgd_image_robustness,
        "cifarnoniid": calculate_pgd_image_robustness,
        "celeba": calculate_pgd_image_robustness,
        "celebanoniid": calculate_pgd_image_robustness,
        "imdb": calculate_pgd_text_robustness,
        "imdbnoniid": calculate_pgd_text_robustness,
        "adult": calculate_pgd_tabular_robustness,
        "adultnoniid": calculate_pgd_tabular_robustness,
    }

    if dataset_name not in dispatch:
        print(f"Unsupported dataset: {dataset_name}")
        return 0.0

    try:
        return dispatch[dataset_name](
            model, x_test, y_test, epsilon, alpha, num_iter, batch_size
        )
    except Exception as e:
        print(f"[ERROR] PGD robustness failed for {dataset_name}: {e}")
        return 0.0


def calculate_pgd_image_robustness(model, x_test, y_test,
                                   epsilon=0.03, alpha=0.007, num_iter=40, batch_size=2048):
    """
    PGD robustness for image models — GPU-optimised.

    Changes vs. CPU-era version:
      - num_iter kept at 40 to stay consistent with MAIN.tex §Setup, which
        specifies 40 PGD iterations for ADULT and CelebA evaluation. With
        GTG-Shapley's `max_permutations=30` cap doing the heavy lifting on
        wall-clock, dropping iters further would only save ~6 min/round and
        introduce a paper-text inconsistency.
      - batch_size 128 → 1024. PGD needs the gradient tape, which retains
        activations for backward — 8× more memory per batch than rob. 1024 fits
        comfortably on 24 GB RTX 4090 with our small CNN. Net effect: 8× fewer
        outer-loop iterations and 8× fewer gradient-tape contexts to spin up.
      - PGD inner step wrapped in @tf.function so the 40-iter loop reuses one
        compiled graph instead of re-tracing per iteration.

    Still measured ONLY on samples correctly classified clean.
    """
    num_samples = x_test.shape[0]
    total_flipped_from_correct = 0
    total_correct = 0

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Single tf.function-compiled PGD step. Cached after the first call;
    # subsequent calls reuse the graph for all coalitions × batches with
    # the same input shape. `model` is captured by reference — set_weights
    # picks up new weights without re-tracing.
    @tf.function(reduce_retracing=True)
    def _pgd_step(x_adv, x_orig, y, eps, alph):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            preds = model(x_adv, training=False)
            loss = loss_fn(y, preds)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alph * tf.sign(grad)
        perturb = tf.clip_by_value(x_adv - x_orig, -eps, eps)
        return tf.clip_by_value(x_orig + perturb, 0.0, 1.0)

    eps_t = tf.constant(epsilon, dtype=tf.float32)
    alph_t = tf.constant(alpha, dtype=tf.float32)

    for i in range(0, num_samples, batch_size):
        x_batch_np = x_test[i:i + batch_size]
        y_batch_np = y_test[i:i + batch_size]
        if x_batch_np.shape[0] == 0:
            continue

        x_batch = tf.convert_to_tensor(x_batch_np, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch_np, dtype=tf.int32)

        # Clean predictions
        logits_clean = model(x_batch, training=False)
        preds_clean = tf.argmax(logits_clean, axis=1, output_type=tf.int32)
        correct_mask = tf.equal(preds_clean, y_batch)
        num_correct_batch = int(tf.reduce_sum(tf.cast(correct_mask, tf.int32)).numpy())

        # PGD attack. Run the loop regardless of correct count so the metric
        # is well-defined when no batch is fully correct.
        x_adv = tf.identity(x_batch)
        for _ in range(num_iter):
            x_adv = _pgd_step(x_adv, x_batch, y_batch, eps_t, alph_t)

        if num_correct_batch == 0:
            continue

        logits_adv = model(x_adv, training=False)
        preds_adv = tf.argmax(logits_adv, axis=1, output_type=tf.int32)
        flipped_mask = tf.logical_and(correct_mask, tf.not_equal(preds_adv, preds_clean))
        num_flipped_batch = int(tf.reduce_sum(tf.cast(flipped_mask, tf.int32)).numpy())

        total_correct += num_correct_batch
        total_flipped_from_correct += num_flipped_batch

    if total_correct == 0:
        print("[WARN] No correctly classified samples found; PGD robustness set to 0.0.")
        return 0.0

    return 1.0 - (total_flipped_from_correct / total_correct)

def calculate_pgd_text_robustness(
    model,
    x_test,
    y_test,
    epsilon=4.0 / 255.0,
    alpha=2.0 / 255.0,
    num_iter=5,
    batch_size=256,
):
    """
    PGD robustness for text models (IMDB), measured ONLY on correctly
    classified clean samples.

    x_test: lista / tömb szövegek
    y_test: címkék, itt int8-ként érkeznek (0/1 vagy több osztály),
            ezeket int32-re castoljuk.
    """

    print("=== PGD TEXT DEBUG ===")
    print(f"epsilon = {epsilon:.6f}, alpha = {alpha:.6f}, num_iter = {num_iter}")
    print(f"num samples (len(x_test)) = {len(x_test)}")

    # 1) Embed szöveg -> vektorok
    from imdb.imdb.task import process_text
    embeddings = process_text(x_test)  # np.array
    print(f"embeddings type: {type(embeddings)}, shape: {getattr(embeddings, 'shape', None)}")

    embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)
    print(f"embeddings TF shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # 2) Label int8 -> int32 (SparseCategoricalCrossentropy ezt várja)
    print(f"y_test raw dtype: {np.array(y_test).dtype}, first 10: {np.array(y_test)[:10]}")
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)
    y_test = tf.cast(y_test, tf.int32)
    print(f"y_test TF dtype after cast: {y_test.dtype}")
    print(f"y_test unique labels (np): {np.unique(y_test.numpy())}")

    # 3) Clean predikciók
    logits_clean = model(embeddings, training=False)
    preds_clean = tf.argmax(logits_clean, axis=1, output_type=tf.int32)

    # Clean accuracy check
    correct_mask_full = tf.equal(preds_clean, y_test)
    total_correct_full = int(tf.reduce_sum(tf.cast(correct_mask_full, tf.int32)).numpy())
    total_samples = int(embeddings.shape[0])

    clean_acc = total_correct_full / total_samples if total_samples > 0 else 0.0
    print(f"[CLEAN] total_samples = {total_samples}, total_correct = {total_correct_full}, clean_acc = {clean_acc:.4f}")

    # Csak a clean-en helyes minták
    correct_mask = correct_mask_full
    idx = tf.where(correct_mask)[:, 0]   # indexek a helyes mintákhoz
    print(f"Number of correctly classified samples used for PGD: {idx.shape[0]}")

    embeddings_corr = tf.gather(embeddings, idx)
    y_corr = tf.gather(y_test, idx)

    print(f"embeddings_corr shape: {embeddings_corr.shape}")
    print(f"y_corr shape: {y_corr.shape}, first 10: {y_corr.numpy()[:10] if y_corr.shape[0] > 0 else '[]'}")

    total_correct = tf.shape(y_corr)[0]
    total_correct_int = int(total_correct.numpy())
    print(f"total_correct_int (clean-correct used for PGD) = {total_correct_int}")

    if total_correct_int == 0:
        print("[WARN] No correctly classified samples found; PGD robustness set to 0.0.")
        return 0.0

    # 4) PGD csak a clean-helyes mintákon
    adv = tf.Variable(embeddings_corr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # debug: skálák az embedding térben
    emb_min = tf.reduce_min(embeddings_corr).numpy()
    emb_max = tf.reduce_max(embeddings_corr).numpy()
    print(f"embeddings_corr value range: min={emb_min:.6f}, max={emb_max:.6f}")

    for it in range(num_iter):
        with tf.GradientTape() as tape:
            preds = model(adv, training=False)
            loss = loss_fn(y_corr, preds)
        grad = tape.gradient(loss, adv)

        if grad is None:
            print(f"[ITER {it}] grad is None! Something is off with the graph.")
            break

        grad_min = tf.reduce_min(grad).numpy()
        grad_max = tf.reduce_max(grad).numpy()
        print(f"[ITER {it}] loss = {loss.numpy():.6f}, grad range: min={grad_min:.6f}, max={grad_max:.6f}")

        # L∞-lépés embedding-térben
        adv.assign_add(alpha * tf.sign(grad))
        perturb = adv - embeddings_corr
        pert_min = tf.reduce_min(perturb).numpy()
        pert_max = tf.reduce_max(perturb).numpy()
        print(f"[ITER {it}] BEFORE clip perturb range: min={pert_min:.6f}, max={pert_max:.6f}")

        perturb = tf.clip_by_value(perturb, -epsilon, epsilon)
        adv.assign(embeddings_corr + perturb)

        pert_min = tf.reduce_min(perturb).numpy()
        pert_max = tf.reduce_max(perturb).numpy()
        print(f"[ITER {it}] AFTER  clip perturb range: min={pert_min:.6f}, max={pert_max:.6f}")

    # 5) PGD utáni predikció
    logits_adv = model(adv, training=False)
    preds_adv = tf.argmax(logits_adv, axis=1, output_type=tf.int32)

    # Robusztus pontosság az eredetileg helyeseken
    still_correct = tf.equal(preds_adv, y_corr)
    num_still_correct = int(
        tf.reduce_sum(tf.cast(still_correct, tf.int32)).numpy()
    )

    print(f"num_still_correct after PGD = {num_still_correct} / {total_correct_int}")
    print(f"preds_adv first 10: {preds_adv.numpy()[:10] if preds_adv.shape[0] > 0 else '[]'}")
    print(f"y_corr    first 10: {y_corr.numpy()[:10] if y_corr.shape[0] > 0 else '[]'}")

    robustness = num_still_correct / total_correct_int
    print(f"[RESULT] PGD robustness = {robustness:.6f}")
    print("=== END PGD TEXT DEBUG ===")

    return float(robustness)


def calculate_pgd_tabular_robustness(model, x_test, y_test,
                                     epsilon=0.03, alpha=0.007, num_iter=40, batch_size=128):
    """
    PGD robustness for tabular models (binary classification),
    measured ONLY on correctly classified clean samples.
    """
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Make sure shapes align (N,) vs (N,1)
    if len(y_test.shape) == 1:
        y_test_vec = tf.reshape(y_test, (-1, 1))
    else:
        y_test_vec = y_test

    # Clean predictions
    preds_clean_prob = model(x_test, training=False)
    preds_clean = tf.round(preds_clean_prob)  # 0/1

    correct_mask = tf.equal(preds_clean, y_test_vec)
    total_correct = tf.reduce_sum(
        tf.cast(correct_mask, tf.int32)
    ).numpy()

    # Build adversarial inputs
    adv = tf.Variable(tf.identity(x_test))
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            preds = model(adv, training=False)
            loss = loss_fn(y_test_vec, preds)
        grad = tape.gradient(loss, adv)

        adv.assign_add(alpha * tf.sign(grad))
        perturb = tf.clip_by_value(adv - x_test, -epsilon, epsilon)
        adv.assign(tf.clip_by_value(x_test + perturb, 0.0, 1.0))

    preds_adv_prob = model(adv, training=False)
    preds_adv = tf.round(preds_adv_prob)

    flipped_mask = tf.logical_and(
        correct_mask,
        tf.not_equal(preds_adv, preds_clean)
    )
    total_flipped_from_correct = tf.reduce_sum(
        tf.cast(flipped_mask, tf.int32)
    ).numpy()

    if total_correct == 0:
        print("[WARN] No correctly classified samples found; PGD robustness set to 0.0.")
        return 0.0

    robustness = 1.0 - (total_flipped_from_correct / total_correct)
    return float(robustness)


# =====================================================================
# Carlini-Wagner L2 robustness
# CW is optimization-based and far slower per sample than PGD, so we
# subsample the test set before attacking. Mirrors the DP/EO split:
# callers pick `adv_pgd` vs `adv_cw` at the method level.
# =====================================================================

_CW_DEFAULT_SUBSAMPLE = 150


def calculate_cw_score(model, x_test, dataset_name, y_test,
                       subsample=_CW_DEFAULT_SUBSAMPLE,
                       confidence=0.0, learning_rate=0.01,
                       binary_search_steps=9, max_iter=100,
                       initial_const=0.01, batch_size=128):
    """Dataset-dispatched CW-L2 robustness (analogous to calculate_pgd_score)."""
    dataset_name = dataset_name.lower()
    dispatch = {
        "cifar": calculate_cw_image_robustness,
        "cifarnoniid": calculate_cw_image_robustness,
        "celeba": calculate_cw_image_robustness,
        "celebanoniid": calculate_cw_image_robustness,
        "imdb": calculate_cw_text_robustness,
        "imdbnoniid": calculate_cw_text_robustness,
        "adult": calculate_cw_tabular_robustness,
        "adultnoniid": calculate_cw_tabular_robustness,
    }
    if dataset_name not in dispatch:
        print(f"Unsupported dataset: {dataset_name}")
        return 0.0
    try:
        return dispatch[dataset_name](
            model, x_test, y_test,
            subsample=subsample, confidence=confidence,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iter=max_iter, initial_const=initial_const,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"[ERROR] CW robustness failed for {dataset_name}: {e}")
        return 0.0


def _subsample_indices(n, k, seed=0):
    """Deterministic subsample of k indices from range(n). Returns all of
    them if n <= k. Seeded so evals across clients stay comparable."""
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)


def calculate_cw_image_robustness(model, x_test, y_test, *, subsample,
                                  confidence, learning_rate,
                                  binary_search_steps, max_iter,
                                  initial_const, batch_size):
    """CW-L2 robustness for multiclass image models (CIFAR)."""
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test).flatten().astype(np.int32)

    idx = _subsample_indices(len(x_test), subsample)
    x_sub, y_sub = x_test[idx], y_test[idx]

    logits_clean = model(x_sub, training=False).numpy()
    preds_clean = np.argmax(logits_clean, axis=1)
    correct_mask = preds_clean == y_sub
    total_correct = int(correct_mask.sum())
    if total_correct == 0:
        print("[WARN] No correctly classified samples found; CW robustness set to 0.0.")
        return 0.0

    x_corr = x_sub[correct_mask]
    y_corr = y_sub[correct_mask]

    nb_classes = int(logits_clean.shape[1])
    x_adv = cw_l2_attack(
        model=model, x=x_corr, y=y_corr,
        nb_classes=nb_classes, clip_values=(0.0, 1.0),
        confidence=confidence, learning_rate=learning_rate,
        binary_search_steps=binary_search_steps, max_iter=max_iter,
        initial_const=initial_const, batch_size=batch_size,
    )
    preds_adv = np.argmax(model(x_adv, training=False).numpy(), axis=1)
    still_correct = int((preds_adv == y_corr).sum())
    return float(still_correct / total_correct)


def calculate_cw_text_robustness(model, x_test, y_test, *, subsample,
                                 confidence, learning_rate,
                                 binary_search_steps, max_iter,
                                 initial_const, batch_size):
    """CW-L2 robustness for IMDB text models, attacked in USE-embedding
    space (same input plane PGD uses for text)."""
    from imdb.imdb.task import process_text
    embeddings = process_text(x_test)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    y_test = np.asarray(y_test).flatten().astype(np.int32)

    idx = _subsample_indices(len(embeddings), subsample)
    emb_sub, y_sub = embeddings[idx], y_test[idx]

    logits_clean = model(emb_sub, training=False).numpy()
    preds_clean = np.argmax(logits_clean, axis=1)
    correct_mask = preds_clean == y_sub
    total_correct = int(correct_mask.sum())
    if total_correct == 0:
        print("[WARN] No correctly classified samples found; CW robustness set to 0.0.")
        return 0.0

    emb_corr = emb_sub[correct_mask]
    y_corr = y_sub[correct_mask]

    clip_low = float(embeddings.min())
    clip_high = float(embeddings.max())
    nb_classes = int(logits_clean.shape[1])

    x_adv = cw_l2_attack(
        model=model, x=emb_corr, y=y_corr,
        nb_classes=nb_classes, clip_values=(clip_low, clip_high),
        confidence=confidence, learning_rate=learning_rate,
        binary_search_steps=binary_search_steps, max_iter=max_iter,
        initial_const=initial_const, batch_size=batch_size,
    )
    preds_adv = np.argmax(model(x_adv, training=False).numpy(), axis=1)
    still_correct = int((preds_adv == y_corr).sum())
    return float(still_correct / total_correct)


def calculate_cw_tabular_robustness(model, x_test, y_test, *, subsample,
                                    confidence, learning_rate,
                                    binary_search_steps, max_iter,
                                    initial_const, batch_size):
    """CW-L2 robustness for Adult (binary sigmoid models). ART's multiclass
    CW requires >=2 output units, so we wrap the 1-unit sigmoid as a
    pseudo-softmax emitting [1-p, p] without mutating the original model.
    """
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test).flatten().astype(np.int32)

    idx = _subsample_indices(len(x_test), subsample)
    x_sub, y_sub = x_test[idx], y_test[idx]

    preds_clean_prob = model(x_sub, training=False).numpy().flatten()
    preds_clean = (preds_clean_prob >= 0.5).astype(np.int32)
    correct_mask = preds_clean == y_sub
    total_correct = int(correct_mask.sum())
    if total_correct == 0:
        print("[WARN] No correctly classified samples found; CW robustness set to 0.0.")
        return 0.0

    x_corr = x_sub[correct_mask]
    y_corr = y_sub[correct_mask]

    clip_low = float(x_test.min())
    clip_high = float(x_test.max())

    wrapped = _wrap_sigmoid_as_softmax(model, input_shape=x_corr.shape[1:])
    x_adv = cw_l2_attack(
        model=wrapped, x=x_corr, y=y_corr,
        nb_classes=2, clip_values=(clip_low, clip_high),
        confidence=confidence, learning_rate=learning_rate,
        binary_search_steps=binary_search_steps, max_iter=max_iter,
        initial_const=initial_const, batch_size=batch_size,
    )
    preds_adv_prob = model(x_adv, training=False).numpy().flatten()
    preds_adv = (preds_adv_prob >= 0.5).astype(np.int32)
    still_correct = int((preds_adv == y_corr).sum())
    return float(still_correct / total_correct)
