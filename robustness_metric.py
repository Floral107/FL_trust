import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# NOTE: imdb's process_text and adult's load_data are imported LAZILY inside
# their dataset branches — top-level imports break hosts that only deploy a
# subset of the app dirs (e.g. the celeba eval box has no imdb/ or adult/).





def aggregate_models(models):
    """Aggregates a list of models using FedAvg."""
    avg_weights = [np.mean([model[i] for model in models], axis=0) for i in range(len(models[0]))]
    return avg_weights

def get_num_round(model_filename):
    """Extracts the round number from a client model filename."""
    parts = model_filename.split("_")
    return int(parts[-1].split(".")[0])


def load_model(model_path):
    """Loads a single Keras model from a file."""
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"Model file not found: {model_path}")
        return None

def load_client_models(model_dir, num_clients, round_num):
    """Loads all client models from the given directory."""
    client_models = {}
    for client_id in range(num_clients):
        model_filename = f"client_{client_id}_round_{round_num}.keras"
        model_files = [f for f in os.listdir(model_dir) if f == model_filename]
        if not model_files:
            print(f"No model found for Client {client_id}")
            continue
        latest_model_file = sorted(model_files, key=get_num_round)[-1]
        model_path = os.path.join(model_dir, latest_model_file)
        model = load_model(model_path)
        if model:
            client_models[client_id] = model
            print(f"Loaded model for Client {client_id}: {latest_model_file}")
    return client_models

def calculate_robustness_score(model, x_test, dataset_name, y_test = None,  sigma=0.1, L=100):
    """Calculate robustness score with memory-efficient processing"""
    try:
        if dataset_name in ("imdb", "imdbnoniid"):
            return calculate_text_robustness(model, x_test, L)
        elif dataset_name in ("cifar", "cifarnoniid", "celeba", "celebanoniid"):
            L=100
            return calculate_image_robustness(model, x_test, sigma, L)
        elif dataset_name in ("adult", "adultnoniid"):
            sigma = 0.1
            # TISZTA adatot kérünk + meta-t
            from adult.adult.task import load_data as load_adult_data
            _, _, x_test, _, meta = load_adult_data(partition_id=0, num_partitions=1, sigma=None, return_meta=True)
            return calculate_tabular_robustness(model=model, data=x_test, sigma=sigma, L=L, meta=meta)
        else:
            print(f"Unsupported dataset: {dataset_name}")
            return 0.0
    except Exception as e:
        print(f"Error in calculate_robustness_score: {str(e)}")
        return 0.0



def calculate_image_robustness(model, x_test, sigma=0.1, L=100, batch_size=8192, chunk_size=10):
    """GPU-optimised certified robustness.

    Changes vs. the CrySyS-era version (which targeted 18-core CPU):
      - batch_size 512 → 4096. RTX 4090 has 24 GB and our CNN is small;
        8× fewer Python roundtrips per coalition is the biggest single win.
      - model.predict() → model(x, training=False) wrapped in @tf.function
        with reduce_retracing. Skips Keras's per-call wrapper overhead and
        re-uses the cached graph across coalitions (weights change is fine
        — weights aren't part of the trace key).
      - chunk_size kept at 10 (the (10, 5000, 64, 64, 3) tensor is ~2.5 GB;
        bigger chunks hit OOM on 24 GB GPUs with current intermediate usage).
    Honors FLR_NOISE_SAMPLES env var (set by robustness.py --noise_samples).
    """
    import os as _os
    L = int(_os.environ.get("FLR_NOISE_SAMPLES", L))

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    N = int(x_test.shape[0])

    # JIT-compiled forward pass. tf.function caches one graph for our fixed
    # input shape (batch_size, 64, 64, 3); subsequent calls hit the cache.
    # `model` is captured by reference, so set_weights() between coalitions
    # picks up the new weights without re-tracing.
    @tf.function(reduce_retracing=True)
    def _fwd_argmax(x):
        return tf.cast(tf.argmax(model(x, training=False), axis=1), tf.int32)

    def _batched_argmax(x):
        n = int(x.shape[0])
        outs = []
        for i in range(0, n, batch_size):
            outs.append(_fwd_argmax(x[i:i + batch_size]))
        return tf.concat(outs, axis=0)

    clean_preds = _batched_argmax(x_test)

    total_mismatches = 0
    total_count = 0
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        current_L = end - start

        noise = tf.random.normal(shape=(current_L, *x_test.shape),
                                 mean=0.0, stddev=sigma, dtype=tf.float32)
        noisy = tf.clip_by_value(x_test[None, ...] + noise, 0.0, 1.0)
        noisy_flat = tf.reshape(noisy, (-1, *x_test.shape[1:]))

        noisy_preds = _batched_argmax(noisy_flat)
        clean_tiled = tf.tile(clean_preds, [current_L])  # row j of noisy_flat ↔ test point j % N

        mismatches = tf.reduce_sum(tf.cast(noisy_preds != clean_tiled, tf.int32))
        total_mismatches += int(mismatches.numpy())
        total_count += current_L * N

        del noise, noisy, noisy_flat, noisy_preds, clean_tiled

    return 1.0 - total_mismatches / total_count



def calculate_tabular_robustness(
    model,
    data,
    sigma: float,
    L: int,
    meta: dict,
    batch_size: int = 512,
    delta: float = 1e-5,
):
    """
    Robusztusság:
      - OHE oszlopokon: randomized response (DP-kalibrált p_keep sigma alapján)
      - numerikus oszlopokon: Gauss zaj N(0, sigma^2), 3*sigma-nál vágva
    """
    import tensorflow as tf
    import numpy as np

    data = np.asarray(data, dtype=np.float32)
    N, D = data.shape

    cat_slices = meta["cat_slices"]
    n_cat = meta["n_cat"]

    # clean predikciók
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    preds_clean = tf.round(model(data_tf, training=False))
    preds_clean = tf.reshape(preds_clean, [-1])

    mismatch_total = 0.0

    for _ in range(L):
        X_noisy = data.copy()

        # 1) OHE blokkok RR-rel
        for sl in cat_slices:
            block = X_noisy[:, sl]
            X_noisy[:, sl] = random_response_one_hot_block(block, sigma=sigma, delta=delta)

        # 2) numerikus oszlopokra Gauss zaj (ha van ilyen)
        if n_cat < D:
            num_part = X_noisy[:, n_cat:]
            noise = np.random.normal(loc=0.0, scale=sigma, size=num_part.shape).astype(np.float32)
            noise = np.clip(noise, -3.0 * sigma, 3.0 * sigma)
            num_part += noise
            X_noisy[:, n_cat:] = num_part

        # 3) modell kiértékelés batch-ekben
        X_noisy_tf = tf.convert_to_tensor(X_noisy, dtype=tf.float32)

        preds_list = []
        for i in range(0, N, batch_size):
            batch = X_noisy_tf[i : i + batch_size]
            preds_batch = tf.round(model(batch, training=False))
            preds_list.append(preds_batch)

        preds_noisy = tf.concat(preds_list, axis=0)
        preds_noisy = tf.reshape(preds_noisy, [-1])

        mismatches = tf.reduce_sum(tf.cast(preds_noisy != preds_clean, tf.float32))
        mismatch_total += float(mismatches.numpy())

    robustness = 1.0 - mismatch_total / (L * float(N))
    return robustness




def calculate_text_robustness(model, texts, L=100, batch_size=256, noise_level=0.1):
    try:
        #L=20
        # 1. Get clean predictions
        from imdb.imdb.task import process_text
        clean_embeddings = process_text(texts)
        predictions_clean = tf.argmax(model.predict(clean_embeddings, batch_size=batch_size, verbose=0), axis=1)

        # 2. Repeat embeddings L times
        clean_embeddings_tiled = tf.repeat(clean_embeddings, repeats=L, axis=0)

        # 3. Add noise to each replicated embedding
        noisy_embeddings = add_noise(clean_embeddings_tiled, noise_level=noise_level)

        # 4. Predict on noisy embeddings
        predictions_noisy = tf.argmax(model.predict(noisy_embeddings, batch_size=batch_size, verbose=0), axis=1)

        # 5. Reshape predictions to (num_texts, L)
        predictions_noisy = tf.reshape(predictions_noisy, (len(texts), L))
        predictions_clean = tf.reshape(predictions_clean, (-1, 1))

        # 6. Compare clean vs noisy
        mismatches = tf.reduce_sum(tf.cast(predictions_noisy != predictions_clean, tf.int32))
        robustness_score = 1.0 - mismatches / (len(texts) * L)

        return robustness_score

    except Exception as e:
        print(f"Error in robustness calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def add_noise(text, noise_level=0.1):
    noise = tf.random.normal(shape=tf.shape(text), mean=0.0, stddev=noise_level, dtype=tf.float32)
    noisy_embeddings = text + noise
    return noisy_embeddings

def random_response_one_hot_block(block: np.ndarray, sigma: float, delta: float = 1e-5) -> np.ndarray:
    """
    block: (N, d) OHE mátrix egy kategóriára.
    sigma: Gauss szórás, amiből p_keep-et számolunk.
    """
    n, d = block.shape
    p_keep = calculate_p_keep(sigma=sigma, d=d, delta=delta)

    # eredeti kategória indexek
    y = block.argmax(axis=1)  # shape: (n,)

    # eldöntjük, hol flipelünk
    u = np.random.rand(n)
    flip_mask = u > p_keep   # True = flip, False = marad

    out = block.copy()

    if flip_mask.any():
        # kategória -> index
        # (feltételezzük, hogy block korrekt OHE, így y 0..d-1 között van)
        y_flip = y[flip_mask]

        # sorsolunk másik kategóriát (d-1 lehetőség)
        r = np.random.randint(0, d - 1, size=y_flip.shape[0])
        r += (r >= y_flip).astype(int)  # ha az eredetivel esne egybe, tolunk rajta 1-et

        # nullázzuk a sort, és beállítjuk az új 1-est
        block_flipped = np.zeros_like(block[flip_mask], dtype=np.float32)
        block_flipped[np.arange(len(r)), r] = 1.0

        out[flip_mask] = block_flipped

    return out.astype(np.float32)

def calculate_p_keep(sigma: float, d: int, delta: float = 1e-5) -> float:
    """
    Calculates the probability 'p' (p_keep) based on the formula provided in the image.
    
    Args:
        sigma: The Gaussian noise scale (standard deviation).
        d: The number of categories (dimension) for the specific feature.
        delta: The privacy delta parameter (default 10^-5).
        
    Returns:
        float: The probability of keeping the true category (p).
    """
    # Calculate the numerator term: exp( (2 * sqrt(ln(1.25/delta))) / sigma )
    numerator_exponent = (2 * np.sqrt(np.log(1.25 / delta))) / sigma
    term = np.exp(numerator_exponent)
    
    # Calculate p based on the image formula: term / (term + d - 1)
    p = term / (term + d - 1)
    return p