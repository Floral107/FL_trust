"""imdbnoniid: A Flower / TensorFlow app."""

import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import toml

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
toml_file = BASE_DIR / "pyproject.toml"

path_to_save = os.path.join(BASE_DIR.parent, "420", "imdbnoniid")

# Redirect TF Hub cache to a plain ASCII path inside the project.
# The default %TEMP%/tfhub_modules/ path is the 8.3 short form of the
# username (e.g. FLRA~1 for Flóra) which causes Windows file-lock
# "Access is denied" errors when multiple Ray workers start in parallel.
_TFHUB_CACHE = str(BASE_DIR.parent / ".tfhub_cache")
os.makedirs(_TFHUB_CACHE, exist_ok=True)
os.environ.setdefault("TFHUB_CACHE_DIR", _TFHUB_CACHE)
# Give concurrent workers up to 5 min to wait for another process's download
# lock before giving up.
os.environ.setdefault("TFHUB_LOCK_WAIT_TIMEOUT_SECONDS", "300")

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import tensorflow as tf
# Cap per-process TF threading (see celebanoniid/task.py for rationale).
try:
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("FLR_TF_INTRA", "4")))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("FLR_TF_INTER", "2")))
except Exception:
    pass
_gpus = tf.config.list_physical_devices("GPU")
for _gpu in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import models
from keras import optimizers
tf.random.set_seed(SEED)
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import tensorflow_hub as hub
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import pickle
import requests
import time
from typing import Union, List

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

USE_MODEL = None  # Lazy-loaded on first call to process_text to avoid GPU OOM at import time
_USE_EMBED_CHUNK = 256  # texts per USE forward pass; keeps peak VRAM/RAM manageable



def load_model():
    set_seed(SEED)
    inputs = layers.Input(shape=(512,), dtype=tf.float32)  # USE embeddings are 512-dimensional
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)  # Two output classes

    model = models.Model(inputs=inputs, outputs=outputs)

    # LR 2e-5 -> 1e-3 (2026-07-08): at 2e-5, K=20 clients (1000 samples, ~32
    # steps/round) never left chance level (advisor comment #1). 1e-3 matches
    # the ADULT MLP; pilot-gated before the full rerun.
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=["accuracy"])
    return model


fds = None  # Cache FederatedDataset

def load_data(partition_id, num_partitions):
    set_seed(SEED)
    global fds
    
    # Directory to store partitions


    max_retries = 3
    retry_delay = 5  # seconds
        
    for attempt in range(max_retries):
        try:
            
            if fds is None:
            # Create a new partition with retry logic
                partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    alpha=0.5,
                    partition_by="label",
                    seed=42
                )
                    
                fds = FederatedDataset(
                    dataset="stanfordnlp/imdb",
                    partitioners={"train": partitioner},
                )
                
            partition = fds.load_partition(partition_id, "train")
            partition.set_format("numpy")
            x, y = partition["text"], partition["label"]
                
 
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = retry_delay * (attempt + 1)
                print(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue

    # Train-test split
    set_seed(SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Optional: truncate for fixed-size experiments
    num_train_samples = 20000
    num_test_samples = 5000
    x_train, y_train = x_train[:num_train_samples], y_train[:num_train_samples]
    x_test, y_test = x_test[:num_test_samples], y_test[:num_test_samples]

    # Ensure labels are numpy arrays
    y_train = np.array(y_train, dtype=np.int8).flatten()
    y_test = np.array(y_test, dtype=np.int8).flatten()

    return x_train, y_train, x_test, y_test


# Disk cache for USE embeddings. USE is deterministic (fixed weights, no
# dropout) and the Dirichlet partition text is fixed by seed=42 INDEPENDENT of
# the FL `seed` env — so a given (partition, K, split) text embeds to the exact
# same 512-d array across every seed/metric/mode. Caching by CONTENT HASH makes
# every re-embed after the first a bit-identical np.load instead of a full
# 256M-param USE forward pass (the imdb training bottleneck: ~10 re-embeds/cell).
_EMB_CACHE_DIR = BASE_DIR.parent / ".emb_cache" / "imdbnoniid"


def _emb_cache_key(texts):
    import hashlib
    h = hashlib.md5()
    for t in texts:
        h.update(repr(t).encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return f"n{len(texts)}_{h.hexdigest()}.npy"


def process_text(text_list):
    """Embed text with USE in small chunks to avoid OOM spikes, with a
    content-hash disk cache.

    Returns a float32 numpy array of shape (N, 512). Chunking keeps peak memory
    proportional to _USE_EMBED_CHUNK. The cache is bit-identical to live
    embedding (USE deterministic + .npy round-trips float32 exactly); disable
    with FLR_EMB_CACHE=0.
    """
    import gc
    # Idempotent: already-embedded (N,512) float input passes straight through,
    # so callers can embed ONCE and reuse across many coalition evals instead of
    # re-embedding per subset.
    _arr = text_list.numpy() if hasattr(text_list, "numpy") else text_list
    if isinstance(_arr, np.ndarray) and _arr.dtype != object and _arr.ndim == 2 and _arr.shape[1] == 512:
        return _arr.astype("float32", copy=False)
    texts = list(text_list)
    use_cache = os.environ.get("FLR_EMB_CACHE", "1") != "0"
    cache = None
    if use_cache and texts:
        cache = _EMB_CACHE_DIR / _emb_cache_key(texts)
        if cache.exists():
            try:
                out = np.load(cache)
                if out.shape[0] == len(texts):
                    return out
            except Exception:
                pass  # corrupt/partial → recompute below

    global USE_MODEL
    if USE_MODEL is None:
        USE_MODEL = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            trainable=False,  # preprocessing only — no fine-tuning
        )
    chunks = []
    for start in range(0, len(texts), _USE_EMBED_CHUNK):
        chunk = texts[start: start + _USE_EMBED_CHUNK]
        emb = USE_MODEL(chunk).numpy()  # .numpy() frees TF tensor immediately
        chunks.append(emb)
        gc.collect()
    out = np.concatenate(chunks, axis=0).astype(np.float32)

    if cache is not None:
        try:
            _EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}.npy")  # atomic: temp + replace
            np.save(tmp, out)
            os.replace(tmp, cache)
        except Exception:
            pass  # cache write failure must never break training
    return out

def get_num_clients():
    # Load the pyproject.toml file
    config = toml.load(toml_file)

    # Extracting values
    num_clients = config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
    return num_clients
config = toml.load(toml_file)

def set_num_clients(num_clients):
    
    config = toml.load(toml_file)

    config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] = num_clients

    with open(toml_file, "w") as f:
        toml.dump(config, f)
        
        
def _resolve_save_dir(
    strategy: str = "fedavg",
    et_mode: str = "none",
    weight_mode: str = "none",
    weight_metric: str = "",
) -> str:
    """Save layout (in precedence order):
        et_mode in {fair,adv,dp}              → 420_et_<mode>/<dataset>/...
        weight_mode in {st,dy} + weight_metric → 420_<wmode>_<metric>/<dataset>/...
        strategy=="fedprox"                   → 420_fedprox/<dataset>/...
        strategy=="fedavg" (BL)               → 420/<dataset>/...
    """
    em = (et_mode or "none").lower()
    if em in ("fair", "adv", "dp"):
        return os.path.join(str(BASE_DIR.parent), f"420_et_{em}", os.path.basename(path_to_save))
    wm = (weight_mode or "none").lower()
    if wm in ("st", "dy") and weight_metric:
        wmet = str(weight_metric).lower()
        return os.path.join(
            str(BASE_DIR.parent),
            f"420_{wm}_{wmet}",
            os.path.basename(path_to_save),
        )
    s = (strategy or "fedavg").lower()
    if s == "fedavg":
        return path_to_save
    return os.path.join(str(BASE_DIR.parent), f"420_{s}", os.path.basename(path_to_save))



def train_with_proximal(model, x, y, mu, epochs, batch_size, verbose=0, shuffle=False):
    """FedProx local training: task_loss + (mu/2) * sum ||w - w_global||^2.
    Call AFTER model.set_weights(global_parameters)."""
    global_trainables = [tf.constant(v.numpy()) for v in model.trainable_weights]
    if len(model.output_shape) >= 2 and model.output_shape[-1] == 1:
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
    loss_fn = tf.keras.losses.get(model.loss)
    optimizer = model.optimizer

    n = int(len(x))
    steps = max(1, int(np.ceil(n / batch_size)))
    for epoch in range(epochs):
        if shuffle:
            perm = tf.random.shuffle(tf.range(n))
            x_ep, y_ep = tf.gather(x, perm), tf.gather(y, perm)
        else:
            x_ep, y_ep = x, y
        epoch_loss = 0.0
        for step in range(steps):
            start = step * batch_size
            end = min(start + batch_size, n)
            x_batch, y_batch = x_ep[start:end], y_ep[start:end]
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                task_loss = tf.reduce_mean(loss_fn(y_batch, preds))
                prox_term = tf.add_n([
                    tf.reduce_sum(tf.square(w - w_g))
                    for w, w_g in zip(model.trainable_weights, global_trainables)
                ])
                loss = task_loss + (mu / 2.0) * prox_term
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss += float(loss.numpy())
        if verbose:
            print(f"[FedProx] Epoch {epoch+1}/{epochs} - loss={epoch_loss/steps:.4f}")


def save_client(model, client_num, partition_id, round_num, strategy="fedavg", et_mode="none",
                weight_mode="none", weight_metric=""):
    """Save the model of a client after training."""
    model_dir = os.path.join(
        _resolve_save_dir(strategy, et_mode, weight_mode, weight_metric),
        str(client_num), str(SEED),
    )
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"client_{partition_id}_round_{round_num}.keras")
    model.save(model_path)


def save_global_model(model, client_num, round_num, strategy="fedavg", et_mode="none",
                      weight_mode="none", weight_metric=""):
    """Save the global model after training."""
    model_dir = os.path.join(
        _resolve_save_dir(strategy, et_mode, weight_mode, weight_metric),
        str(client_num), str(SEED),
    )
    if round_num == 0:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "global_model_round_0.keras")
        model.save(path)
    else:
        model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
        model.save(model_path)


# ── ET training: import shared functions from project root ────────────────────
import sys as _sys
_PROJECT_ROOT = str(BASE_DIR.parent)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
from et_training import train_with_adv, train_with_dpsgd  # noqa: E402
# IMDB has no sensitive attribute → train_with_fair is intentionally not imported.
