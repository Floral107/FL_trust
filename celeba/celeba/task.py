"""celeba: A Flower / TensorFlow app.

CelebA with `Smiling` as the binary target and `Male` as the sensitive
attribute for fairness (DP/EO). Images are resized to 64x64 to keep
training tractable — the raw 178x218 would blow up conv costs without
materially changing the fairness signal for the thesis workload.

Output is a 2-class softmax (not sigmoid) so the existing image-style
PGD / CW dispatch and SparseCategoricalCrossentropy loss path in
cont_evals.py work without special-casing this dataset.
"""

import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
_gpus = tf.config.list_physical_devices("GPU")
for _gpu in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass
from sklearn.model_selection import train_test_split
tf.random.set_seed(SEED)

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
toml_file = BASE_DIR / "pyproject.toml"

path_to_save = os.path.join(BASE_DIR.parent, "420", "celeba")

# Redirect HuggingFace datasets cache to a plain ASCII path inside the project.
# The default ~/.cache/huggingface path can be an accented username on Windows
# (e.g. Flóra → FLRA~1) which causes file-lock "Access is denied" errors when
# multiple Ray workers start in parallel and all try to download/verify the
# dataset simultaneously.
_HF_CACHE = str(BASE_DIR.parent / ".hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(_HF_CACHE) / "datasets"))

import keras
tf.keras.utils.set_random_seed(SEED)

import toml
from keras import layers, regularizers
from keras import Sequential
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)


IMG_SIZE = 64
TARGET_ATTR = "Smiling"
SENSITIVE_ATTR = "Male"
# Cap sample counts to match the CIFAR workload order-of-magnitude.
NUM_TRAIN_SAMPLES = 30000
NUM_TEST_SAMPLES = 5000

# Disk cache for preprocessed partition arrays — avoids re-downloading and
# re-running PIL resize on every run / every Ray actor process.
_DISK_CACHE_DIR = BASE_DIR.parent / ".partition_cache" / "celeba"

_op_determinism_set = False


def set_seed(seed: int):
    global _op_determinism_set
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if not _op_determinism_set:
        tf.config.experimental.enable_op_determinism()
        _op_determinism_set = True


set_seed(SEED)


def load_model(weight_decay=5e-4):
    """Small CNN for 64x64 face images, 2-class softmax head.
    Three conv blocks give enough capacity for Smiling/Male-style
    attributes without needing a full ResNet."""
    l2 = regularizers.l2(weight_decay)

    model = Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, use_bias=False, kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(2, activation='softmax', kernel_regularizer=l2),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


fds = None  # Cache FederatedDataset (one per Ray actor process)
_num_clients_cache: int | None = None


def _pil_batch_to_array(pil_list, size, _chunk=2000):
    """Legacy path: receives a pre-materialised list of PIL objects.

    Kept for callers that already hold a Python list. New code should prefer
    _stream_partition_to_array which reads PIL chunks straight from the HF
    dataset and never holds the full set in RAM.

    Processes images in chunks of _chunk and nulls out each entry after
    conversion so PIL objects can be GC'd incrementally.
    """
    import gc
    n = len(pil_list)
    out = np.empty((n, size, size, 3), dtype=np.float32)
    for start in range(0, n, _chunk):
        end = min(start + _chunk, n)
        for i in range(start, end):
            img = pil_list[i]
            if hasattr(img, "resize"):
                arr = np.asarray(img.resize((size, size)), dtype=np.float32)
            else:
                arr = tf.image.resize(np.asarray(img, dtype=np.float32), (size, size)).numpy()
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            out[i] = arr
            pil_list[i] = None  # drop PIL reference so GC can reclaim it
        gc.collect()
    out /= 255.0
    return out


def _stream_partition_to_array(partition, size, _chunk=1000):
    """Materialise (image, target, sensitive) chunk-by-chunk from a HF partition.

    Memory-bounded: peak Python heap for PIL is ~_chunk images at a time
    (~117 KB each at 178x218x3 uint8). At _chunk=1000 that's ~117 MB peak
    versus list(partition["image"]) which decodes all N images at once
    (~4.7 GB for N=40k — enough to OOM the 23 GiB CrySyS server).

    Returns (images_float32_normalized, y_int32, s_int32).
    """
    import gc
    n = len(partition)
    images = np.empty((n, size, size, 3), dtype=np.float32)
    y = np.asarray(partition[TARGET_ATTR], dtype=np.int32)
    s = np.asarray(partition[SENSITIVE_ATTR], dtype=np.int32)

    for start in range(0, n, _chunk):
        end = min(start + _chunk, n)
        chunk = partition.select(range(start, end))
        pil_list = list(chunk["image"])
        for i, img in enumerate(pil_list):
            if hasattr(img, "resize"):
                arr = np.asarray(img.resize((size, size)), dtype=np.float32)
            else:
                arr = tf.image.resize(np.asarray(img, dtype=np.float32), (size, size)).numpy()
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            images[start + i] = arr
        del chunk, pil_list
        gc.collect()

    images /= 255.0
    return images, y, s


def _partition_cache_path(partition_id: int, num_partitions: int) -> Path:
    # SEED excluded: IidPartitioner and train_test_split(random_state=42) are
    # fully deterministic — all seeds share the same cached arrays.
    return _DISK_CACHE_DIR / f"n{num_partitions}_p{partition_id}_tr{NUM_TRAIN_SAMPLES}_te{NUM_TEST_SAMPLES}.npz"


def load_data(
    partition_id: int,
    num_partitions: int,
    return_sensitive: bool = False,
    sensitive_attr: str = SENSITIVE_ATTR,
    test_only: bool = False,
):
    """Return (x_train, y_train, x_test, y_test[, s_train, s_test]).

    Preprocessed arrays are cached to .partition_cache/celeba/ as .npz files.
    An exclusive file lock ensures only one process does the expensive
    download+PIL-resize work; all others (including parallel seed runs) block
    until the cache is ready, then load from disk.

    When test_only=True, x_train/y_train (and s_train if return_sensitive)
    are returned as None so eval scripts don't pay the ~1.5 GB cost of
    materialising the train tensor we never look at.
    """
    import gc, sys
    if sys.platform == "win32":
        import msvcrt, time
        def _acquire(fobj):
            while True:
                try:
                    msvcrt.locking(fobj.fileno(), msvcrt.LK_LOCK, 1)
                    return
                except OSError:
                    time.sleep(0.1)
    else:
        import fcntl
        def _acquire(fobj):
            fcntl.flock(fobj, fcntl.LOCK_EX)

    cache_path = _partition_cache_path(partition_id, num_partitions)
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Single dataset-wide lock (not per-partition): fds.load_partition()
    # internally calls datasets.shuffle() which writes a SHARED indices
    # cache file under HF_DATASETS_CACHE. Per-partition locks let 20 Ray
    # actors race on that shared file and corrupt it (pyarrow then sees
    # "schema message null"). Serialising the prep step across actors is
    # cheap because once the cache is warm, everyone takes the fast path.
    with open(_DISK_CACHE_DIR / "_dataset.lock", "w") as _lf:
        _acquire(_lf)
        if not cache_path.exists():
            global fds
            if fds is None:
                fds = FederatedDataset(
                    dataset="flwrlabs/celeba",
                    partitioners={"train": IidPartitioner(num_partitions=num_partitions)},
                )

            partition = fds.load_partition(partition_id, "train")

            # OOM guard: with num_partitions=1 the partition contains the full
            # ~200k CelebA train split. Materialising every PIL image into a
            # (N, 64, 64, 3) float32 array allocates ~10 GB before we even get
            # to train_test_split, which doubles it. The cache only ever stores
            # NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES rows, so subsample stratified
            # at the HF level BEFORE the PIL→numpy step. Cheap path: reading
            # one label column doesn't decode images.
            # train_test_split(test_size=0.2) needs N >= 37500 to yield 30k train
            # post-truncation; pad to 40k for headroom against rounding.
            _NEEDED = 40000
            if len(partition) > _NEEDED:
                y_all = np.asarray(partition[TARGET_ATTR], dtype=np.int32)
                keep_idx, _ = train_test_split(
                    np.arange(len(y_all)),
                    train_size=_NEEDED,
                    random_state=42,
                    stratify=y_all,
                )
                del y_all
                partition = partition.select(sorted(keep_idx.tolist()))
                del keep_idx; gc.collect()

            # Stream PIL→numpy in 1000-image chunks. The previous
            # list(partition["image"]) decoded all 40k PIL images at once and
            # OOM'd the 23 GiB CrySyS VM (peak ~4.7 GiB just for PIL objects,
            # plus the float32 output array, plus train_test_split copies).
            # Note: this assumes the caller wants both train and test slices.
            # Callers passing return_sensitive use the same `sensitive_attr`
            # column name that the streaming helper resolves via SENSITIVE_ATTR,
            # so we re-fetch the sensitive column here if a non-default name
            # was requested.
            images, y, s = _stream_partition_to_array(partition, IMG_SIZE)
            if sensitive_attr != SENSITIVE_ATTR:
                s = np.asarray(partition[sensitive_attr], dtype=np.int32)
            del partition; gc.collect()

            x_tr, x_te, y_tr, y_te, s_tr, s_te = train_test_split(
                images, y, s, test_size=0.2, random_state=42, stratify=y,
            )
            del images, y, s; gc.collect()

            np.savez_compressed(
                cache_path,
                x_train=x_tr[:NUM_TRAIN_SAMPLES], y_train=y_tr[:NUM_TRAIN_SAMPLES],
                x_test=x_te[:NUM_TEST_SAMPLES],   y_test=y_te[:NUM_TEST_SAMPLES],
                s_train=s_tr[:NUM_TRAIN_SAMPLES],  s_test=s_te[:NUM_TEST_SAMPLES],
            )

    d = np.load(cache_path)
    x_test  = tf.convert_to_tensor(d["x_test"],  dtype=tf.float32)
    y_test  = tf.convert_to_tensor(d["y_test"],  dtype=tf.int32)
    if test_only:
        x_train = y_train = None
        s_train = None
    else:
        x_train = tf.convert_to_tensor(d["x_train"], dtype=tf.float32)
        y_train = tf.convert_to_tensor(d["y_train"], dtype=tf.int32)
        s_train = d["s_train"] if return_sensitive else None
    if return_sensitive:
        return x_train, y_train, x_test, y_test, s_train, d["s_test"]
    return x_train, y_train, x_test, y_test


def get_num_clients():
    global _num_clients_cache
    if _num_clients_cache is None:
        cfg = toml.load(toml_file)
        _num_clients_cache = cfg["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
    return _num_clients_cache


config = toml.load(toml_file)


def set_num_clients(num_clients):
    config = toml.load(toml_file)
    config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] = num_clients
    with open(toml_file, "w") as f:
        toml.dump(config, f)


def _resolve_save_dir(strategy: str = "fedavg") -> str:
    """FedAvg saves to 420/<dataset>/..., FedProx to 420_fedprox/<dataset>/..."""
    s = (strategy or "fedavg").lower()
    if s == "fedavg":
        return path_to_save
    return os.path.join(str(BASE_DIR.parent), f"420_{s}", os.path.basename(path_to_save))



def save_client(model, client_num, partition_id, round_num, strategy="fedavg"):
    model_dir = os.path.join(_resolve_save_dir(strategy), str(client_num), str(SEED))
    model_path = os.path.join(model_dir, f"client_{partition_id}_round_{round_num}.keras")
    model.save(model_path)


def save_global_model(model, client_num, round_num, strategy="fedavg"):
    model_dir = os.path.join(_resolve_save_dir(strategy), str(client_num), str(SEED))
    if round_num == 0:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "global_model_round_0.keras")
        model.save(path)
    else:
        model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
        model.save(model_path)


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
