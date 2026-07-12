"""adultnoniid: A Flower / TensorFlow app."""


import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#

import toml
import sys

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent  
toml_file = BASE_DIR / "pyproject.toml"

path_to_save = os.path.join(BASE_DIR.parent, "420" , "adultnoniid")


import pandas as pd
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import tensorflow as tf
# Cap per-process TF threading (see celebanoniid/task.py for rationale):
# prevents each Ray actor from grabbing all cores for tiny ops.
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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras import layers, models, optimizers, regularizers
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
import json



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def dirichlet_partition(y_data, num_partitions, alpha=0.5):
    """Partition data indices using Dirichlet distribution."""
    partition_indices = [[] for _ in range(num_partitions)]
    y_data = np.array(y_data)

    for class_label in np.unique(y_data):
        class_indices = np.where(y_data == class_label)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_partitions)
        class_counts = (proportions * len(class_indices)).astype(int)
        
        # Fix rounding errors
        while class_counts.sum() < len(class_indices):
            class_counts[np.argmin(class_counts)] += 1
        while class_counts.sum() > len(class_indices):
            class_counts[np.argmax(class_counts)] -= 1

        # Split and assign
        start = 0
        for i in range(num_partitions):
            end = start + class_counts[i]
            partition_indices[i].extend(class_indices[start:end])
            start = end

    return partition_indices

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

def apply_randomized_response(series: pd.Series, p_keep: float) -> pd.Series:
    categories = series.unique()
    d = len(categories)
    values = series.to_numpy()
    n = len(values)

    # random U(0,1)
    u = np.random.rand(n)
    keep_mask = u < p_keep
    flip_mask = ~keep_mask

    out = values.copy()

    if flip_mask.any():
        # kategória -> index
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        idx_to_cat = np.array(categories)

        original_idx = np.array([cat_to_idx[v] for v in values[flip_mask]])

        # sorsolás 0..d-2 közül
        r = np.random.randint(0, d - 1, size=original_idx.shape[0])
        # ha nagyobb vagy egyenlő az eredetivel, toljuk +1-gyel (így kihagyjuk az eredetit)
        r += (r >= original_idx).astype(int)

        new_vals = idx_to_cat[r]
        out[flip_mask] = new_vals

    return pd.Series(out, index=series.index)


def load_model(seed=SEED, input_shape=108, lr=1e-3, l2=1e-4, clipnorm=1.0):
    # reproducibility
    set_seed(seed)

    initializer = tf.keras.initializers.HeNormal(seed=seed)

    model = models.Sequential([
        layers.Input(shape=(input_shape,), dtype='float32'),

        # wider first layer (tabular data often benefits), LayerNormalization instead of BatchNorm
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     kernel_initializer=initializer),
        layers.LayerNormalization(),
        layers.Dropout(0.1),  # smaller dropout

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     kernel_initializer=initializer),
        layers.LayerNormalization(),
        layers.Dropout(0.05),

        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     kernel_initializer=initializer),
        layers.LayerNormalization(),

        layers.Dense(1, activation='sigmoid')
    ])

    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

fds = None  # Cache FederatedDataset

#Partition has already been saved, can be loaded, otherwise it will be created


def load_data(
    partition_id: int,
    num_partitions: int,
    return_sensitive: bool = False,
    sensitive_attr: str = "sex",
    sigma: float = None,  # <--- NEW ARGUMENT for Gaussian Noise Scale
    return_meta: bool = False
):
    np.random.seed(SEED)

    # 1. Fetch dataset
    dataset = fetch_openml(data_id=1590, as_frame=True)
    df = dataset.data.copy()
    df["income"] = dataset.target

    # 2. Build sensitive attribute S from raw columns
    if sensitive_attr == "sex":
        s = (df["sex"].str.strip() == "Female").astype(np.int32)
    elif sensitive_attr == "race":
        s = (df["race"].str.strip() == "White").astype(np.int32)
    else:
        raise ValueError(f"Unsupported sensitive_attr={sensitive_attr}")

    # 3. Prepare features
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols   = df.select_dtypes(include=np.number).columns.tolist()


    if "income" in categorical_cols:
        categorical_cols.remove("income")
    if "income" in numerical_cols: # Biztonság kedvéért
        numerical_cols.remove("income")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler  = StandardScaler()

    df_encoded = pd.DataFrame(
        encoder.fit_transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
    )
    
    # ===== OHE META =====
    cat_sizes = [len(c) for c in encoder.categories_]   # pl. [8, 5, 14, ...]
    cat_slices = []
    start = 0
    for size in cat_sizes:
        cat_slices.append(slice(start, start + size))
        start += size
    n_cat = start  # első n_cat oszlop: OHE
    # ====================



    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numerical_cols]),
        columns=numerical_cols,
    )

    # 4. Combine features + labels + sensitive
    df_processed = pd.concat([df_encoded, df_scaled], axis=1).astype(np.float32)
    df_processed["income"]    = (df["income"].str.strip() == ">50K").astype(np.float32)
    df_processed["sensitive"] = s.astype(np.float32)

    # 5. Shuffle once
    df_processed = df_processed.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 6. Partition
    partition_indices = dirichlet_partition(df_processed["income"].values, num_partitions, 0.5)
    partition_data = [df_processed.iloc[indices] for indices in partition_indices]

    client_data = partition_data[partition_id]

    # 7. Train–test split
    train, test = train_test_split(client_data, test_size=0.2, random_state=42)

    feature_cols = [c for c in df_processed.columns if c not in ["income", "sensitive"]]

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train["income"].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)
    y_test  = test["income"].values.astype(np.float32)

    s_train = train["sensitive"].values.astype(np.int32)
    s_test  = test["sensitive"].values.astype(np.int32)
    
    meta = {"cat_slices": cat_slices, "n_cat": n_cat}

    if return_sensitive and return_meta:
        return X_train, y_train, X_test, y_test, s_train, s_test, meta
    elif return_sensitive: 
        return X_train, y_train, X_test, y_test, s_train, s_test
    elif return_meta:
        return X_train, y_train, X_test, y_test, meta
    else:
        return X_train, y_train, X_test, y_test



def get_num_clients():
    # Load the pyproject.toml file
    config = toml.load(toml_file)

    # Extracting values
    num_clients = config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
    return num_clients
config = toml.load(toml_file)

def set_num_clients(num_clients, toml_file=toml_file):
    
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

    ET takes precedence over weight-mode (ET runs always use plain FedAvg).
    Weight-mode takes precedence over fedprox (they're mutually exclusive
    strategy choices).
    """
    em = (et_mode or "none").lower()
    if em in ("fair", "adv", "dp"):
        return os.path.join(str(BASE_DIR.parent), f"420_et_{em}", os.path.basename(path_to_save))
    wm = (weight_mode or "none").lower()
    if wm in ("st", "dy") and weight_metric:
        wmet = str(weight_metric).lower()
        # FLR_SAVE_PREFIX redirects ST/DY checkpoint trees (default "420" =
        # canonical). Set by transform-A/B sandboxes (st_dy_guarded_run OUTPRE)
        # so ablation runs never overwrite canonical models.
        prefix = os.environ.get("FLR_SAVE_PREFIX", "420")
        return os.path.join(
            str(BASE_DIR.parent),
            f"{prefix}_{wm}_{wmet}",
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
    # If save_global_model(round=0) hasn't run yet (or this ET combo never
    # produced a round-0 ckpt before), the dir won't exist and model.save
    # raises FileNotFoundError. Cheap fix: ensure it now.
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
from et_training import train_with_fair, train_with_adv, train_with_dpsgd  # noqa: E402