"""cifarnoniid: A Flower / TensorFlow app."""



import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#


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
path_to_save = os.path.join(BASE_DIR.parent, "420", "cifarnoniid")

import keras
tf.keras.utils.set_random_seed(SEED)

import toml
from keras import layers, optimizers, regularizers
from keras import Sequential
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)

import pandas as pd
import pickle
import toml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()  # Ensures deterministic GPU behavior
    
set_seed(SEED)  # Set seed for reproducibility

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def load_model(weight_decay=5e-4):
    l2 = regularizers.l2(weight_decay)

    model = Sequential([
        layers.Input(shape=(32, 32, 3)),
        
        # Block 1: Fast, lightweight feature detection
        layers.Conv2D(
            32, (3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=l2
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2: Slightly deeper features
        layers.Conv2D(
            64, (3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=l2
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Head: lightweight
        layers.Flatten(),
        layers.Dense(
            64,
            use_bias=False,
            kernel_regularizer=l2
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(
            10,
            activation='softmax',
            kernel_regularizer=l2
        ),
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model




fds = None  # Cache FederatedDataset


def _make_grayscale_subset(x, y, frac=0.05):
   
    x = x.copy()
    num_samples = x.shape[0]

    # Biztons�g kedv��rt int32-re hozzuk a labelt
    y = np.asarray(y, dtype=np.int32)

    # Egyedi oszt�lyok (CIFAR-10-ben 0..9)
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue

        k = int(np.round(len(idx) * frac))
        if k < 1:
            k = 1  # legal�bb 1 legyen, ha van minta

        chosen = np.random.choice(idx, size=k, replace=False)

        # Kiv�lasztott k�pek: (k, 32, 32, 3)
        imgs = x[chosen]  # float16 [0, 1]

        # Grayscale: �tlag a 3 csatorna ment�n
        gray = imgs.mean(axis=-1, keepdims=True)  # (k, 32, 32, 1)

        # Visszam�soljuk 3 csatorn�ra (R=G=B)
        x[chosen] = np.repeat(gray, 3, axis=-1)

    return x


def load_data(partition_id, num_partitions):
    set_seed(SEED)  # Ensure reproducibility

    global fds

    # Init dataset once
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            alpha=0.5,
            partition_by="label",
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    x, y = partition["img"], partition["label"]

    # Normaliz�l�s [0,1] �s float16
    x = (x / 255.0).astype(np.float16)

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Ha akarsz limitet:
    num_train_samples = 30000
    num_test_samples = 8000
    x_train, y_train = x_train[:num_train_samples], y_train[:num_train_samples]
    x_test, y_test = x_test[:num_test_samples], y_test[:num_test_samples]

    # >>> Itt alak�tjuk grayscale-l� a k�pek 5%-�t oszt�lyonk�nt <<<
    x_train = _make_grayscale_subset(x_train, y_train, frac=0.05)
    x_test = _make_grayscale_subset(x_test, y_test, frac=0.05)

    # Vissza: tensork�nt, ahogy eddig
    return (
        tf.convert_to_tensor(x_train, dtype=tf.float32),
        tf.convert_to_tensor(y_train, dtype=tf.int32),
        tf.convert_to_tensor(x_test, dtype=tf.float32),
        tf.convert_to_tensor(y_test, dtype=tf.int32),
    )



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

def _resolve_save_dir(strategy: str = "fedavg") -> str:
    """FedAvg saves to 420/<dataset>/..., FedProx to 420_fedprox/<dataset>/..."""
    s = (strategy or "fedavg").lower()
    if s == "fedavg":
        return path_to_save
    return os.path.join(str(BASE_DIR.parent), f"420_{s}", os.path.basename(path_to_save))



def train_with_proximal(model, x, y, mu, epochs, batch_size, verbose=0, shuffle=False):
    """
    FedProx local training: minimize task_loss + (mu/2) * sum ||w - w_global||^2.
    Call AFTER model.set_weights(global_parameters).
    """
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


def save_client(model, client_num, partition_id, round_num, strategy="fedavg"):
    """Save the model of a client after training."""
    model_dir = os.path.join(_resolve_save_dir(strategy), str(client_num), str(SEED))
    model_path = os.path.join(model_dir, f"client_{partition_id}_round_{round_num}.keras")
    model.save(model_path)


def save_global_model(model, client_num, round_num, strategy="fedavg"):
    """Save the global model after training."""
    model_dir = os.path.join(_resolve_save_dir(strategy), str(client_num), str(SEED))
    if round_num == 0:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "global_model_round_0.keras")
        model.save(path)
    else:
        model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
        model.save(model_path)



