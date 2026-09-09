"""cifarnoniid: A Flower / TensorFlow app."""



import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#


import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.random.set_seed(SEED)

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent  
toml_file = BASE_DIR / "pyproject.toml"
path_to_save = os.path.join(BASE_DIR.parent, "420", "cifarnoniid")

import keras
tf.keras.utils.set_random_seed(SEED)

import toml
from keras import layers, optimizers
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


def load_model():
    set_seed(SEED)
    initializer = tf.keras.initializers.he_normal(seed=SEED)
    regularizer = tf.keras.regularizers.l2(1e-4)

    model = Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model




fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    set_seed(SEED)  # Ensure reproducibility

    global fds


        # Init dataset once
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.5, partition_by="label", seed=SEED)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    x, y = partition["img"], partition["label"]

    x = (x / 255.0).astype(np.float32)        
        

    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
            
    
    num_train_samples = 3000
    num_test_samples = 1000
    x_train, y_train = x_train[:num_train_samples], y_train[:num_train_samples]
    x_test, y_test = x_test[:num_test_samples], y_test[:num_test_samples]

    
    
    return tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int32), \
       tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.int32)




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

def save_client(model, client_num, partition_id, round_num):
    """Save the model of a client after training."""
    model_dir = os.path.join(path_to_save, str(client_num), str(SEED))
    model_path = os.path.join(model_dir, f"client_{partition_id}_round_{round_num}.keras")
    model.save(model_path)
    
def save_global_model(model, client_num, round_num):
    """Save the global model after training."""
    model_dir = os.path.join(path_to_save, str(client_num), str(SEED))
    if round_num == 0:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "global_model_round_0.keras")
        model.save(path)
    else:
        model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")
        model.save(model_path)



