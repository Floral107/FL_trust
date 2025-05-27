"""imdb: A Flower / TensorFlow app."""


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

path_to_save = os.path.join(BASE_DIR.parent, "420" , "imdb")


from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import tensorflow as tf
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


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    


USE_MODEL = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=True)

def load_model():
    set_seed(SEED)
    inputs = layers.Input(shape=(512,), dtype=tf.float32)  # USE embeddings are 512-dimensional
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)  # Two output classes

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-5), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=["accuracy"])
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    set_seed(SEED)
    global fds
    # Only initialize the dataset once
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="stanfordnlp/imdb",
            partitioners={"train": partitioner},
        )

        # Get this client's partition
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    x, y = partition["text"], partition["label"]
        
    # Train-test split
    set_seed(SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Optional: truncate for fixed-size experiments
    num_train_samples = 1000
    num_test_samples = 500
    x_train, y_train = x_train[:num_train_samples], y_train[:num_train_samples]
    x_test, y_test = x_test[:num_test_samples], y_test[:num_test_samples]

    # Ensure labels are numpy arrays
    y_train = np.array(y_train, dtype=np.int8).flatten()
    y_test = np.array(y_test, dtype=np.int8).flatten()

    return x_train, y_train, x_test, y_test
    


def process_text(text_list):
    """Preprocesses text for Universal Sentence Encoder."""
    embeddings = USE_MODEL(text_list)
    return embeddings


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
    
