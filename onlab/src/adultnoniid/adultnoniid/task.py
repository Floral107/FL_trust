"""adultnoniid: A Flower / TensorFlow app."""

import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#

import toml
import pickle  # Changed from json to pickle

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
tf.random.set_seed(SEED)
from keras import layers, models, regularizers, optimizers
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def load_model():
    """Create and compile the neural network model."""
    set_seed(SEED)
    model = models.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


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


def load_data(partition_id, num_partitions, alpha=0.5):
    """Load or create partitioned data for a specific client."""
    set_seed(SEED)
    
    # Load and preprocess data
    dataset = fetch_openml(data_id=1590, as_frame=True)
    df = dataset.frame
    df["income"] = dataset.target
    
    # Preprocessing
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=np.number).columns


    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    df_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)  
    df_processed = pd.concat([df_encoded, df_scaled], axis=1).astype(np.float32) 
    df_processed["income"] = (df["income"] == ">50K").astype(np.float32)        


    partition_indices = dirichlet_partition(df_processed["income"].values, num_partitions, alpha)
    partition_data = [df_processed.iloc[indices] for indices in partition_indices]

    client_data = partition_data[partition_id]
    train_df, test_df = train_test_split(client_data, test_size=0.2, random_state=42)
    
    return (
        train_df.drop(columns=["income"]).values.astype(np.float32),
        train_df["income"].values.astype(np.float32),
        test_df.drop(columns=["income"]).values.astype(np.float32),
        test_df["income"].values.astype(np.float32)
    )
    


def get_num_clients():
    """Get number of clients from config file."""
    config = toml.load(toml_file)
    return config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]


def set_num_clients(num_clients):
    """Update number of clients in config file."""
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