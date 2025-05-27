"""adult: A Flower / TensorFlow app."""


import os
#------------------------------------#
SEED = int(os.environ.get("seed"))
#------------------------------------#

import toml
import sys

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent  
toml_file = BASE_DIR / "pyproject.toml"

path_to_save = os.path.join(BASE_DIR.parent, "420" , "adult")


import pandas as pd
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
import tensorflow as tf
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


def load_model():
    set_seed(SEED)  # Set seed for reproducibility
    model = models.Sequential([
        layers.Input(shape=(6,), dtype='float32'),  # Input layer

        # Layer 1: Dense + BatchNorm + Dropout
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Layer 2: Dense + BatchNorm + Dropout
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Layer 3: Dense + BatchNorm
        layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),

        # Output Layer
        layers.Dense(1, activation='sigmoid')  # Binary classification
         
    ])
    set_seed(SEED)           
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

fds = None  # Cache FederatedDataset

#Partition has already been saved, can be loaded, otherwise it will be created

def load_data(partition_id, num_partitions):
    set_seed(SEED)  
    # Fetch dataset
    dataset = fetch_openml(data_id=1590, as_frame=True)
    df = dataset.data
    df["income"] = dataset.target  # Add target variable

    # Preprocess categorical and numerical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    df_encoded.columns = encoder.get_feature_names_out(categorical_cols)

    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

    # Combine processed features
    df_processed = pd.concat([df_encoded, df_scaled], axis=1).astype(np.float32) 
    df_processed["income"] = (df["income"] == ">50K").astype(np.float32)        
    
    df_processed = df_processed.sample(frac=1, random_state=42)  # Shuffle
    
    # Split into partitions
    partition_size = len(df_processed) // num_partitions
    partition = df_processed.iloc[partition_id*partition_size : (partition_id+1)*partition_size]
    
    # Train-test split
    train, test = train_test_split(partition, test_size=0.2, random_state=42)
    
    # Separate features and labels
    return (
        train.drop(columns=["income"]).values.astype(np.float32),
        train["income"].values.astype(np.float32),
        test.drop(columns=["income"]).values.astype(np.float32),
        test["income"].values.astype(np.float32)
    )


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