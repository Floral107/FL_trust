import os
import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from imdb.imdb.task import process_text # Assuming process_text is defined elsewhere and uses USE
from flwr.server.strategy import FedAvg
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from typing import List, Union



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

def calculate_robustness_score(model, x_test, dataset_name, sigma=0.1, L=100):
    """Calculate robustness score with memory-efficient processing"""
    try:
        if dataset_name in ("imdb", "imdbnoniid"):
            return calculate_text_robustness(model, x_test, L)
        elif dataset_name in ("cifar", "cifarnoniid"):
            return calculate_image_robustness(model, x_test, sigma, L)
        elif dataset_name in ("adult", "adultnoniid"):
            return calculate_tabular_robustness(model, x_test, sigma, L)
        else:
            print(f"Unsupported dataset: {dataset_name}")
            return 0.0
    except Exception as e:
        print(f"Error in calculate_robustness_score: {str(e)}")
        return 0.0

def calculate_image_robustness(model, x_test, sigma, L):
    D_size = tf.shape(x_test)[0]
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)

    # Predict for the entire test set once
    predictions_clean = tf.argmax(model.predict(x_test_tensor), axis=1)

    # Generate all noisy samples at once (N * L, ...)
    noise = tf.random.normal(shape=(L, *tf.shape(x_test_tensor)), mean=0.0, stddev=sigma, dtype=tf.float32)
    noisy_test_all = tf.clip_by_value(x_test_tensor + noise, 0.0, 1.0)

    # Reshape to (N * L, ...) for batched prediction
    noisy_test_reshaped = tf.reshape(noisy_test_all, (-1, *tf.shape(x_test_tensor)[1:]))

    # Predict for all noisy samples in a single batch
    predictions_noisy_all = tf.argmax(model.predict(noisy_test_reshaped), axis=1)

    # Reshape back to (L, N) for comparison
    predictions_noisy_reshaped = tf.reshape(predictions_noisy_all, (L, D_size))

    # Compare clean predictions with each set of noisy predictions
    mismatches = tf.cast(tf.not_equal(tf.expand_dims(predictions_clean, axis=0), predictions_noisy_reshaped), dtype=tf.float32)

    # Calculate the sum of mismatches across all noisy samples and average
    robustness_sum = tf.reduce_sum(mismatches)
    robustness_score = 1.0 - robustness_sum / (L * tf.cast(D_size, dtype=tf.float32))

    return robustness_score


def calculate_tabular_robustness(model, data, sigma, L):
    """Calculate robustness for tabular data using only TensorFlow tensors"""
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    preds_clean = tf.round(model(data))
    preds_clean = tf.reshape(preds_clean, [-1])

    mismatch_count = 0.0
    for _ in range(L):
        noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=sigma)
        noise = tf.clip_by_value(noise, -3.0 * sigma, 3.0 * sigma)
        noisy_data = data + noise

        preds_noisy = tf.round(model(noisy_data))
        preds_noisy = tf.reshape(preds_noisy, [-1])

        mismatch_count += tf.reduce_sum(
            tf.cast(preds_noisy != preds_clean, tf.float32)
        )

    robustness = 1.0 - mismatch_count / (L * tf.cast(tf.shape(data)[0], tf.float32))
    return robustness



def calculate_text_robustness(model, texts, L=100, batch_size=256):
    try:
        # 1. Process clean texts and get predictions (convert to tensors)
        clean_embeddings = tf.convert_to_tensor(process_text(texts))
        predictions_clean = tf.argmax(model.predict(clean_embeddings, batch_size=batch_size, verbose=0), axis=1)
        
        # 2. Generate and process noisy texts
        noisy_texts = []
        for text in texts:
            for _ in range(L):
                noisy_text = add_noise(text)
                noisy_texts.append(noisy_text)
        
        # Convert to tensor before processing
        noisy_texts_tensor = tf.convert_to_tensor(noisy_texts)
        noisy_embeddings = tf.convert_to_tensor(process_text(noisy_texts_tensor.numpy()))  # Process as numpy then convert back
        
        # Get predictions for noisy texts
        predictions_noisy = tf.argmax(model.predict(noisy_embeddings, batch_size=batch_size, verbose=0), axis=1)
        
        # 3. Reshape and compare using tensor operations
        predictions_noisy = tf.reshape(predictions_noisy, [len(texts), L])
        predictions_clean = tf.reshape(predictions_clean, [-1, 1])
        
        # Count mismatches
        mismatch_count = tf.reduce_sum(tf.cast(predictions_noisy != predictions_clean, tf.int32))
        robustness_score = 1.0 - mismatch_count / (len(texts) * L)
        
        return robustness_score
    
    except Exception as e:
        print(f"Error in robustness calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def add_noise(text, noise_level=0.1):
    """Adds character-level noise to a text string with more control"""
    if not text or not isinstance(text, str):
        return text
        
    chars = list(text)
    num_chars = len(chars)
    if num_chars == 0:
        return text
        
    # More controlled noise addition
    num_noisy_chars = max(1, min(num_chars, int(num_chars * noise_level)))
    indices = random.sample(range(num_chars), num_noisy_chars)
    
    for idx in indices:
        # Preserve uppercase if original was uppercase
        original_char = chars[idx]
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")
    
    return ''.join(chars)
