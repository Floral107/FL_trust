# -*- coding: utf-8 -*-
"""Robustness Evaluation with Label Swapping on IMDB Dataset"""

# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the IMDB dataset
imdb_data = load_dataset("imdb")

# Prepare the dataset as a list of (text, label) tuples for training and testing
# You can adjust the range to use more data
train_data = [(example['text'], example['label']) for example in imdb_data['train'].select(range(1000))]
test_data = [(example['text'], example['label']) for example in imdb_data['test'].select(range(500))]

# Set the dataset to be used for the project
dataset = train_data  # Use train_data as the main dataset for robustness evaluation

# Verify the format
print("First 5 samples of the dataset:")
for sample in dataset[:5]:
    print(sample)
    print("-" * 80)

# Load Universal Sentence Encoder (USE) from TensorFlow Hub
print("Loading Universal Sentence Encoder (this may take a moment)...")
use_model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

def preprocess_text(text_list):
    """
    Preprocesses text for Universal Sentence Encoder by embedding the text list.
    """
    # Embed the text list
    embeddings = use_model(text_list)
    return embeddings

def add_noise(text, noise_level=1.0):
    """
    Adds character-level noise to a text string by randomly replacing characters.
    """
    chars = list(text)
    num_noisy_chars = max(1, int(len(chars) * noise_level))

    for _ in range(num_noisy_chars):
        idx = random.randint(0, len(chars) - 1)
        # Replace with a random lowercase letter or space
        chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")

    return ''.join(chars)

# Test the noise injection function
sample_text = "This is a sample text for robustness evaluation."
noisy_text = add_noise(sample_text, noise_level=1.0)
print("\nTesting add_noise function:")
print("Original:", sample_text)
print("Noisy:", noisy_text)
print("-" * 80)

# Swap labels for 5% of the dataset once
total_samples = len(dataset)
num_swaps = int(total_samples * 0.05)
swap_indices = random.sample(range(total_samples), num_swaps)

# Apply label swapping
for idx in swap_indices:
    text, label = dataset[idx]
    possible_labels = [0, 1]
    possible_labels.remove(label)
    swapped_label = possible_labels[0]
    dataset[idx] = (text, swapped_label)

def create_random_subset(data, inclusion_prob=0.5):
    """
    Creates a random subset of data with each element having a specified chance of being included.
    Returns:
    - subset: List of (text, label) tuples.
    - included_indices: List of indices in the original dataset of the included samples.
    """
    subset = []
    included_indices = []
    for idx, item in enumerate(data):
        if random.random() < inclusion_prob:
            subset.append(item)
            included_indices.append(idx)
    return subset, included_indices

def train_model_on_subset(subset_data, epochs=1, batch_size=4):
    """
    Trains a model on USE embeddings from a given subset of data.
    """
    # Prepare the data for training
    texts = [text for text, _ in subset_data]
    labels = np.array([label for _, label in subset_data])

    # Preprocess texts into embeddings
    embeddings = preprocess_text(texts)

    # Build the model using USE embeddings
    inputs = Input(shape=(512,), dtype=tf.float32)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=2e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model on the subset
    model.fit(embeddings, labels, epochs=epochs, batch_size=batch_size, verbose=0)

    return model

def get_predictions(model, all_texts, L=100, batch_size=16):
    """
    Gets predictions for clean and noisy texts.
    Returns:
    - predictions_clean: Array of predictions for clean texts.
    - predictions_noisy: Array of predictions for noisy texts reshaped to (num_samples, L).
    """
    num_samples = len(all_texts)

    # Embed all original texts
    embeddings_clean = preprocess_text(all_texts)

    # Get predictions for clean embeddings
    predictions_clean = model.predict(embeddings_clean, batch_size=batch_size, verbose=0)
    predictions_clean = np.argmax(predictions_clean, axis=1)

    # Generate noisy texts
    noisy_texts = []
    for text in all_texts:
        noisy_versions = [add_noise(text, noise_level=0.1) for _ in range(L)]
        noisy_texts.extend(noisy_versions)

    # Embed noisy texts in batches to reduce memory usage
    embeddings_noisy = []
    for i in range(0, len(noisy_texts), batch_size):
        batch_noisy_texts = noisy_texts[i:i+batch_size]
        batch_embeddings = preprocess_text(batch_noisy_texts)
        embeddings_noisy.append(batch_embeddings.numpy())
    embeddings_noisy = np.concatenate(embeddings_noisy, axis=0)

    # Get predictions for noisy embeddings
    predictions_noisy = []
    for i in range(0, len(embeddings_noisy), batch_size):
        batch_embeddings = embeddings_noisy[i:i+batch_size]
        batch_predictions = model.predict(batch_embeddings, batch_size=batch_size, verbose=0)
        predictions_noisy.extend(np.argmax(batch_predictions, axis=1))
    predictions_noisy = np.array(predictions_noisy)

    # Reshape noisy predictions to (num_samples, L)
    predictions_noisy = predictions_noisy.reshape(num_samples, L)

    return predictions_clean, predictions_noisy

# Initialize arrays for accumulating scores and counts
num_samples = len(dataset)
R_plus = np.zeros(num_samples)
R_minus = np.zeros(num_samples)
C = np.zeros(num_samples)
Swapped_counts = np.zeros(num_samples)

# Initialize 2D array to store robustness scores per sample per iteration
K = 40   # Number of subsets/models to train
robustness_scores_per_iteration = np.zeros((num_samples, K))

# Parameters
L = 100   # Number of noisy variations per sample

# Loop over subsets and calculate robustness scores
for k in range(K):
    subset, included_indices = create_random_subset(dataset, inclusion_prob=0.5)
    # Identify swapped indices in the included samples
    swapped_indices = [idx for idx in included_indices if idx in swap_indices]

    print(f"\nTraining model {k+1}/{K} on subset with {len(subset)} samples.")
    print(f"Number of label-swapped samples in this subset: {len(swapped_indices)}")

    # Train model on the subset
    model = train_model_on_subset(subset)

    all_texts = [text for text, _ in dataset]

    # Get predictions
    predictions_clean, predictions_noisy = get_predictions(model, all_texts, L=L, batch_size=16)

    # Calculate robustness for each sample
    R_sample = np.zeros(num_samples)
    for i in range(num_samples):
        robustness_score = np.sum(predictions_noisy[i] != predictions_clean[i]) / L
        R_sample[i] = robustness_score

    # Store per-sample robustness scores for this iteration
    robustness_scores_per_iteration[:, k] = R_sample

    # Update R_plus, R_minus, C, Swapped_counts
    for i in range(num_samples):
        if i in included_indices:
            R_plus[i] += R_sample[i]
            C[i] += 1
            if i in swapped_indices:
                Swapped_counts[i] += 1
        else:
            R_minus[i] += R_sample[i]

# Calculate final robustness scores
R_plus_component = np.where(C > 0, R_plus / C, 0)
R_minus_component = np.where((K - C) > 0, R_minus / (K - C), 0)
robustness_scores = R_minus_component - R_plus_component

# Convert to float32
robustness_scores = robustness_scores.astype(np.float32)

# Save the robustness scores to npy files
np.save('robustness_scores.npy', robustness_scores)
np.save('robustness_scores_per_iteration.npy', robustness_scores_per_iteration)
np.save('swapped_counts.npy', Swapped_counts)
print("\nRobustness scores and swapped counts saved.")

# Create DataFrame
df = pd.DataFrame({
    'Index': range(num_samples),
    'Robustness Score': robustness_scores,
    'Text': [dataset[i][0] for i in range(num_samples)],
    'Label': [dataset[i][1] for i in range(num_samples)],
    'Swapped Count': Swapped_counts
})

# Add a column indicating if the sample was ever label-swapped
df['Label Swapped'] = df['Index'].isin(swap_indices)

# Sort the DataFrame based on 'Robustness Score'
df_sorted = df.sort_values('Robustness Score')

# Get top 100 worst robustness scores
worst_100_df = df_sorted.head(100)

# Get top 100 best robustness scores
best_100_df = df_sorted.tail(100)

# Display the top 100 worst robustness scores with swap information
print("\nTop 100 Worst Robustness Scores:")
for index, row in worst_100_df.iterrows():
    print(f"Index: {row['Index']}, Score: {row['Robustness Score']:.4f}, Label: {row['Label']}, Label Swapped: {row['Label Swapped']}")
    print(f"Text: {row['Text'][:200]}...")  # Display first 200 characters
    print("-" * 80)

# Display the top 100 best robustness scores with swap information
print("\nTop 100 Best Robustness Scores:")
for index, row in best_100_df.iterrows():
    print(f"Index: {row['Index']}, Score: {row['Robustness Score']:.4f}, Label: {row['Label']}, Label Swapped: {row['Label Swapped']}")
    print(f"Text: {row['Text'][:200]}...")
    print("-" * 80)

# Save to CSV with swap information
worst_100_df.to_csv('top_100_worst_with_swap.csv', index=False)
best_100_df.to_csv('top_100_best_with_swap.csv', index=False)
print("\nTop 100 best and worst robustness scores saved to CSV files.")

# Analyze label-swapped samples vs. non-swapped samples
swapped_samples = df[df['Label Swapped']]
non_swapped_samples = df[~df['Label Swapped']]

# Get robustness scores
robustness_scores_swapped = swapped_samples['Robustness Score'].dropna()
robustness_scores_non_swapped = non_swapped_samples['Robustness Score'].dropna()

# Calculate means
mean_swapped = robustness_scores_swapped.mean()
mean_non_swapped = robustness_scores_non_swapped.mean()

print(f"\nAverage robustness score of label-swapped samples: {mean_swapped:.4f}")
print(f"Average robustness score of non-swapped samples: {mean_non_swapped:.4f}")

# Mann-Whitney U Test
stat, p_value = mannwhitneyu(robustness_scores_swapped, robustness_scores_non_swapped, alternative='two-sided')
print('\nMann-Whitney U Test Results:')
print(f'Statistic: {stat:.2f}')
print(f'p-value: {p_value:.4f}')

# Effect Size (Cohen's d)
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / dof)
    return (x.mean() - y.mean()) / pooled_std

effect_size = cohen_d(robustness_scores_swapped, robustness_scores_non_swapped)
print(f"Cohen's d effect size: {effect_size:.2f}")

# Plot histograms
plt.figure(figsize=(12, 6))
plt.hist(robustness_scores_non_swapped, bins=50, alpha=0.5, label='Non-swapped Samples')
plt.hist(robustness_scores_swapped, bins=50, alpha=0.5, label='Label-swapped Samples')
plt.legend()
plt.xlabel('Robustness Score')
plt.ylabel('Number of Samples')
plt.title('Histogram of Robustness Scores')
plt.show()

# Boxplots
plt.figure(figsize=(8, 6))
data_to_plot = [robustness_scores_non_swapped, robustness_scores_swapped]
plt.boxplot(data_to_plot, labels=['Non-swapped', 'Label-swapped'])
plt.ylabel('Robustness Score')
plt.title('Boxplot of Robustness Scores')
plt.show()

# Save histograms to image files
plt.figure(figsize=(12, 6))
plt.hist(robustness_scores_non_swapped, bins=50, alpha=0.5, label='Non-swapped Samples')
plt.hist(robustness_scores_swapped, bins=50, alpha=0.5, label='Label-swapped Samples')
plt.legend()
plt.xlabel('Robustness Score')
plt.ylabel('Number of Samples')
plt.title('Histogram of Robustness Scores')
plt.tight_layout()
plt.savefig('robustness_scores_histogram.png', dpi=300)
plt.close()

# Save boxplots to image files
plt.figure(figsize=(8, 6))
data_to_plot = [robustness_scores_non_swapped, robustness_scores_swapped]
plt.boxplot(data_to_plot, labels=['Non-swapped', 'Label-swapped'])
plt.ylabel('Robustness Score')
plt.title('Boxplot of Robustness Scores')
plt.tight_layout()
plt.savefig('robustness_scores_boxplot.png', dpi=300)
plt.close()

# Save statistical test results to a text file
with open('statistical_results.txt', 'w') as f:
    f.write('Mann-Whitney U Test Results:\n')
    f.write(f'Statistic: {stat:.2f}\n')
    f.write(f'p-value: {p_value:.4f}\n')
    if p_value < 0.05:
        f.write('Result: Reject the null hypothesis (significant difference).\n')
    else:
        f.write('Result: Fail to reject the null hypothesis (no significant difference).\n')
    f.write(f"Cohen's d effect size: {effect_size:.2f}\n\n")
    f.write(f"Average robustness score of label-swapped samples: {mean_swapped:.4f}\n")
    f.write(f"Average robustness score of non-swapped samples: {mean_non_swapped:.4f}\n")

print("\nPlots saved as 'robustness_scores_histogram.png' and 'robustness_scores_boxplot.png'.")
print("Statistical test results saved in 'statistical_results.txt'.")

# Save the full DataFrame to a CSV file
df.to_csv('robustness_scores_full.csv', index=False)
print("Full robustness scores DataFrame saved to 'robustness_scores_full.csv'.")
