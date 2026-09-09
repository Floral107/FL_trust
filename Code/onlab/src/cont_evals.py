import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from robustness_metric import get_num_round, load_client_models, calculate_robustness_score, aggregate_models
from imdb.imdb.task import process_text

def evaluate_round(model_dir, num_clients, x_test, y_test, round_num, dataset, method="l1o"):
    """Evaluates robustness and accuracy for a specific round."""
    global_model_path = os.path.join(model_dir, f"global_model_round_{round_num}.keras")

    if not os.path.exists(global_model_path):
        print(f"Global model for round {round_num} not found.")
        return None, None, None  # (robustness, accuracy, contributions)

    try:
        # Load the global model
        global_model = tf.keras.models.load_model(global_model_path)

        global_robustness = calculate_robustness_score(global_model, x_test, dataset)  
        
        # Process input data
        if dataset.lower() in ['imdb', 'imdbnoniid']:
            x_test_processed = process_text(x_test)
        else:    
            x_test_processed = x_test  


        # Calculate predictions and accuracy
        global_preds = global_model.predict(x_test_processed, verbose=0)

        # Process input data
        if dataset.lower() in ['cifar', 'cifarnoniid', 'imdb', 'imdbnoniid']:  # Multi-class
            predicted_labels = np.argmax(global_preds, axis=1)
            y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            if dataset.lower() in ['cifar', 'cifarnoniid',]:
                averageing_method = 'macro'
            else:
                averageing_method = 'binary'
        else:
            predicted_labels = (global_preds > 0.5).astype(int).flatten()
            y_true = y_test.flatten()
            #auc = roc_auc_score(y_true, global_preds.flatten())
            loss_fn = tf.keras.losses.BinaryCrossentropy()
            averageing_method = 'binary'
        # Compute loss manually
        loss = loss_fn(y_true, global_preds).numpy()

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_true, predicted_labels),
            "precision": precision_score(y_true, predicted_labels, average=averageing_method),
            "recall": recall_score(y_true, predicted_labels, average=averageing_method),
            "f1": f1_score(y_true, predicted_labels, average=averageing_method),
            "loss": loss
            }
        
        # Print all metrics
        print(f"Round {round_num} | Robustness: {global_robustness:.4f}")
        print(
            f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | "
            f"Loss: {metrics['loss']:.4f}"
              )
        # Client contribution analysis (robustness only)
        client_models = load_client_models(model_dir, num_clients, round_num)
        client_weights = {cid: model.get_weights() for cid, model in client_models.items()}

        if method == "l1o":
            contributions = {}
            for client_id in client_models.keys():
                remaining_clients = [client_weights[cid] for cid in client_models.keys() if cid != client_id]
                if not remaining_clients:
                    continue
                
                try:
                    temp_model = tf.keras.models.clone_model(global_model)
                    temp_model.set_weights(aggregate_models(remaining_clients))
                    if dataset.lower() in ['adult', 'adultnoniid',]:
                        temp_model.compile(optimizer='adam', loss='binary_crossentropy')
                    else:
                        temp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                    
                    loo_robustness = calculate_robustness_score(temp_model, x_test, dataset)
                    contributions[client_id] = global_robustness - loo_robustness
                except Exception as e:
                    print(f"Error in client {client_id} calc: {str(e)}")
            
            return global_robustness, metrics, contributions

        return global_robustness, metrics, None

    except Exception as e:
        print(f"Error in global model eval: {str(e)}")
        return None, None, None