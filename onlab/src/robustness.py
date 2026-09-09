import argparse
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from cifar.cifar.task import load_data as load_cifar
from imdb.imdb.task import load_data as load_imdb
from adult.adult.task import load_data as load_adult
from adultnoniid.adultnoniid.task import load_data as load_adultnoniid
from cifarnoniid.cifarnoniid.task import load_data as load_cifarnoniid
from imdbnoniid.imdbnoniid.task import load_data as load_imdbnoniid
from adult.adult.task import set_seed
from plots import generate_all_visualizations, perform_statistical_analysis

from tqdm import tqdm
from cont_evals import evaluate_round

dataloaders = {
    "cifar": load_cifar,
    "imdb": load_imdb,
    "adult": load_adult,
    "imdbnoniid": load_imdbnoniid,
    "adultnoniid": load_adultnoniid,
    "cifarnoniid": load_cifarnoniid,
}

def parse_arguments():
    """Parse command line arguments for dataset and partition type."""
    parser = argparse.ArgumentParser(description='Evaluate federated learning robustness.')
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cifar', 'imdb', 'adult', 'cifarnoniid', 'imdbnoniid', 'adultnoniid'],
                       help='Dataset to evaluate (cifar/imdb/adult with noniid variants)')
    
    parser.add_argument('--partition', type=str, required=True,
                       choices=['iid', 'noniid'],
                       help='Partition type (iid or noniid)')
                       
    parser.add_argument('--num_rounds', type=int, required=True,
                       help='Number of training rounds to evaluate')
                       
    parser.add_argument('--seed', type=int, required=True,
                       help='Seed')
    
    parser.add_argument('--root_dir', type=str, default="/home/ulrich/420",
                       help='Root directory containing experiment results')
    

    return parser.parse_args()


def load_all_test_data(dataset):
    """Load test data for the specified dataset."""
    print(f"Loading test data for {dataset}...")
    
    test_data = {}
    y_data = {}
    
    # Load the requested dataset
    _, _, x_test, y_test = dataloaders[dataset](0, 1)
    
    test_data[dataset] = x_test
    y_data[dataset] = y_test
    
    print("Test data loading complete.")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Data type: {x_test.dtype}")
    print(f"Label type: {y_test.dtype}")
    print(f"first couple of y: {y_test[:5]}")
    print(np.mean(y_test))  

    return test_data, y_data

def evaluate_all_experiments(root_dir, test_data, y_data, dataset, partition, num_rounds):
    """Evaluate all experiments across datasets, partitions, and seeds."""
    args = parse_arguments()
    set_seed(args.seed)
    results = []
    for num_clients in [4, 20]:
        exp_dir = os.path.join(root_dir, dataset, str(num_clients), str(args.seed))
                
        if not os.path.exists(exp_dir):
                print(f"Directory not found: {exp_dir}")
                continue
  
                # Get test data for this dataset
        x_test = test_data[dataset]
        y_test = y_data[dataset]   
                # Evaluate each round
        for round_num in tqdm(range(1, num_rounds+1)):
            try:
                global_robustness, metrics, contributions = evaluate_round(
                        exp_dir, 
                        num_clients, 
                        x_test, 
                        y_test,
                        round_num, 
                        dataset,
                        "l1o"
                           )
                        
                if global_robustness is not None:
                    for client_id, contribution in contributions.items():
                            results.append({
                                'dataset': dataset.replace('noniid', ''),
                                'partition': partition,
                                'num_clients': num_clients,
                                'seed': args.seed,
                                'round': round_num,
                                'client_id': client_id,
                                'global_robustness': global_robustness,
                                'contribution': contribution,
                                'abs_contribution': abs(contribution),
                                **metrics
                                })
            except Exception as e:
                    print(f"Error evaluating {exp_dir} round {round_num}: {str(e)}")    
                
    
    return pd.DataFrame(results)


def convert_tensor(x):
    if isinstance(x, tf.Tensor):
        return float(x.numpy())  
    return x


def main():
    args = parse_arguments()
    
    test_data, y_data = load_all_test_data(args.dataset)
    
    results_df = evaluate_all_experiments(
        root_dir=args.root_dir,
        test_data=test_data,
        y_data=y_data,
        dataset=args.dataset,
        partition=args.partition,
        num_rounds=args.num_rounds
    )
    
    results_df = results_df.applymap(convert_tensor)
    output_file = f"robustness_results_{args.dataset}_{args.seed}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
