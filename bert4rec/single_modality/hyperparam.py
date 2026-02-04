#!/usr/bin/env python3
"""
BERT4Rec Hyperparameter search using random search
Usage: python hyperparam.py --data_path your_data.csv --n_trials 30 --num_users 2000
"""

import os
import json
import subprocess
import time
import random
import argparse
import sys
from datetime import datetime

def sample_hyperparameters(num_users=2000):
    """
    Sample random hyperparameters for BERT4Rec
    Based on common ranges used in research papers
    """
    
    hidden_dim = random.choice([32, 64, 128, 256])

    possible_heads = [h for h in [1, 2, 4, 8] if hidden_dim % h == 0 and h <= hidden_dim // 8]
    num_heads = random.choice(possible_heads) if possible_heads else 1
    
    params = {
        #model architecture
        'hidden_dim': hidden_dim,
        'num_layers': random.choice([1, 2, 3, 4, 6]),
        'num_heads': num_heads,
        'max_seq_len': random.choice([25, 50, 75, 100]),
        
        #training hyperparameters  
        'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01]),
        'batch_size': random.choice([64, 128, 256, 512]),
        'dropout_rate': round(random.uniform(0.1, 0.5), 3),
        'mask_prob': round(random.uniform(0.1, 0.3), 3),
        
        #training settings
        'num_epochs': 20, 
        'num_users': num_users  
    }
    
    return params

def run_experiment(data_path, params, exp_id, results_dir):
    """
    Run a single experiment with given hyperparameters
    """

    print(f"\nExperiment {exp_id}")
    print(f"Params: hidden_dim={params['hidden_dim']}, layers={params['num_layers']}, "
          f"heads={params['num_heads']}, lr={params['learning_rate']}, batch={params['batch_size']}")

    cmd = [sys.executable, 'train.py',
        '--data_path', data_path,
        '--hidden_dim', str(params['hidden_dim']),
        '--num_layers', str(params['num_layers']),
        '--num_heads', str(params['num_heads']),
        '--max_seq_len', str(params['max_seq_len']),
        '--learning_rate', str(params['learning_rate']),
        '--batch_size', str(params['batch_size']),
        '--dropout_rate', str(params['dropout_rate']),
        '--mask_prob', str(params['mask_prob']),
        '--num_epochs', str(params['num_epochs']),
        '--num_users', str(params['num_users'])
    ]

    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='.'
    )

    duration = time.time() - start_time

    hr10, ndcg10 = parse_metrics_from_output(result.stdout)

    print(f"HR@10: {hr10:.4f}, NDCG@10: {ndcg10:.4f} ({duration:.1f}s)")

    # Only save error logs if there was an error
    if result.returncode != 0 or hr10 == 0.0:
        trial_log_path = os.path.join(results_dir, f'trial_{exp_id}_error.log')
        with open(trial_log_path, 'w') as f:
            f.write(f"=== Trial {exp_id} FAILED ===\n")
            f.write(f"Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n=== Errors ===\n")
            f.write(result.stderr if result.stderr else "No error message")

    return {
        'trial': exp_id,
        'hr10': hr10,
        'ndcg10': ndcg10,
        'duration': duration,
        'params': params,
        'checkpoint_dir': None
    }

def parse_metrics_from_output(output_text):
    """
    Parse HR@10 and NDCG@10 from training script output
    """
    lines = output_text.split('\n')
    
    for line in lines:
        if 'Final Results:' in line:
            idx = lines.index(line)
            for i in range(idx, min(idx + 5, len(lines))):
                result_line = lines[i]
                if 'HR@10:' in result_line:
                    hr10 = float(result_line.split('HR@10:')[1].split(',')[0].strip())
                if 'NDCG@10:' in result_line:
                    ndcg10 = float(result_line.split('NDCG@10:')[1].strip())
            return hr10, ndcg10
    
    return 0.0, 0.0

def run_random_search(data_path, n_trials, results_dir, num_users):
    """
    Run random hyperparameter search
    """
    print(f"Number of trials: {n_trials}")
    print(f"Users per trial: {num_users}")
    
    all_results = []
    best_hr10 = 0.0
    best_result = None
    
    for trial in range(1, n_trials + 1):
        params = sample_hyperparameters(num_users=num_users)

        result = run_experiment(data_path, params, trial, results_dir)
        all_results.append(result)
        
        # Track best result
        if result['hr10'] > best_hr10:
            best_hr10 = result['hr10']
            best_result = result

            print(f"\n*** New best HR@10: {best_hr10:.4f} (Trial {trial}) ***\n")

            with open(os.path.join(results_dir, 'best_hyperparams.json'), 'w') as f:
                json.dump({
                    'best_hr10': best_hr10,
                    'best_ndcg10': result['ndcg10'],
                    'params': params,
                    'trial': trial
                }, f, indent=2)
        
        # Save all results incrementally
        with open(os.path.join(results_dir, 'all_search_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Progress: {trial}/{n_trials} trials completed")
    
    # Final summary
    print(f"Search done")

    if best_result is not None:
        print(f"Best result:")
        print(f"   Trial: {best_result['trial']}")
        print(f"   HR@10: {best_result['hr10']:.4f}")
        print(f"   NDCG@10: {best_result['ndcg10']:.4f}")
        print(f"\nBest hyperparam:")
        for key, value in best_result['params'].items():
            print(f"   {key}: {value}")

        print(f"\nResults saved to: {results_dir}/")
        print(f"  - best_hyperparams.json")
        print(f"  - all_search_results.json")
    else:
        print("Warning: No valid results found. All trials may have failed.")
        print(f"Check trial logs in {results_dir}/ for details")

    return all_results

def main():
    parser = argparse.ArgumentParser(description='BERT4Rec Hyperparameter search')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to csv dataset')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of random trials to run (default: 25)')
    parser.add_argument('--num_users', type=int, default=2000,
                        help='Number of users for hyperparameter search (default: 2000')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    #random seed
    random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hyperparam_search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run hyperparameter search
    results = run_random_search(args.data_path, args.n_trials, results_dir, args.num_users)

if __name__ == "__main__":
    main()