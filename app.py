from flask import Flask, render_template, jsonify
import subprocess
import pandas as pd
import os
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def parse_metrics(output):
    """Extract metrics from training output"""
    metrics = {}
    
    # Extract train metrics
    train_match = re.search(r'=== TRAIN METRICS ===\s+Rows:\s+(\d+)\s+Accuracy:\s+([\d.]+)\s+Log loss:\s+([\d.]+)\s+AUC:\s+([\d.]+)', output)
    if train_match:
        metrics['train'] = {
            'rows': train_match.group(1),
            'accuracy': train_match.group(2),
            'log_loss': train_match.group(3),
            'auc': train_match.group(4)
        }
    
    # Extract val metrics
    val_match = re.search(r'=== VAL METRICS ===\s+Rows:\s+(\d+)\s+Accuracy:\s+([\d.]+)\s+Log loss:\s+([\d.]+)\s+AUC:\s+([\d.]+)', output)
    if val_match:
        metrics['val'] = {
            'rows': val_match.group(1),
            'accuracy': val_match.group(2),
            'log_loss': val_match.group(3),
            'auc': val_match.group(4)
        }
    
    # Extract test metrics
    test_match = re.search(r'=== TEST METRICS ===\s+Rows:\s+(\d+)\s+Accuracy:\s+([\d.]+)\s+Log loss:\s+([\d.]+)\s+AUC:\s+([\d.]+)', output)
    if test_match:
        metrics['test'] = {
            'rows': test_match.group(1),
            'accuracy': test_match.group(2),
            'log_loss': test_match.group(3),
            'auc': test_match.group(4)
        }
    
    return metrics

@app.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    try:
        results = {}
        
        # Step 1: Enrich with NBA API
        result = subprocess.run('python src/enrich_with_nba_api.py', 
                              shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Failed at enrich_with_nba_api.py', 'details': result.stderr}), 500
        
        # Extract season info
        seasons_match = re.search(r'Seasons found in odds file: \[(.*?)\]', result.stdout)
        rows_match = re.search(r'Rows in merged dataframe: (\d+)', result.stdout)
        warning_match = re.search(r'WARNING: (\d+) rows have no matching', result.stdout)
        
        results['enrich'] = {
            'seasons': seasons_match.group(1) if seasons_match else 'N/A',
            'merged_rows': rows_match.group(1) if rows_match else 'N/A',
            'unmatched_rows': warning_match.group(1) if warning_match else '0'
        }
        
        # Step 2: Build dataset
        result = subprocess.run('python src/build_dataset.py', 
                              shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Failed at build_dataset.py', 'details': result.stderr}), 500
        
        dropped_match = re.search(r'Dropped (\d+) rows', result.stdout)
        final_rows_match = re.search(r'Rows: (\d+)', result.stdout)
        
        results['dataset'] = {
            'dropped_rows': dropped_match.group(1) if dropped_match else 'N/A',
            'final_rows': final_rows_match.group(1) if final_rows_match else 'N/A'
        }
        
        # Step 3: Build features
        result = subprocess.run('python src/build_features.py', 
                              shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Failed at build_features.py', 'details': result.stderr}), 500
        
        features_match = re.search(r'Rows: (\d+) Columns: (\d+)', result.stdout)
        
        results['features'] = {
            'rows': features_match.group(1) if features_match else 'N/A',
            'columns': features_match.group(2) if features_match else 'N/A'
        }
        
        # Step 4: Split balanced
        result = subprocess.run('python src/split_balanced.py', 
                              shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Failed at split_balanced.py', 'details': result.stderr}), 500
        
        train_split_match = re.search(r'train\s+(\d+)', result.stdout)
        val_split_match = re.search(r'val\s+(\d+)', result.stdout)
        test_split_match = re.search(r'test\s+(\d+)', result.stdout)
        
        results['splits_balanced'] = {
            'train': train_split_match.group(1) if train_split_match else 'N/A',
            'val': val_split_match.group(1) if val_split_match else 'N/A',
            'test': test_split_match.group(1) if test_split_match else 'N/A'
        }
        
        # Step 5: Split time-based (if exists)
        time_split_exists = os.path.exists('src/split_time_based.py')
        if time_split_exists:
            result = subprocess.run('python src/split_time_based.py', 
                                  shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                return jsonify({'error': 'Failed at split_time_based.py', 'details': result.stderr}), 500
            
            train_split_match = re.search(r'train\s+(\d+)', result.stdout)
            val_split_match = re.search(r'val\s+(\d+)', result.stdout)
            test_split_match = re.search(r'test\s+(\d+)', result.stdout)
            
            results['splits_time'] = {
                'train': train_split_match.group(1) if train_split_match else 'N/A',
                'val': val_split_match.group(1) if val_split_match else 'N/A',
                'test': test_split_match.group(1) if test_split_match else 'N/A'
            }
        
        # Step 6: Train baseline
        baseline_result = subprocess.run('python src/train_winpct_baseline.py', 
                                        shell=True, capture_output=True, text=True)
        if baseline_result.returncode != 0:
            return jsonify({'error': 'Failed at train_winpct_baseline.py', 'details': baseline_result.stderr}), 500
        
        # Parse balanced and time-based metrics
        output_parts = baseline_result.stdout.split('Running win% baseline on TIME-BASED splits')
        baseline_balanced = parse_metrics(output_parts[0])
        baseline_time = parse_metrics(output_parts[1]) if len(output_parts) > 1 else {}
        
        results['baseline'] = {
            'balanced': baseline_balanced,
            'time': baseline_time
        }
        
        # Step 7: Train full features
        full_result = subprocess.run('python src/train_full_features.py', 
                                    shell=True, capture_output=True, text=True)
        if full_result.returncode != 0:
            return jsonify({'error': 'Failed at train_full_features.py', 'details': full_result.stderr}), 500
        
        # Parse balanced and time-based metrics
        output_parts = full_result.stdout.split('Running full-feature logistic regression on TIME-BASED splits')
        full_balanced = parse_metrics(output_parts[0])
        full_time = parse_metrics(output_parts[1]) if len(output_parts) > 1 else {}
        
        results['full'] = {
            'balanced': full_balanced,
            'time': full_time
        }
        
        # Step 8: Train LGBM (if exists)
        lgbm_exists = os.path.exists('src/train_lgbm_full_features.py')
        if lgbm_exists:
            lgbm_result = subprocess.run('python src/train_lgbm_full_features.py', 
                                        shell=True, capture_output=True, text=True)
            if lgbm_result.returncode == 0:
                # Parse LGBM metrics
                output_parts = lgbm_result.stdout.split('TIME-BASED')
                lgbm_balanced = parse_metrics(output_parts[0])
                lgbm_time = parse_metrics(output_parts[1]) if len(output_parts) > 1 else {}
                
                results['lgbm'] = {
                    'balanced': lgbm_balanced,
                    'time': lgbm_time
                }
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)