import torch
import pandas as pd
import time
from preprocess import load_csv_data, analyze_dataset
from train import train_model
from evaluate import create_summary_table
from utils import set_seeds, get_hardware_info
import json
import os

class ExperimentController:
    def __init__(self, data_path='../data/data.csv'):
        set_seeds(42)
        self.hardware_info = get_hardware_info()
        self.data_path = data_path
        self.results = {}
        self.configs = {}
        
    def create_experiment_configs(self, vocab_size=10000):
        """Create all experiment configurations"""
        base_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 100,
            'hidden_dim': 64,
            'n_layers': 2,
            'dropout': 0.3,
            'batch_size': 32,
            'epochs': 10  
        }
        
        experiments = {}
        
        # Focus on key comparisons - skip some combinations for speed
        key_configs = [
            # RNN experiments (6 configurations)
            ('rnn', 'relu', 'adam', 25, None),
            ('rnn', 'relu', 'sgd', 50, 1.0),
            ('rnn', 'sigmoid', 'rmsprop', 100, None),
            ('rnn', 'tanh', 'adam', 50, 1.0),
            ('rnn', 'relu', 'rmsprop', 100, 1.0),
            ('rnn', 'tanh', 'sgd', 25, None),
            
            # LSTM experiments (7 configurations)
            ('lstm', 'relu', 'adam', 25, None),
            ('lstm', 'relu', 'adam', 50, 1.0),
            ('lstm', 'sigmoid', 'sgd', 100, None),
            ('lstm', 'tanh', 'rmsprop', 50, 1.0),
            ('lstm', 'relu', 'sgd', 100, None),
            ('lstm', 'sigmoid', 'adam', 25, 1.0),
            ('lstm', 'tanh', 'adam', 100, None),
            
            # BiLSTM experiments (7 configurations)
            ('bilstm', 'relu', 'adam', 25, 1.0),
            ('bilstm', 'relu', 'rmsprop', 50, None),
            ('bilstm', 'sigmoid', 'adam', 100, 1.0),
            ('bilstm', 'tanh', 'sgd', 25, None),
            ('bilstm', 'relu', 'sgd', 50, 1.0),
            ('bilstm', 'sigmoid', 'rmsprop', 100, None),
            ('bilstm', 'tanh', 'adam', 50, None),
        ]
        
        
        for arch, activation, optimizer, seq_len, grad_clip in key_configs:
            config_name = f"{arch}_{activation}_{optimizer}_{seq_len}_clip{grad_clip}"
            experiments[config_name] = {
                'architecture': arch,
                'activation': activation,
                'optimizer': optimizer,
                'seq_length': seq_len,
                'grad_clip': grad_clip,
                **base_config
            }
        
        return experiments
    
    def run_experiments(self):
        """Run experiments and print results table"""
        print("Hardware Information:")
        for key, value in self.hardware_info.items():
            print(f"  {key}: {value}")
        
        # Analyze dataset first
        print("\nAnalyzing dataset...")
        self.analyze_data()
        
        # Load data
        print("\nLoading data...")
        sequence_lengths = [25, 50, 100]
        word_to_idx, dataloaders, stats = load_csv_data(
            csv_path=self.data_path, 
            sequence_lengths=sequence_lengths
        )
        
        # Get experiment configurations
        all_configs = self.create_experiment_configs(vocab_size=len(word_to_idx))
        self.configs = all_configs
        
        print(f"\nRunning {len(all_configs)} experiments...")
        print("This may take a while. Current progress:")
        
        for config_name, config in all_configs.items():
            print(f"\n▶ Running: {config_name}")
            start_time = time.time()
            
            # Get appropriate data loader for sequence length
            seq_len = config['seq_length']
            train_loader, test_loader = dataloaders[seq_len]
            
            # Train model
            try:
                results = train_model(config, train_loader, test_loader, num_epochs=config['epochs'])
                self.results[config_name] = results
                
                experiment_time = time.time() - start_time
                print(f"  ✅ Completed in {experiment_time:.2f}s | "
                      f"Best Acc: {results['best_test_acc']:.4f} | "
                      f"Best F1: {results['best_test_f1']:.4f}")
                      
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        # Generate and display results table
        self.display_results()
    
    def display_results(self):
        """Display results in a formatted table"""
        if not self.results:
            print("No results to display!")
            return
        
        # Create summary table
        summary_df = create_summary_table(self.results, self.configs)
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Find best configuration
        best_config_name = max(self.results.keys(), 
                              key=lambda x: self.results[x]['best_test_acc'])
        best_results = self.results[best_config_name]
        
        print(f"\n🏆 BEST CONFIGURATION: {best_config_name}")
        print(f"   Accuracy: {best_results['best_test_acc']:.4f}")
        print(f"   F1-Score: {best_results['best_test_f1']:.4f}")
        print(f"   Avg Epoch Time: {best_results['avg_epoch_time']:.2f}s")
        
        # Save simple results to CSV (no plots)
        summary_df.to_csv('results_summary.csv', index=False)
        print(f"\n💾 Results saved to: results_summary.csv")
        
        # Performance analysis
        self.analyze_performance()
    
    def analyze_performance(self):
        """Analyze performance across different architectures"""
        print(f"\n📊 PERFORMANCE ANALYSIS BY ARCHITECTURE:")
        
        arch_results = {'rnn': [], 'lstm': [], 'bilstm': []}
        
        for config_name, results in self.results.items():
            arch = self.configs[config_name]['architecture']
            arch_results[arch].append(results['best_test_acc'])
        
        for arch, accuracies in arch_results.items():
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"   {arch.upper():6}: {avg_acc:.4f} (from {len(accuracies)} configs)")
    
    def analyze_data(self):
        """Quick dataset analysis"""
        try:
            df = analyze_dataset(self.data_path)
            return df
        except Exception as e:
            print(f"Data analysis skipped: {e}")
            return None

def main():
    print("🚀 Starting RNN Architecture Comparison")
    print("Note: Plot generation is disabled for faster results")
    print("Only results table will be generated\n")
    
    controller = ExperimentController()
    controller.run_experiments()

if __name__ == "__main__":
    main()