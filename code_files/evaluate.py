import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import os

def create_summary_table(results_dict, configs):
    """Create summary table of all experiments"""
    summary_data = []
    
    for config_name, results in results_dict.items():
        config = configs[config_name]
        summary_data.append({
            'Model': config['architecture'].upper(),
            'Activation': config['activation'],
            'Optimizer': config['optimizer'],
            'Seq Length': config['seq_length'],
            'Grad Clipping': 'Yes' if config.get('grad_clip') else 'No',
            'Accuracy': f"{results['best_test_acc']:.4f}",
            'F1': f"{results['best_test_f1']:.4f}",
            'Epoch Time (s)': f"{results['avg_epoch_time']:.2f}"
        })
    
    return pd.DataFrame(summary_data)

def plot_training_curves(results_dict, configs, save_path=None):
    """Plot training curves for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_dict)))
    
    for (config_name, results), color in zip(results_dict.items(), colors):
        # Training loss
        axes[0, 0].plot(results['train_losses'], label=config_name, color=color, linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training accuracy
        axes[0, 1].plot(results['train_accs'], label=config_name, color=color, linewidth=2)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test accuracy
        axes[1, 0].plot(results['test_accs'], label=config_name, color=color, linewidth=2)
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test F1-score
        axes[1, 1].plot(results['test_f1s'], label=config_name, color=color, linewidth=2)
        axes[1, 1].set_title('Test F1-Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_sequence_length_comparison(results_dict, configs, save_path=None):
    """Plot performance vs sequence length"""
    seq_lengths = [25, 50, 100]
    architectures = ['rnn', 'lstm', 'bilstm']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for arch in architectures:
        accuracies = []
        f1_scores = []
        
        for seq_len in seq_lengths:
            # Find matching configuration
            matching_configs = [
                (config_name, results) 
                for config_name, results in results_dict.items()
                if configs[config_name]['architecture'] == arch 
                and configs[config_name]['seq_length'] == seq_len
            ]
            
            if matching_configs:
                # Take the first matching configuration
                config_name, results = matching_configs[0]
                accuracies.append(results['best_test_acc'])
                f1_scores.append(results['best_test_f1'])
            else:
                accuracies.append(0)
                f1_scores.append(0)
        
        axes[0].plot(seq_lengths, accuracies, 'o-', label=arch.upper(), linewidth=2, markersize=8)
        axes[1].plot(seq_lengths, f1_scores, 'o-', label=arch.upper(), linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(seq_lengths)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score vs Sequence Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(seq_lengths)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimizer_comparison(results_dict, configs, save_path=None):
    """Plot performance comparison across optimizers"""
    optimizers = ['adam', 'sgd', 'rmsprop']
    architectures = ['rnn', 'lstm', 'bilstm']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    bar_width = 0.25
    x_pos = np.arange(len(architectures))
    
    for i, optimizer in enumerate(optimizers):
        accuracies = []
        f1_scores = []
        
        for arch in architectures:
            # Find matching configuration (using seq_len=50 as reference)
            matching_configs = [
                (config_name, results) 
                for config_name, results in results_dict.items()
                if configs[config_name]['architecture'] == arch 
                and configs[config_name]['optimizer'] == optimizer
                and configs[config_name]['seq_length'] == 50  # Use consistent seq_len
            ]
            
            if matching_configs:
                config_name, results = matching_configs[0]
                accuracies.append(results['best_test_acc'])
                f1_scores.append(results['best_test_f1'])
            else:
                accuracies.append(0)
                f1_scores.append(0)
        
        axes[0].bar(x_pos + i * bar_width, accuracies, bar_width, label=optimizer.upper())
        axes[1].bar(x_pos + i * bar_width, f1_scores, bar_width, label=optimizer.upper())
    
    axes[0].set_xlabel('Architecture')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Optimizer and Architecture')
    axes[0].set_xticks(x_pos + bar_width)
    axes[0].set_xticklabels([arch.upper() for arch in architectures])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Architecture')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score by Optimizer and Architecture')
    axes[1].set_xticks(x_pos + bar_width)
    axes[1].set_xticklabels([arch.upper() for arch in architectures])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_activation_comparison(results_dict, configs, save_path=None):
    """Plot performance comparison across activation functions"""
    activations = ['relu', 'sigmoid', 'tanh']
    architectures = ['rnn', 'lstm', 'bilstm']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    bar_width = 0.25
    x_pos = np.arange(len(architectures))
    
    for i, activation in enumerate(activations):
        accuracies = []
        f1_scores = []
        
        for arch in architectures:
            # Find matching configuration (using seq_len=50, adam as reference)
            matching_configs = [
                (config_name, results) 
                for config_name, results in results_dict.items()
                if configs[config_name]['architecture'] == arch 
                and configs[config_name]['activation'] == activation
                and configs[config_name]['seq_length'] == 50
                and configs[config_name]['optimizer'] == 'adam'
            ]
            
            if matching_configs:
                config_name, results = matching_configs[0]
                accuracies.append(results['best_test_acc'])
                f1_scores.append(results['best_test_f1'])
            else:
                accuracies.append(0)
                f1_scores.append(0)
        
        axes[0].bar(x_pos + i * bar_width, accuracies, bar_width, label=activation.upper())
        axes[1].bar(x_pos + i * bar_width, f1_scores, bar_width, label=activation.upper())
    
    axes[0].set_xlabel('Architecture')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Activation Function and Architecture')
    axes[0].set_xticks(x_pos + bar_width)
    axes[0].set_xticklabels([arch.upper() for arch in architectures])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Architecture')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('F1-Score by Activation Function and Architecture')
    axes[1].set_xticks(x_pos + bar_width)
    axes[1].set_xticklabels([arch.upper() for arch in architectures])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(results_dict, configs, save_dir='../results'):
    """Generate a comprehensive report with all plots and analysis"""
    
    # Create summary table
    summary_df = create_summary_table(results_dict, configs)
    
    # Find best configuration
    best_config_name = max(results_dict.keys(), key=lambda x: results_dict[x]['best_test_acc'])
    best_config = configs[best_config_name]
    best_results = results_dict[best_config_name]
    
    print("=" * 60)
    print("COMPREHENSIVE EXPERIMENT REPORT")
    print("=" * 60)
    print(f"\nBest Configuration: {best_config_name}")
    print(f"Best Accuracy: {best_results['best_test_acc']:.4f}")
    print(f"Best F1-Score: {best_results['best_test_f1']:.4f}")
    print(f"Average Epoch Time: {best_results['avg_epoch_time']:.2f}s")
    
    print(f"\nBest Configuration Details:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Generate all plots
    plot_training_curves(results_dict, configs, os.path.join(save_dir, 'plots/training_curves.png'))
    plot_sequence_length_comparison(results_dict, configs, os.path.join(save_dir, 'plots/sequence_length_comparison.png'))
    plot_optimizer_comparison(results_dict, configs, os.path.join(save_dir, 'plots/optimizer_comparison.png'))
    plot_activation_comparison(results_dict, configs, os.path.join(save_dir, 'plots/activation_comparison.png'))
    
    # Save detailed analysis
    analysis = {
        'best_configuration': best_config_name,
        'best_accuracy': float(best_results['best_test_acc']),
        'best_f1_score': float(best_results['best_test_f1']),
        'configurations_tested': len(results_dict),
        'summary_statistics': {
            'mean_accuracy': summary_df['Accuracy'].astype(float).mean(),
            'std_accuracy': summary_df['Accuracy'].astype(float).std(),
            'mean_f1': summary_df['F1'].astype(float).mean(),
            'mean_epoch_time': summary_df['Epoch Time (s)'].astype(float).mean()
        }
    }
    
    with open(os.path.join(save_dir, 'analysis_report.json'), 'w') as f:
        import json
        json.dump(analysis, f, indent=2)
    
    print(f"\nReport generated in: {save_dir}")
    return summary_df, analysis