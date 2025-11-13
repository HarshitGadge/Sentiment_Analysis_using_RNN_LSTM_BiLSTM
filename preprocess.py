import pandas as pd
import numpy as np
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        sequence = []
        for word in text:
            if word in self.word_to_idx:
                sequence.append(self.word_to_idx[word])
            else:
                sequence.append(self.word_to_idx['<UNK>'])  # Unknown word
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'length': torch.tensor(min(len(text), self.max_length), dtype=torch.long)
        }

def simple_tokenize(text):
    """Simple tokenizer that doesn't require NLTK punkt"""
    # Split on whitespace and filter out empty strings
    return [word for word in text.split() if word.strip()]

def preprocess_text(text):
    """Preprocess text according to project requirements"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def convert_labels_to_numeric(labels):
    """Convert string labels to numeric (0 and 1)"""
    numeric_labels = []
    for label in labels:
        if isinstance(label, str):
            # Handle string labels
            label_lower = label.lower().strip()
            if label_lower in ['positive', 'pos', '1', 'true', 'yes']:
                numeric_labels.append(1)
            elif label_lower in ['negative', 'neg', '0', 'false', 'no']:
                numeric_labels.append(0)
            else:
                # Try to convert to float, default to 0 if fails
                try:
                    numeric_labels.append(float(label))
                except:
                    numeric_labels.append(0)
        else:
            # Already numeric
            numeric_labels.append(float(label))
    return numeric_labels

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from texts"""
    # Count word frequencies
    word_freq = Counter()
    for text in texts:
        word_freq.update(text)
    
    # Get most common words
    most_common = word_freq.most_common(max_vocab_size - 2)  # -2 for <PAD> and <UNK>
    
    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in most_common:
        word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx

def load_csv_data(csv_path='../data/data.csv', sequence_lengths=[25, 50, 100], 
                 max_vocab_size=10000, batch_size=32, test_size=0.5):
    """Load custom CSV dataset for sentiment classification"""
    
    # Load CSV file
    print(f"Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please make sure your data.csv file exists in the data/ folder.")
    
    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert labels to numeric
    print("Converting labels to numeric...")
    df['label_numeric'] = convert_labels_to_numeric(df['label'].tolist())
    
    # Check label distribution
    print(f"Original label distribution:\n{df['label'].value_counts()}")
    print(f"Numeric label distribution:\n{df['label_numeric'].value_counts()}")
    
    # Preprocess texts
    print("Preprocessing texts...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['tokens'] = df['processed_text'].apply(simple_tokenize)
    
    # Split data (50/50 as specified)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['label_numeric']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx = build_vocab(train_df['tokens'].tolist(), max_vocab_size)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Create datasets for different sequence lengths
    dataloaders = {}
    
    for seq_len in sequence_lengths:
        print(f"Creating datasets for sequence length: {seq_len}")
        
        # Create datasets
        train_dataset = TextDataset(
            train_df['tokens'].tolist(), 
            train_df['label_numeric'].tolist(), 
            word_to_idx, 
            seq_len
        )
        
        test_dataset = TextDataset(
            test_df['tokens'].tolist(), 
            test_df['label_numeric'].tolist(), 
            word_to_idx, 
            seq_len
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        dataloaders[seq_len] = (train_loader, test_loader)
    
    # Calculate dataset statistics
    review_lengths = [len(tokens) for tokens in train_df['tokens']]
    stats = {
        'vocab_size': len(word_to_idx),
        'total_reviews': len(df),
        'avg_review_length': np.mean(review_lengths),
        'max_review_length': max(review_lengths),
        'min_review_length': min(review_lengths),
        'training_samples': len(train_df),
        'testing_samples': len(test_df),
        'label_distribution_train': train_df['label_numeric'].value_counts().to_dict(),
        'label_distribution_test': test_df['label_numeric'].value_counts().to_dict()
    }
    
    print(f"Dataset statistics:")
    for key, value in stats.items():
        if 'distribution' not in key:
            print(f"  {key}: {value}")
    
    return word_to_idx, dataloaders, stats

def analyze_dataset(csv_path='../data/data.csv'):
    """Analyze the dataset before processing"""
    df = pd.read_csv(csv_path)
    
    print("Dataset Analysis:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Check label types
    print(f"Label types: {df['label'].apply(type).value_counts()}")
    
    # Text length analysis
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    print(f"Average text length: {df['text_length'].mean():.2f} words")
    print(f"Max text length: {df['text_length'].max()} words")
    print(f"Min text length: {df['text_length'].min()} words")
    
    return df