import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


from models import RNNModel, BidirectionalLSTM

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, 
                 criterion, device, grad_clip=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['label']
            
            self.optimizer.zero_grad()
            
            predictions = self.model(batch)
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Convert predictions to binary
            binary_predictions = predictions.round().detach().cpu()
            all_predictions.extend(binary_predictions.tolist())
            all_labels.extend(labels.cpu().tolist())
        
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss / len(self.train_loader), epoch_acc, epoch_f1
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.valid_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch['label']
                
                predictions = self.model(batch)
                loss = self.criterion(predictions, labels)
                
                epoch_loss += loss.item()
                
                binary_predictions = predictions.round().cpu()
                all_predictions.extend(binary_predictions.tolist())
                all_labels.extend(labels.cpu().tolist())
        
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return epoch_loss / len(self.valid_loader), epoch_acc, epoch_f1

def train_model(model_config, train_loader, test_loader, num_epochs=10):
    """Train a model with given configuration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    VOCAB_SIZE = model_config['vocab_size']
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128  # Increased for better performance
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.3
    
    # Create model
    if model_config['architecture'] == 'rnn':
        model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, DROPOUT, 'rnn', False, model_config['activation'])
    elif model_config['architecture'] == 'lstm':
        model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, DROPOUT, 'lstm', False, model_config['activation'])
    elif model_config['architecture'] == 'bilstm':
        model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, DROPOUT, 'lstm', True, model_config['activation'])
    
    model = model.to(device)
    
    # Better optimizer settings
    if model_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif model_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif model_config['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    criterion = nn.BCELoss()
    
    # Gradient clipping
    grad_clip = model_config.get('grad_clip', None)
    
    trainer = Trainer(model, train_loader, test_loader, optimizer, 
                     criterion, device, grad_clip)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_accs = []
    test_f1s = []
    epoch_times = []
    
    best_acc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_f1 = trainer.train_epoch()
        test_loss, test_acc, test_f1 = trainer.evaluate()
        
        epoch_time = time.time() - start_time
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        epoch_times.append(epoch_time)
        
        print(f'  Epoch {epoch+1:02}: Train Loss: {train_loss:.3f} | '
              f'Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1*100:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model.state_dict().copy()
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'test_f1s': test_f1s,
        'epoch_times': epoch_times,
        'best_test_acc': best_acc,
        'best_test_f1': max(test_f1s) if test_f1s else 0,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'model': model
    }
    
    return results