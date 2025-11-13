import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, rnn_type='rnn', bidirectional=False, 
                 activation='relu'):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.activation = activation
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Choose RNN type
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                             bidirectional=bidirectional, dropout=dropout,
                             batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout,
                              batch_first=True)
        else:
            raise ValueError("RNN type must be 'rnn' or 'lstm'")
        
        # Calculate factor for bidirectional
        self.fc_factor = 2 if bidirectional else 1
        
        # Hidden layers
        self.fc1 = nn.Linear(hidden_dim * self.fc_factor, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch):
        text, lengths = batch['text'], batch['length']
        
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        
        # Handle final hidden state
        if self.bidirectional:
            # For bidirectional, concatenate the final forward and backward hidden states
            if self.rnn_type == 'lstm':
                hidden_forward = hidden[-2, :, :]  # Last forward layer
                hidden_backward = hidden[-1, :, :]  # Last backward layer
                hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
            else:
                hidden_forward = hidden[-2, :, :]
                hidden_backward = hidden[-1, :, :]
                hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            # For unidirectional, take the last layer hidden state
            hidden = hidden[-1, :, :]
        
        # Apply activation function to hidden state
        if self.activation == 'relu':
            hidden = F.relu(hidden)
        elif self.activation == 'sigmoid':
            hidden = torch.sigmoid(hidden)
        elif self.activation == 'tanh':
            hidden = torch.tanh(hidden)
        
        # Fully connected layers
        x = self.dropout(hidden)
        x = self.fc1(x)
        x = self.apply_activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.apply_activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return torch.sigmoid(x).squeeze(1)
    
    def apply_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)

class BidirectionalLSTM(RNNModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, dropout, activation='relu'):
        super().__init__(vocab_size, embedding_dim, hidden_dim, output_dim,
                        n_layers, dropout, 'lstm', True, activation)