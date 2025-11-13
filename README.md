# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates various RNN architectures (RNN, LSTM, BiLSTM) for sentiment classification on custom CSV datasets. The system systematically tests different configurations including activation functions, optimizers, sequence lengths, and gradient clipping strategies.

## 📋 Project Requirements

- **Python Version**: 3.8 or higher
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 1GB free space
- **Platform**: Windows, macOS, or Linux

## 🔧 Setup Instructions

### 1. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually if needed:
pip install torch>=2.0.0 numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0
pip install seaborn>=0.11.0 scikit-learn>=0.24.0 tqdm>=4.62.0 nltk>=3.6.0

# Download NLTK data for tokenization
python -c "import nltk; nltk.download('punkt')"
```
