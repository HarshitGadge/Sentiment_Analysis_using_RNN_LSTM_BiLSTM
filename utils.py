import torch
import random
import numpy as np
import psutil
import platform

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_hardware_info():
    """Get hardware information"""
    info = {
        'device': 'GPU' if torch.cuda.is_available() else 'CPU',
        'cpu_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'ram_size_gb': f"{psutil.virtual_memory().total / (1024**3):.1f}",
        'platform': platform.platform(),
        'python_version': platform.python_version()
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}"
    
    return info

def save_model(model, path):
    """Save model to file"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model from file"""
    model.load_state_dict(torch.load(path))
    return model