import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch: int, checkpoint_path: str):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    return epoch

def log_metrics(history: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
    
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm

def plot_loss_curves(history: Dict, output_path: str):
    if len(history['epoch']) < 2:
        return
    
    epochs = history['epoch']
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()