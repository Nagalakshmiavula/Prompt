import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

def save_image_grid(cover, secret, stego, secret_recon, output_path: str):
    num_images = min(cover.shape[0], 4)
    
    fig, axes = plt.subplots(4, num_images, figsize=(12, 12))
    
    if num_images == 1:
        axes = axes.reshape(4, 1)
    
    for i in range(num_images):
        cover_np = cover[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        secret_np = secret[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        stego_np = stego[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        secret_recon_np = secret_recon[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        
        axes[0, i].imshow(cover_np)
        axes[0, i].set_title("Cover", fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(secret_np)
        axes[1, i].set_title("Secret", fontsize=10)
        axes[1, i].axis('off')
        
        axes[2, i].imshow(stego_np)
        axes[2, i].set_title("Stego", fontsize=10)
        axes[2, i].axis('off')
        
        axes[3, i].imshow(secret_recon_np)
        axes[3, i].set_title("Recovered", fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_metrics(history: Dict, output_path: str):
    if len(history['epoch']) < 2:
        return
    
    epochs = history['epoch']
    
    train_psnr_stego = [m.get('psnr_stego_mean', 0) for m in history['train_metrics']]
    val_psnr_stego = [m.get('psnr_stego_mean', 0) for m in history['val_metrics']]
    
    train_psnr_secret = [m.get('psnr_secret_mean', 0) for m in history['train_metrics']]
    val_psnr_secret = [m.get('psnr_secret_mean', 0) for m in history['val_metrics']]
    
    train_ssim_stego = [m.get('ssim_stego_mean', 0) for m in history['train_metrics']]
    val_ssim_stego = [m.get('ssim_stego_mean', 0) for m in history['val_metrics']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs, train_psnr_stego, label='Train', marker='o', markersize=4)
    axes[0, 0].plot(epochs, val_psnr_stego, label='Val', marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0, 0].set_title('Stego PSNR', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_psnr_secret, label='Train', marker='o', markersize=4)
    axes[0, 1].plot(epochs, val_psnr_secret, label='Val', marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0, 1].set_title('Secret Recovery PSNR', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, train_ssim_stego, label='Train', marker='o', markersize=4)
    axes[1, 0].plot(epochs, val_ssim_stego, label='Val', marker='s', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('SSIM', fontsize=11)
    axes[1, 0].set_title('Stego SSIM', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    axes[1, 1].axis('off')
    best_epoch = np.argmax(val_psnr_stego) + 1 if val_psnr_stego else 0
    info_text = f"Best Epoch: {best_epoch}\n"
    info_text += f"Best PSNR Stego: {max(val_psnr_stego):.2f} dB\n"
    info_text += f"Best PSNR Secret: {max(val_psnr_secret):.2f} dB\n"
    info_text += f"Best SSIM Stego: {max(val_ssim_stego):.4f}\n"
    info_text += f"Total Epochs: {len(epochs)}"
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()