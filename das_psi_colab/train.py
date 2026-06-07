import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
import random
import time

from config import get_config
from dataset import create_dataloaders
from model import DiffusionSteganographyModel, count_parameters
from losses import HybridLoss
from metrics import MetricsCalculator
from utils import (
    set_seed, save_checkpoint, load_checkpoint, log_metrics,
    plot_loss_curves, denormalize_image
)
from visualization import save_image_grid, plot_metrics

class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model = DiffusionSteganographyModel(config).to(self.device)
        
        total_params = count_parameters(self.model)
        
        print(f"\n{'='*60}")
        print(f"Model Parameters: {total_params:,}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {config.mixed_precision}")
        print(f"{'='*60}\n")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        self.loss_fn = HybridLoss(config, str(self.device))
        
        self.scaler = GradScaler() if config.mixed_precision else None
        
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        all_metrics = []
        
        for batch_idx, (cover, secret) in enumerate(train_loader):
            cover = cover.to(self.device)
            secret = secret.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with autocast():
                    stego, secret_reconstructed = self.model(cover, secret, training=True)
                    loss_dict = self.loss_fn(stego, cover, secret_reconstructed, secret)
                    total_loss = loss_dict['total']
                
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                stego, secret_reconstructed = self.model(cover, secret, training=True)
                loss_dict = self.loss_fn(stego, cover, secret_reconstructed, secret)
                total_loss = loss_dict['total']
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            epoch_loss += total_loss.item()
            
            batch_metrics = MetricsCalculator.compute_metrics_batch(
                stego, cover, secret_reconstructed, secret, self.config
            )
            all_metrics.append(batch_metrics)
        
        avg_loss = epoch_loss / len(train_loader)
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        
        return {'loss': avg_loss, **aggregated_metrics}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        epoch_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (cover, secret) in enumerate(val_loader):
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                stego, secret_reconstructed = self.model(cover, secret, training=False)
                loss_dict = self.loss_fn(stego, cover, secret_reconstructed, secret)
                total_loss = loss_dict['total']
                
                epoch_loss += total_loss.item()
                
                batch_metrics = MetricsCalculator.compute_metrics_batch(
                    stego, cover, secret_reconstructed, secret, self.config
                )
                all_metrics.append(batch_metrics)
        
        avg_loss = epoch_loss / len(val_loader)
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        
        return {'loss': avg_loss, **aggregated_metrics}
    
    def aggregate_metrics(self, all_metrics: list) -> Dict[str, float]:
        aggregated = {}
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
            if values:
                aggregated[key] = float(np.mean(values))
        
        return aggregated
    
    def save_sample_outputs(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        
        try:
            cover, secret = next(iter(val_loader))
            cover = cover[:2].to(self.device)
            secret = secret[:2].to(self.device)
            
            with torch.no_grad():
                stego, secret_recon = self.model(cover, secret, training=False)
            
            cover_denorm = denormalize_image(cover)
            secret_denorm = denormalize_image(secret)
            stego_denorm = denormalize_image(stego)
            secret_recon_denorm = denormalize_image(secret_recon)
            
            save_image_grid(
                cover_denorm, secret_denorm, stego_denorm, secret_recon_denorm,
                os.path.join(self.config.output_path, f"epoch_{epoch:04d}.png")
            )
        except Exception as e:
            print(f"Warning: Could not save sample outputs: {e}")
    
    def detect_convergence(self) -> Dict:
        if len(self.train_history['val_loss']) < 2:
            return {'status': 'early', 'best_epoch': -1, 'reason': 'Not enough epochs'}
        
        val_losses = self.train_history['val_loss']
        train_losses = self.train_history['train_loss']
        
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_epoch]
        
        if len(val_losses) > 1:
            overfitting_ratio = val_losses[-1] / (train_losses[-1] + 1e-8)
        else:
            overfitting_ratio = 1.0
        
        if len(val_losses) > 50:
            recent_improvement = min(val_losses[-50:]) - max(val_losses[-50:])
            if abs(recent_improvement) < 0.0005:
                status = 'converged'
                reason = 'Validation loss plateau detected (50 epoch window)'
            else:
                status = 'improving'
                reason = 'Still improving'
        else:
            status = 'training'
            reason = 'Early training phase'
        
        return {
            'status': status,
            'reason': reason,
            'best_epoch': best_val_epoch,
            'best_val_loss': float(best_val_loss),
            'overfitting_ratio': float(overfitting_ratio),
            'current_epoch': len(val_losses)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        start_time = time.time()
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        
        print(f"\nStarting training on device: {self.device}")
        print(f"Expected convergence: 100-300 epochs (max {self.config.num_epochs})\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}]", end=" ")
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            print(f"| Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f} | Time: {epoch_time:.1f}s")
            
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_sample_outputs(val_loader, epoch + 1)
                
                convergence_info = self.detect_convergence()
                print(f"  → Status: {convergence_info['status']} ({convergence_info['reason']})")
                print(f"  → Best epoch: {convergence_info['best_epoch'] + 1}, Overfitting ratio: {convergence_info['overfitting_ratio']:.2f}")
                
                checkpoint_path = os.path.join(
                    self.config.checkpoints_path,
                    f"checkpoint_epoch_{epoch + 1:04d}.pt"
                )
                save_checkpoint(self.model, self.optimizer, epoch, checkpoint_path)
                
                log_metrics(self.train_history, os.path.join(self.config.output_path, 'metrics.json'))
                
                plot_loss_curves(self.train_history, os.path.join(self.config.output_path, 'loss_curves.png'))
                plot_metrics(self.train_history, os.path.join(self.config.output_path, 'metrics_plot.png'))
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                epochs_no_improve = 0
                
                best_checkpoint_path = os.path.join(self.config.checkpoints_path, 'best_model.pt')
                save_checkpoint(self.model, self.optimizer, epoch, best_checkpoint_path)
            else:
                epochs_no_improve += 1
            
            convergence_info = self.detect_convergence()
            if convergence_info['status'] == 'converged' and epochs_no_improve >= 50:
                print(f"\n✓ Convergence detected at epoch {epoch + 1}")
                break
            
            if epochs_no_improve >= self.config.early_stopping_patience:
                print(f"\n✓ Early stopping at epoch {epoch + 1} (no improvement for {epochs_no_improve} epochs)")
                break
        
        total_time = time.time() - start_time
        
        final_checkpoint_path = os.path.join(self.config.checkpoints_path, 'final_model.pt')
        save_checkpoint(self.model, self.optimizer, len(self.train_history['epoch']) - 1, final_checkpoint_path)
        
        convergence_report = {
            'best_epoch': best_epoch + 1,
            'best_val_loss': float(best_val_loss),
            'total_epochs': len(self.train_history['epoch']),
            'total_training_time_seconds': total_time,
            'total_training_time_hours': total_time / 3600,
            'early_stopping_reason': 'Convergence detected' if convergence_info['status'] == 'converged' else 'No improvement',
            'convergence_analysis': self.detect_convergence()
        }
        
        with open(os.path.join(self.config.output_path, 'convergence_report.json'), 'w') as f:
            json.dump(convergence_report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"Best Epoch: {best_epoch + 1}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Total Time: {total_time/3600:.2f} hours ({int(total_time/60)} minutes)")
        print(f"{'='*60}\n")

def main():
    config = get_config()
    set_seed(config.seed)
    
    print(f"\n{'='*60}")
    print("DAS-PSI: Colab-Optimized Training")
    print(f"{'='*60}")
    
    train_loader, val_loader, _ = create_dataloaders(config)
    
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()