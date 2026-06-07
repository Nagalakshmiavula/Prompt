import torch
import numpy as np
from typing import Dict

class MetricsCalculator:
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1_np = img1.detach().cpu().clamp(0, 1).numpy()
        img2_np = img2.detach().cpu().clamp(0, 1).numpy()
        
        mse = np.mean((img1_np - img2_np) ** 2)
        
        if mse < 1e-10:
            return 100.0
        
        psnr_value = 20 * np.log10(1.0 / np.sqrt(mse))
        
        return float(np.clip(psnr_value, 0, 100))
    
    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1_np = img1.detach().cpu().clamp(0, 1).numpy()
        img2_np = img2.detach().cpu().clamp(0, 1).numpy()
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        
        mean1 = np.mean(img1_np)
        mean2 = np.mean(img2_np)
        var1 = np.var(img1_np)
        var2 = np.var(img2_np)
        cov = np.mean((img1_np - mean1) * (img2_np - mean2))
        
        ssim_value = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
                     ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        return float(np.clip(ssim_value, -1, 1))
    
    @staticmethod
    def mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1_np = img1.detach().cpu().clamp(0, 1).numpy()
        img2_np = img2.detach().cpu().clamp(0, 1).numpy()
        
        mse_value = np.mean((img1_np - img2_np) ** 2)
        
        return float(mse_value)
    
    @staticmethod
    def bpp(latent_dim: int = 256, image_size: int = 256) -> float:
        total_pixels = image_size * image_size * 3
        bpp_value = latent_dim / total_pixels
        return float(bpp_value)
    
    @staticmethod
    def compute_metrics_batch(
        stego_batch: torch.Tensor,
        cover_batch: torch.Tensor,
        secret_recon_batch: torch.Tensor,
        secret_batch: torch.Tensor,
        config
    ) -> Dict[str, float]:
        
        metrics = {
            'psnr_stego': [],
            'ssim_stego': [],
            'mse_stego': [],
            'psnr_secret': [],
            'ssim_secret': [],
            'mse_secret': [],
        }
        
        batch_size = stego_batch.shape[0]
        
        for i in range(batch_size):
            stego = stego_batch[i]
            cover = cover_batch[i]
            secret_recon = secret_recon_batch[i]
            secret = secret_batch[i]
            
            metrics['psnr_stego'].append(MetricsCalculator.psnr(stego, cover))
            metrics['ssim_stego'].append(MetricsCalculator.ssim(stego, cover))
            metrics['mse_stego'].append(MetricsCalculator.mse(stego, cover))
            
            metrics['psnr_secret'].append(MetricsCalculator.psnr(secret_recon, secret))
            metrics['ssim_secret'].append(MetricsCalculator.ssim(secret_recon, secret))
            metrics['mse_secret'].append(MetricsCalculator.mse(secret_recon, secret))
        
        return {
            'psnr_stego_mean': float(np.mean(metrics['psnr_stego'])),
            'psnr_stego_std': float(np.std(metrics['psnr_stego'])),
            'ssim_stego_mean': float(np.mean(metrics['ssim_stego'])),
            'ssim_stego_std': float(np.std(metrics['ssim_stego'])),
            'mse_stego_mean': float(np.mean(metrics['mse_stego'])),
            'mse_stego_std': float(np.std(metrics['mse_stego'])),
            'psnr_secret_mean': float(np.mean(metrics['psnr_secret'])),
            'psnr_secret_std': float(np.std(metrics['psnr_secret'])),
            'ssim_secret_mean': float(np.mean(metrics['ssim_secret'])),
            'ssim_secret_std': float(np.std(metrics['ssim_secret'])),
            'mse_secret_mean': float(np.mean(metrics['mse_secret'])),
            'mse_secret_std': float(np.std(metrics['mse_secret'])),
            'bpp': MetricsCalculator.bpp(config.latent_dim, config.image_size)
        }