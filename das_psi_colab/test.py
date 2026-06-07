import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from config import get_config
from dataset import create_dataloaders
from model import DiffusionSteganographyModel
from metrics import MetricsCalculator
from utils import load_checkpoint, denormalize_image
from visualization import save_image_grid

class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model = DiffusionSteganographyModel(config).to(self.device)
        
        best_checkpoint_path = os.path.join(config.checkpoints_path, 'best_model.pt')
        
        if os.path.exists(best_checkpoint_path):
            load_checkpoint(self.model, None, best_checkpoint_path)
            print(f"✓ Loaded best model from {best_checkpoint_path}\n")
        else:
            raise FileNotFoundError(f"Best model checkpoint not found at {best_checkpoint_path}")
    
    def test(self, test_loader) -> Dict[str, float]:
        self.model.eval()
        all_metrics = []
        
        print(f"{'='*60}")
        print("Testing on Unseen Test Set")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            for batch_idx, (cover, secret) in enumerate(tqdm(test_loader, desc="Testing")):
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                stego, secret_reconstructed = self.model(cover, secret, training=False)
                
                batch_metrics = MetricsCalculator.compute_metrics_batch(
                    stego, cover, secret_reconstructed, secret, self.config
                )
                all_metrics.append(batch_metrics)
                
                if batch_idx == 0:
                    cover_denorm = denormalize_image(cover)
                    secret_denorm = denormalize_image(secret)
                    stego_denorm = denormalize_image(stego)
                    secret_recon_denorm = denormalize_image(secret_reconstructed)
                    
                    save_image_grid(
                        cover_denorm[:2], secret_denorm[:2], stego_denorm[:2], secret_recon_denorm[:2],
                        os.path.join(self.config.output_path, "test_samples.png")
                    )
        
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        return aggregated_metrics
    
    def aggregate_metrics(self, all_metrics: list) -> Dict[str, float]:
        aggregated = {}
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
        
        return aggregated

def main():
    config = get_config()
    
    _, _, test_loader = create_dataloaders(config)
    
    tester = Tester(config)
    test_metrics = tester.test(test_loader)
    
    print(f"\n{'='*60}")
    print("Test Results (Unseen Test Set)")
    print(f"{'='*60}\n")
    
    print("Stego Image Quality:")
    print(f"  PSNR: {test_metrics.get('psnr_stego_mean_mean', 0):.2f} ± {test_metrics.get('psnr_stego_mean_std', 0):.2f} dB")
    print(f"  SSIM: {test_metrics.get('ssim_stego_mean_mean', 0):.4f} ± {test_metrics.get('ssim_stego_mean_std', 0):.4f}")
    print(f"  MSE: {test_metrics.get('mse_stego_mean_mean', 0):.6f} ± {test_metrics.get('mse_stego_mean_std', 0):.6f}")
    
    print("\nSecret Recovery Quality:")
    print(f"  PSNR: {test_metrics.get('psnr_secret_mean_mean', 0):.2f} ± {test_metrics.get('psnr_secret_mean_std', 0):.2f} dB")
    print(f"  SSIM: {test_metrics.get('ssim_secret_mean_mean', 0):.4f} ± {test_metrics.get('ssim_secret_mean_std', 0):.4f}")
    print(f"  MSE: {test_metrics.get('mse_secret_mean_mean', 0):.6f} ± {test_metrics.get('mse_secret_mean_std', 0):.6f}")
    
    print(f"\nPayload Capacity (BPP): {test_metrics.get('bpp_mean', 0):.6f}")
    
    print(f"{'='*60}\n")
    
    test_report = {
        'test_metrics': test_metrics,
        'total_test_samples': len(test_loader) * config.batch_size
    }
    
    with open(os.path.join(config.output_path, 'test_report.json'), 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"✓ Test report saved to {os.path.join(config.output_path, 'test_report.json')}\n")

if __name__ == "__main__":
    main()