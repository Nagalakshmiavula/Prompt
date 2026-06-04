import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    device: str = "cuda"
    seed: int = 42
    
    image_size: int = 256
    image_channels: int = 3
    
    batch_size: int = 8
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    diffusion_steps: int = 10
    noise_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    latent_dim: int = 256
    secret_encoder_channels: list = None
    decoder_channels: list = None
    unet_base_channels: int = 16
    
    loss_weights: dict = None
    
    checkpoint_interval: int = 100
    log_interval: int = 100
    
    dataset_split: dict = None
    data_path: str = "data"
    output_path: str = "outputs"
    checkpoints_path: str = "checkpoints"
    
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    num_workers: int = 2
    pin_memory: bool = True
    
    early_stopping_patience: int = 150
    early_stopping_threshold: float = 0.001
    
    subset_size: Optional[dict] = None
    generate_synthetic_dataset: bool = True
    synthetic_dataset_path: str = "data/synthetic_images"
    
    def __post_init__(self):
        if self.secret_encoder_channels is None:
            self.secret_encoder_channels = [16, 32, 64, 128]
        
        if self.decoder_channels is None:
            self.decoder_channels = [64, 32, 16, 8]
        
        if self.loss_weights is None:
            self.loss_weights = {
                'cover': 1.0,
                'secret': 0.8,
                'ssim': 0.3,
                'perceptual': 0.2,
                'consistency': 0.1,
                'edge': 0.05
            }
        
        if self.dataset_split is None:
            self.dataset_split = {
                'train': 0.70,
                'val': 0.15,
                'test': 0.15
            }
        
        if self.subset_size is None:
            self.subset_size = {
                'train': 300,
                'val': 60,
                'test': 60
            }
        
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.synthetic_dataset_path, exist_ok=True)

def get_config() -> Config:
    return Config()
