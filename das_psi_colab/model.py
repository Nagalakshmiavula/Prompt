import torch
import torch.nn as nn
from typing import Tuple

class SecretEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        channels = config.secret_encoder_channels
        
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels[0])
        
        self.block1 = self.make_residual_block(channels[0], channels[0])
        
        self.down1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels[1]),
            nn.SiLU()
        )
        
        self.block2 = self.make_residual_block(channels[1], channels[1])
        
        self.down2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels[2]),
            nn.SiLU()
        )
        
        self.block3 = self.make_residual_block(channels[2], channels[2])
        
        self.down3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, channels[3]),
            nn.SiLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], config.latent_dim)
    
    def make_residual_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        
        res = x
        x = self.block1(x)
        x = x + res
        
        x = self.down1(x)
        res = x
        x = self.block2(x)
        x = x + res
        
        x = self.down2(x)
        res = x
        x = self.block3(x)
        x = x + res
        
        x = self.down3(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

class SecretDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        channels = config.decoder_channels
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 8)
        
        self.block1 = self.make_residual_block(8, 8)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU()
        )
        
        self.block2 = self.make_residual_block(16, 16)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        
        self.block3 = self.make_residual_block(32, 32)
        
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_bottleneck = nn.Linear(64, config.latent_dim)
        
        self.fc_expand = nn.Linear(config.latent_dim, 256)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 8),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 8),
            nn.SiLU()
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 8),
            nn.SiLU()
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 8),
            nn.SiLU()
        )
        
        self.conv_out = nn.Conv2d(8, 3, kernel_size=3, padding=1)
    
    def make_residual_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        
        res = x
        x = self.block1(x)
        x = x + res
        
        x = self.down1(x)
        res = x
        x = self.block2(x)
        x = x + res
        
        x = self.down2(x)
        res = x
        x = self.block3(x)
        x = x + res
        
        x = self.down3(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_bottleneck(x)
        
        x = self.fc_expand(x)
        x = x.view(-1, 16, 4, 4)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        x = self.conv_out(x)
        
        return x

class DiffusionSteganographyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.secret_encoder = SecretEncoder(config)
        self.secret_decoder = SecretDecoder(config)
        
        from diffusion import DiffusionEmbeddingModule, AdaptiveInjectionSchedule
        self.diffusion_module = DiffusionEmbeddingModule(config)
        self.injection_schedule = AdaptiveInjectionSchedule(config.diffusion_steps)
    
    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        z_secret = self.secret_encoder(secret)
        
        noise = torch.randn_like(cover)
        t = torch.randint(0, self.config.diffusion_steps, (cover.shape[0],), device=cover.device)
        
        x_t = self.diffusion_module.scheduler.add_noise(cover, t, noise)
        
        x_noisy = x_t.clone()
        for step in range(self.config.diffusion_steps - 1, -1, -1):
            lambda_t = self.injection_schedule.get_lambda(step)
            x_noisy = self.diffusion_module.reverse_diffusion_step(x_noisy, step, z_secret, lambda_t)
        
        stego = x_noisy
        secret_reconstructed = self.secret_decoder(stego)
        
        return stego, secret_reconstructed

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)