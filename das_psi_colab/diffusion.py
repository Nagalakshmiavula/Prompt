import torch
import torch.nn as nn
import math
from typing import Tuple

class NoiseScheduler:
    def __init__(
        self,
        num_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            s = 0.008
            steps = torch.arange(num_steps + 1)
            alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        device = x0.device
        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

class LightweightUNet(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 16, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        
        ch = base_channels
        
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, ch * 4),
            nn.SiLU()
        )
        
        self.secret_proj = nn.Sequential(
            nn.Linear(latent_dim, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch * 4)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch * 4),
            nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch * 4),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2 * 2, ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU()
        )
        
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x_t: torch.Tensor, x_0: torch.Tensor, z_secret: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, x_0], dim=1)
        
        x = self.conv_in(x)
        h1 = x
        
        x = self.down1(x)
        h2 = x
        
        x = self.down2(x)
        
        s_emb = self.secret_proj(z_secret)
        s_emb = s_emb.view(x.shape[0], x.shape[1], 1, 1)
        x = x * (1.0 + s_emb * 0.1)
        
        x = self.bottleneck(x)
        
        x = self.up2(x)
        x = torch.cat([x, h2], dim=1)
        
        x = self.up1(x)
        x = torch.cat([x, h1], dim=1)
        x = nn.functional.conv2d(x, torch.ones(self.base_channels, x.shape[1], 1, 1).to(x.device) / x.shape[1], padding=0)
        
        x = self.conv_out(x)
        
        return x

class DiffusionEmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scheduler = NoiseScheduler(
            num_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule_type=config.noise_schedule
        )
        
        self.unet = LightweightUNet(
            in_channels=6,
            out_channels=3,
            base_channels=config.unet_base_channels,
            latent_dim=config.latent_dim
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        z_secret: torch.Tensor
    ) -> torch.Tensor:
        
        predicted_noise = self.unet(x_t, x_0, z_secret)
        return predicted_noise
    
    def reverse_diffusion_step(
        self,
        x_t: torch.Tensor,
        t: int,
        z_secret: torch.Tensor,
        lambda_t: float = 0.1
    ) -> torch.Tensor:
        
        predicted_noise = self.unet(x_t, x_t, z_secret)
        
        device = x_t.device
        self.scheduler.to(device)
        
        alpha_t = self.scheduler.alphas_cumprod[t]
        alpha_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)
        
        x_prev = (sqrt_alpha_prev / sqrt_alpha_t) * (x_t - sqrt_one_minus_alpha_t * predicted_noise)
        x_prev = x_prev + sqrt_one_minus_alpha_prev * predicted_noise
        
        return x_prev

class AdaptiveInjectionSchedule:
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps
    
    def get_lambda(self, t: int) -> float:
        if t < self.num_steps // 3:
            return 0.15 + 0.1 * (1 - (t / (self.num_steps // 3 + 1e-6)))
        elif t < 2 * self.num_steps // 3:
            return 0.075 + 0.025 * (1 - ((t - self.num_steps // 3) / (self.num_steps // 3 + 1e-6)))
        else:
            return 0.03 + 0.02 * (1 - ((t - 2 * self.num_steps // 3) / (self.num_steps // 3 + 1e-6)))