import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        
        mean_x = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        mean_y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
        
        var_x = F.avg_pool2d(x ** 2, kernel_size=5, stride=1, padding=2) - mean_x ** 2
        var_y = F.avg_pool2d(y ** 2, kernel_size=5, stride=1, padding=2) - mean_y ** 2
        cov_xy = F.avg_pool2d(x * y, kernel_size=5, stride=1, padding=2) - mean_x * mean_y
        
        ssim_map = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / \
                   ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2))
        
        return 1.0 - ssim_map.mean()

class EdgePreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_edges_x = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        x_edges_y = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        x_edges = torch.sqrt(x_edges_x ** 2 + x_edges_y ** 2 + 1e-8)
        
        y_edges_x = F.conv2d(y.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        y_edges_y = F.conv2d(y.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        y_edges = torch.sqrt(y_edges_x ** 2 + y_edges_y ** 2 + 1e-8)
        
        return F.l1_loss(x_edges, y_edges)

class HybridLoss(nn.Module):
    def __init__(self, config, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservationLoss()
    
    def forward(
        self,
        stego: torch.Tensor,
        cover: torch.Tensor,
        secret_reconstructed: torch.Tensor,
        secret: torch.Tensor
    ) -> dict:
        
        loss_cover = self.mse_loss(stego, cover)
        loss_secret = self.mse_loss(secret_reconstructed, secret)
        loss_ssim = self.ssim_loss(stego, cover)
        loss_edge = self.edge_loss(stego, cover)
        
        total_loss = (
            self.config.loss_weights['cover'] * loss_cover +
            self.config.loss_weights['secret'] * loss_secret +
            self.config.loss_weights['ssim'] * loss_ssim +
            self.config.loss_weights['edge'] * loss_edge
        )
        
        return {
            'total': total_loss,
            'cover': loss_cover.item(),
            'secret': loss_secret.item(),
            'ssim': loss_ssim.item(),
            'edge': loss_edge.item()
        }