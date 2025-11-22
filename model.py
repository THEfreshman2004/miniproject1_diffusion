import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_dataset_and_make_dataloaders

class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise) # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

#This is just the residual block
class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y

def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80): #it samples sigma for size n=1 (scalar)
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)

def sample_constants(data,sigma_data): #It return c_in,c_out,c_skip and c_noise
    sigma = sample_sigma(data.shape[0]) #sigma has the same size as the number of images in our batch

    c_in = 1/(torch.sqrt((sigma_data**2 + sigma)))
    c_out = (sigma*sigma_data)/(sigma**2+sigma_data**2)
    c_skip = sigma_data**2/(sigma**2 + sigma_data**2)
    c_noise = 0.25*torch.log(sigma)

    return (c_in,c_out,c_skip,c_noise)


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

#Training process (Task 1):
dl , info = load_dataset_and_make_dataloaders(
    dataset_name='FashionMNIST',
    root_dir='data', # choose the directory to store the data 
    batch_size=400,
    num_workers=0   # you can use more workers if you see the GPU is waiting for the batches
)
train_load = dl.train #each batch is of size: (400,1,32,32)
valid_load = dl.valid #each batch is of size: (800,1,32,32)

model = Model(1,nb_channels=64,num_blocks=3,cond_channels=64)
