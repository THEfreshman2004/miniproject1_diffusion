import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_dataset_and_make_dataloaders
from PIL import Image
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np

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

def sample_constants(sigma_data): #It return c_in,c_out,c_skip and c_noise
    sigma = sample_sigma(1) #sigma has the same size as the number of images in our batch

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

#Training process(FashionMNIST) (Task 1):
dl , info = load_dataset_and_make_dataloaders(
    dataset_name='FashionMNIST',
    root_dir='data', # choose the directory to store the data 
    batch_size=400,
    num_workers=0   # you can use more workers if you see the GPU is waiting for the batches
)
train_load = dl.train #each batch is of size: (400,1,32,32)
valid_load = dl.valid #each batch is of size: (800,1,32,32)
Sigma = info.sigma_data #the global standard deviation of the data
#For testing if the code works (It does)
model = Model(1,nb_channels=64,num_blocks=3,cond_channels=64)
optimizer = torch.optim.Adam(model.parameters()) #we'll not specify lr and decay for now
criterion = nn.MSELoss()

def train_model(model,optimizer,criterion,nb_epochs): #1 epoch took me 450 seconds (around 8 minutes) !
    model.train(True) #We're in training mode
    for _ in range(nb_epochs):
        for images,_ in train_load:
            sigma = sample_sigma(1) #sigma is a number
            noise = torch.randn_like(images)*sigma*images
            noisy_batch = images + noise
            c_in,c_out,c_skip,cnoise = sample_constants(torch.std(images))
            output = model(c_in*noisy_batch,cnoise)
            loss = criterion(output,(images-c_skip*noisy_batch)/c_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.train(False)

#Task 2 : Build sampling pipeline
def Denoiser(x,model,sigma_data):
    cin,cout,cskip,cnoise = sample_constants(sigma_data)
    input = cin*x
    output = model(input,cnoise)
    return cskip*x + cout*output

#How about we modify the function Euler_sampling so that we get also a list of tensors
#where l[i,j] = the image i at the setp j. the list will be of size : nbr_images*steps
#each tensor of l holds the images throughout the process

def Euler_sampling(noise,sigmas,model,sigma_data):
    x = noise.clone()
    process = [] #the list we talked about
    for i, sigma in enumerate(sigmas):
    
        with torch.no_grad():
            x_denoised = Denoiser(x,model,sigma_data)  
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma
    
        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method
        process.append(x) #we add a snapshot of this step to process
#the output is of the size of the noise. We iteratively apply the denoiser
    return x,process

def Sampling(nbr_images,model,nbr_steps):
    sigmas = build_sigma_schedule(steps=nbr_steps) #i'll just keep the setps 50 by default
    #it returns a tensor with number of steps elements
    noise = torch.randn(size=(nbr_images,1,32,32)) * sigmas[0]
    images,_ = Euler_sampling(noise,sigmas,model,Sigma)
    return images
noise = torch.randn(size=(3,1,32,32))
sigmas = sample_sigma(20)
x , proc = Euler_sampling(noise,sigmas,model,Sigma)
print(x.shape,proc[0].shape)

#We visualize the denoising process using the process output from the Euler_sampling function
import time
def visualize_process(t):
    for x in t:
        x = x.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
        x = make_grid(x)
        x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
        x.show()
        time.sleep(10) #we wait 10 seconds to display each photo
