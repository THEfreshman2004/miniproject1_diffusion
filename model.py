
device = 'cuda' #Choose the device

from data import load_dataset_and_make_dataloaders

dl , info = load_dataset_and_make_dataloaders(
    dataset_name='FashionMNIST',
    root_dir='data', # choose the directory to store the data
    batch_size=64,
    num_workers=0   # you can use more workers if you see the GPU is waiting for the batches
)
train_load = dl.train #each batch is of size: (64,1,32,32)
valid_load = dl.valid #each batch is of size: (B,1,32,32) B:batch size
Sigma = info.sigma_data #the global standard deviation of the data

################# Model and training:
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np

class Model(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        nb_channels: int = 64, #divisible by 8 (for the group normalization)
        num_blocks: int = 4,
        cond_channels: int = 128, #Better to be even,
        nbr_classes : int = 10,
        null_init = False #It declares if we want to choose null initialisation or not
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.batch = conditional_BN(num_channels=nb_channels,cond_channels=cond_channels)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels,cond_channels=cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
      
        #class conditioning:
        self.class_embed = nn.Embedding(nbr_classes,cond_channels)

        if null_init:
            #Null initialisation for conv_out:
            nn.init.zeros_(self.conv_out.weight)
            if self.conv_in.bias is not None:
                nn.init.zeros_(self.conv_out.bias)

            #Null initialisation for resBlocks:
            for block in self.blocks:
                for m in block.modules():
                    if isinstance(m,nn.Conv2d):
                        nn.init.zeros_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor,clas : int) -> torch.Tensor: #clas are the labels of our batch
        device = noisy_input.device
        cond_noise = self.noise_emb(c_noise.to(device)) # It has shape (B,cond_channels)
        cond_class = self.class_embed(clas.to(device)) #class embedding shape (B,cond_channels)
        cond = cond_class + cond_noise
        x = self.conv_in(noisy_input)
        x = self.activation(x) #conv -> activ
        for block in self.blocks:
            x = block(x,cond)
        x = self.conv_out(x)
        return x 


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor: #input needs to heave shape (B,)
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1) #it produces noise_emb with shape (B,cond_channels)

class conditional_BN(nn.Module): #The idea is to have regular batchNorm with no coefficients (affine = False)
    #And then we modify the output using our own predicted affine parameters
    def __init__(self,num_channels,cond_channels):
        super().__init__()
        self.num_channels = num_channels
        self.bn = nn.GroupNorm(num_groups=8,num_channels=num_channels,affine=False)
        self.lin = nn.Sequential(
            nn.Linear(cond_channels, 2 * num_channels),
            nn.SiLU(),
            nn.Linear(2 * num_channels, 2 * num_channels)
        ) #MLP that predicts the affine parameters
        #We need a gamma and a beta for each channel, that's why we have 2*num_channels
    def forward(self,x,cond): #cond:noise conditioning
        x = self.bn(x)
        out = self.lin(cond)
        gamma,beta = out.chunk(2,dim=-1) #We split the output since we have 2*num_channels
        gamma = gamma[-1,self.num_channels,1,1] #We do this to make the coefficients broadcastable when manipluating them with x
        beta = beta[-1,self.num_channels,1,1] #this has the same effect as doing unsqueeze multiple times
        return gamma * x + beta

#This is just the residual block
class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int,cond_channels) -> None:
        super().__init__()
        self.norm1 = nn.conditional_BN(nb_channels,cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.conditional_BN(nb_channels,cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.norm1(self.conv1(x))) #conv -> norm -> activ
        y = self.activation(self.norm2(self.conv2(y)))
        return x+y


def train_model(model,optimizer,criterion,nb_epochs): #trained to give the class cl
    model.train(True) #We're in training mode
    loss_list = []
    print(f'Our training is {nb_epochs} epochs.')
    for i in range(nb_epochs):
      n_batches = 0
      epoch_loss = 0
      for images,labels in train_load:
          images = images.to(device)
          labels = labels.to(device=device,dtype=torch.long)
          B = images.size(0) #the batch size
          sigma = sample_sigma(B).to(device) #sigma has shape (B,)
          noise = torch.randn_like(images,device=device)*sigma.view(-1,1,1,1)
          noisy_batch = images + noise
          c_in,c_out,c_skip,cnoise = sample_constants(Sigma,sigma)
          output = model(c_in*noisy_batch,cnoise,labels)
          loss = criterion(output,(images-c_skip*noisy_batch)/c_out)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
          n_batches += 1
      print(f'the loss at epoch {i} is : ',epoch_loss/n_batches)
      loss_list.append(epoch_loss/n_batches)
    model.train(False)
    return loss_list

############### Sampling functions:

def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=1.5): #it samples sigma for size n=Batch_size -> (B,) shape
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)

def sample_constants(sigma_data,sigma): #It return c_in,c_out,c_skip and c_noise
    c_in = 1/(torch.sqrt((sigma_data**2 + sigma**2)))
    c_out = (sigma*sigma_data)/(sigma**2+sigma_data**2)
    c_skip = sigma_data**2/(sigma**2 + sigma_data**2)
    c_noise = 0.25*torch.log(sigma)

    c_in = c_in.view(-1,1,1,1)
    c_out = c_out.view(-1,1,1,1)
    c_skip = c_skip.view(-1,1,1,1)
    c_noise = c_noise.view(-1) #because we use it in the noise_emb layer, it needs to have shape (B,)

    return c_in,c_out,c_skip,c_noise


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=1.5):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def Euler_sampling(noise,sigmas,model,sigma_data,clas_idx):
    #check if the class idx is a number or a tensor of labels
    model.eval()
    if clas_idx.numel() == 1:
            class_tensor = torch.full((noise.shape[0],), int(clas_idx.item()), dtype=torch.long, device='cuda')
    x = noise.clone().to('cuda')
    process = []
    for i, sigma in enumerate(sigmas):
        sigma = sigma.repeat(x.shape[0]).to(device) #so that it has the shape that sample_constants expects
        with torch.no_grad():
            x = x.to(device)
            cin,cout,cskip,cnoise = sample_constants(sigma_data,sigma)
            cin = cin.to(device)
            cout = cout.to(device)
            cskip = cskip.to(device)
            cnoise = cnoise.to(device)
            input = cin*x
            output = model(input,cnoise,class_tensor)
            x_denoised = cskip*x + cout*output
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        sigma = sigma.view(-1,1,1,1)
        sigma_next = sigma_next.view(-1,1,1,1)
        d = (x - x_denoised) / sigma

        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method
        process.append(x.clone().to('cpu')) #we add a snapshot of this step to process
#the output is of the size of the noise. We iteratively apply the denoiser
    return x.to('cpu'),process

def Sampling(nbr_images,model,nbr_steps,cl):
    sigmas = build_sigma_schedule(steps=nbr_steps)
    #it returns a tensor with number of steps elements
    noise = torch.randn(size=(nbr_images,1,32,32)) * sigmas[0]
    images,process = Euler_sampling(noise,sigmas,model,Sigma,cl)
    return images,process


#We visualize the images using the process output from the Euler_sampling function
def visualize_images(images):
        x = images.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
        x = make_grid(x)
        x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
        plt.imshow(x)
        plt.axis("off")
        plt.show()




