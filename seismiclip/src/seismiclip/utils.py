"""
Utility functions.
@hatsyim
"""

import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter
from math import sqrt
import os

def expand_array(array, sigmas):
    num_smoothings = len(sigmas)
    expanded_array = np.zeros((array.shape[0] + num_smoothings * array.shape[0], array.shape[1], array.shape[2]))

    expanded_array[::num_smoothings + 1] = array

    for i in range(num_smoothings):
        smoothed_image = gaussian_filter(array, sigma=[0,sigmas[i],sigmas[i]])
        expanded_array[i+1::num_smoothings + 1] = smoothed_image

    return expanded_array

def set_seed(seed):
    """
    :param seed: An integer of random number.

    :return: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def generate_gathers(vp_true, vs_true, rho_true, elastic=False):
        
    # Set parameters
    freq = 7
    dx = 25
    dz = 3
    dt = 0.004 # 1ms
    
    nt = int(2.4 / dt) # 6s
    num_dims = 2
    num_shots = 3 #259
    num_sources_per_shot = 1
    num_receivers_per_shot = 2*64 #256
    source_spacing = 1270 #255.9 #255 #159 #30.0
    receiver_spacing = 2*9.9
    device = torch.device('cuda')
    
    vp_true = vp_true.float().to(device)
    vs_true = vs_true.float().to(device)
    rho_true = rho_true.float().to(device)

    source_locations = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    source_locations[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
    source_locations[:, 0, 0] += dx
    receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    receiver_locations[:, :, 0] += dx
    receiver_locations[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
    receiver_locations[:, :, 1] = receiver_locations[0, :, 1].repeat(num_shots, 1)

    source_locations = source_locations/10
    receiver_locations = receiver_locations/10
    
    # print(source_locations)

    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
        .repeat(num_shots, num_sources_per_shot, 1)
        .to(device)
    ).clone().to(device)

    if elastic:
        out = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_true, vs_true, rho_true),
            [dz, dx], dt, source_amplitudes_y=source_amplitudes.to(device),
            source_locations_y=source_locations.to(device),
            receiver_locations_p=receiver_locations.to(device),
            accuracy=4,
            pml_freq=freq,
            pml_width=[50, 50, 50, 50]
        )[-3] # output pressure fields
        
    else:
        out = deepwave.scalar(vp_true, [dz,dx], dt, source_amplitudes=source_amplitudes.to(device),
                    source_locations=source_locations.to(device),
                    receiver_locations=receiver_locations.to(device),
                    accuracy=8, pml_width=[0, 50, 50, 50],
                    pml_freq=freq)[-1]
    
    receiver_amplitudes_true = out #+ 2e-9*torch.randn_like(out)
    
    return receiver_amplitudes_true.detach().cpu().half().numpy()

def normalize_min1toplus1(x, xmin, xmax):
    return(2*(x-xmin)/(xmax-xmin)-1)

def denormalize_min1toplus1(x, xmin, xmax):
    return (x+1)/2*(xmax-xmin)+xmin

def normalize_mean_std(x):
    return((x - x.mean()) / x.std())

def log1p(x, alpha=1):
    """Log1p weighting"""
    return torch.sign(x) * torch.log10(torch.abs(alpha*x)+1)


def log1p_1(x, alpha=1):
    """Inverse Log1p weighting"""
    return torch.sign(x) * (10**torch.abs(x)-1)/alpha

class SeismicVelocityDataset(torch.utils.data.Dataset):
    def __init__(self, seismic, velocity, device='cuda', normalize=True):
        
        self.seismic = seismic
        self.velocity = velocity
        
        if normalize:
            self.velocity[:,0] = normalize_min1toplus1(velocity[:,0], velocity[:,0].min(), velocity[:,0].max())
            self.velocity[:,1] = normalize_min1toplus1(velocity[:,1], velocity[:,1].min(), velocity[:,1].max())
            self.velocity[:,2] = normalize_min1toplus1(velocity[:,2], velocity[:,2].min(), velocity[:,2].max())
            
        self.device = device
    
    def __len__(self):
        return self.velocity.shape[0]
    
    def __getitem__(self, index):
        return self.seismic[index].to(self.device), self.velocity[index].to(self.device)
    
class SeismicVelocityDataset2(torch.utils.data.Dataset):
    def __init__(self, data_dirs, shot_indices=None, device='cuda', normalize=True, norm_type='mean_std', transform=None, alpha=1, return_real_vel=False):
        # Check if seismic_dirs and velocity_dirs are lists of directories
        if not isinstance(data_dirs, list):
            raise ValueError("seismic_dirs and velocity_dirs should be lists of directory paths.")

        self.seismic_files = []
        self.vp_files = []
        self.vs_files = []
        self.labels = []  # Store the label/group information

        # Load files from each directory
        for group_label, data_dir in enumerate(data_dirs):
            seismic_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('data_z_')])
            vp_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('vp_')])
            vs_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('vs_')])

            self.seismic_files.extend(seismic_files)
            self.vp_files.extend(vp_files)
            self.vs_files.extend(vs_files)
            self.labels.extend([group_label] * len(vp_files))  # Assign group label for each file

        self.device = device
        self.normalize = normalize
        self.shot_indices = shot_indices
        self.norm_type = norm_type
        self.transform = transform
        self.alpha = alpha
        self.return_real_vel = return_real_vel

        # Initialize memory-mapped arrays for seismic and velocity data
        self.seismic_mmaps = [np.load(f, mmap_mode='r') for f in self.seismic_files]
        self.vp_mmaps = [np.load(f, mmap_mode='r') for f in self.vp_files]
        self.vs_mmaps = [np.load(f, mmap_mode='r') for f in self.vs_files]
        
        # Calculate the total number of samples across all files
        self.total_samples = sum(mmap.shape[0] for mmap in self.vp_mmaps)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        # Determine which file and sample to access
        file_index = 0
        while index >= self.vp_mmaps[file_index].shape[0]:
            index -= self.vp_mmaps[file_index].shape[0]
            file_index += 1
        
        # Load data using memory-mapped arrays
        seismic_data = self.seismic_mmaps[file_index][index].transpose(0, 2, 1)
        
        # If shot indices are specified, select only those shots
        if self.shot_indices is not None:
            seismic_data = seismic_data[self.shot_indices][0]
        
        vp_data = self.vp_mmaps[file_index][index, 0]
        vs_data = self.vs_mmaps[file_index][index, 0]
        rho_data = vp_to_rho(vp_data)
        
        # Stack the velocity properties to form the velocity array
        velocity_data = np.stack([vp_data, vs_data, rho_data], axis=0)
        velocity_data = velocity_data[:, 3:-3, 3:-3] # Crop manually to 64 x 64

        if self.return_real_vel:
            velocity_data_real = np.copy(velocity_data)
        
        # Normalize if required
        if self.normalize:
            if self.norm_type == "min1toplus1":
                seismic_data = normalize_min1toplus1(seismic_data, seismic_data.min(), seismic_data.max())
                velocity_data[0] = normalize_min1toplus1(velocity_data[0], velocity_data[0].min(), velocity_data[0].max())
                velocity_data[1] = normalize_min1toplus1(velocity_data[1], velocity_data[1].min(), velocity_data[1].max())
                velocity_data[2] = normalize_min1toplus1(velocity_data[2], velocity_data[2].min(), velocity_data[2].max())
            elif self.norm_type == "mean_std":
                seismic_data = normalize_mean_std(seismic_data)
                velocity_data[0] = normalize_mean_std(velocity_data[0])
                velocity_data[1] = normalize_mean_std(velocity_data[1])
                velocity_data[2] = normalize_mean_std(velocity_data[2])

        if self.transform == "log1p":
            seismic_data = log1p(torch.tensor(seismic_data), alpha=self.alpha)

        # Get the group label for the current file
        group_label = self.labels[file_index]
        
        if self.return_real_vel:
            return torch.tensor(seismic_data), torch.tensor(velocity_data), group_label, torch.tensor(velocity_data_real)

        return torch.tensor(seismic_data), torch.tensor(velocity_data), group_label

def gaussian_2d_filter(x, sigmax=1.0, sigmay=1.0):
    """
    Apply Gaussian filter to 2D image

    :param x: input image
    :param sigmax: sigma for x axis
    :param sigmay: sigma for y axis
    
    :return: filtered 2D image
    """
    return gaussian_filter(x, sigma=[sigmax, sigmay])

def set_seed(seed):
    """
    :param seed: An integer of random number.

    :return: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def butter_filter(data, lowcut, highcut, fs, order=6, filt_type='band'):
    """
    Define a butter filter.

    :params data: A time series array.
    :params lowcut: Low frequency band (Hz).
    :params highcut: High frequency band (Hz).
    :params fs: Sampling frequency (Hz).
    
    :return: Filtered data.
    """
    
    b, a = butter(order, [lowcut, highcut], fs=fs, btype=filt_type)
    y = lfilter(b, a, data)
    return y

from scipy.ndimage import gaussian_filter

def expand_array(array, sigmas):
    """
    Expand a given Numpy array using its smoothed version of size len(sigmas).

    :params array: A numpy array.
    :params sigmas: A list of Gaussian sigmas.
    
    :return: Extended array.
    """
    
    num_smoothings = len(sigmas)
    expanded_array = np.zeros((array.shape[0] + num_smoothings * array.shape[0], array.shape[1], array.shape[2]))

    expanded_array[::num_smoothings + 1] = array

    for i in range(num_smoothings):
        smoothed_image = gaussian_filter(array, sigma=[0,sigmas[i],sigmas[i]])
        expanded_array[i+1::num_smoothings + 1] = smoothed_image

    return expanded_array

def snr(x, x_est):
    """
    Compute the signal-to-noise ratio (SNR) in dB.
    
    :params x: Original signal
    :params x_est: Estimated signal
    
    :return: SNR in dB
    """
    
    return 10.0 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    
    :return: an [N x dim] Tensor of positional embeddings.
    """ 
    
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half,
                                             dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], 
                              dim=-1)
    return embedding

def norm_layer(channels):
    """
    Create group normalization layer.
    
    :param channels: int number of channels.
    
    :return: PyTorch module.
    """
    
    return nn.GroupNorm(32, channels)

def linear_beta_schedule(timesteps):
    """
    Linear noise scheduler.
    
    :param timesteps: array of time steps.
    
    :return: PyTorch array.
    """
    
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    #beta_end = scale * 0.0005
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise scheduler in https://arxiv.org/abs/2102.09672.
    
    :param timesteps: array of time steps.
    
    :return: PyTorch array.
    """
    
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def vp_to_vs(vp):
    """
    Convert compressional to shear velocity.
    
    :param vp: a compressional velocity array.
    
    :return vs: a shear velocity array.
    """
    
    return vp/sqrt(3)

def vp_to_rho(vp):
    """
    Convert compressional to shear velocity.
    
    :param vp: a compressional velocity array.
    
    :return rho: a density array.
    """
    
    return 0.31*(vp)**(0.25)

def normalize_(vp, vmax=5000, vmin=3000):
    """
    Normalize compressional velocity.
    
    :param vp: a compressional velocity array.
    :param vmax: maximum vp.
    :param vmin: minimum vp.
    
    :return vp: a normalized vp [0-1].
    """
    
    return (vp - vmin)/(vmax-vmin)*2.0 - 1.0

def denormalize_(vp, vmax=5000, vmin=3000):
    """
    Denormalize compressional velocity.
    
    :param vp: a normalized compressional velocity array [0-1].
    :param vmax: maximum vp.
    :param vmin: minimum vp.
    
    :return vp: a denormalized vp.
    """
    
    return (vp + 1.0)/2.0*(vmax-vmin)+vmin

def split_data_to_size(data, size, stride):
    """
    Extract patches of images.

    :param data: Input images.
    :param size: Kernel size.
    :param stride: Stride size.

    :return: Patch of images.
    """
    b, c, h, w = data.shape
    h_split, w_split =  (h-size[0]+1)//stride[0], (w-size[1]+1)//stride[1]
    w_split_add = (w-size[1]+1)%stride[1]
    b_new, c_new, h_new, w_new = h_split * (w_split + w_split_add), c, size[0], size[1]
    data_split = torch.zeros(b_new,c_new,h_new,w_new).to(data.device)
    for i in range(h_split):
        for j in range(w_split):
            data_split[i*w_split+j,:,:,:] = data[:,:,i*stride[0]:i*stride[0]+h_new,j*stride[1]:j*stride[1]+w_new]
    for i in range(w_split_add):
        data_split[(h_split-1)*w_split+w_split+i,:,:,:] = data[:,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h_new,w_split*stride[1]+i:w_split*stride[1]+i+w_new]
    return data_split.reshape(-1,c_new,h_new,w_new)
    
def merge_data_to_size(data, size, stride):
    """
    Extract patches of images.

    :param data: Input images.
    :param size: Kernel size.
    :param stride: Stride size.

    :return: Patch of images.
    """
    b, c, h, w = data.shape
    h_split, w_split =  (size[0]-h+1)//stride[0], (size[1]-w+1)//stride[1]
    w_split_add = (size[1]-w+1)%stride[1]
    b_new, c_new, h_new, w_new = 1, c, size[0], size[1]
    data_merge = torch.zeros(b_new,c_new,h_new,w_new).to(data.device)
    mask = torch.zeros((b_new*c, h_new, w_new)).reshape(b_new,c,h_new,w_new).to(data.device)
    for i in range(h_split):
        for j in range(w_split):
            data_merge[0,:,i*stride[0]:i*stride[0]+h,j*stride[1]:j*stride[1]+w] += data[i*w_split+j,:,:,:]
            mask[0,:,i*stride[0]:i*stride[0]+h,j*stride[1]:j*stride[1]+w] +=1
    for i in range(w_split_add):
        data_merge[0,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h,w_split*stride[1]+i:w_split*stride[1]+i+w] += data[w_split+i,:,:,:]
        mask[0,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h,w_split*stride[1]+i:w_split*stride[1]+i+w] += 1
    return data_merge.reshape(-1,c,h_new,w_new)/mask

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, normalize=False, water_velocity=1.5, water_depth=0, image_size=128):
        
        if water_depth != 0:
            data[:,:water_depth,:] = water_velocity
              
        self.image_size = image_size 
        
        if normalize:
            self.max_velocity = torch.max(data)    
            self.min_velocity = torch.min(data)   
            self.data = self._normalize(data)
        else:
            self.data = data 
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample.reshape(-1,self.image_size,self.image_size).float()

    def _normalize(self, array):
        normalized_array = 2 * (array - self.min_velocity) / (self.max_velocity - self.min_velocity) - 1
        return normalized_array