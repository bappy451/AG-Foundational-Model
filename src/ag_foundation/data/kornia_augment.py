import torch
import torch.nn as nn
import kornia.augmentation as K

class RandomRot90(nn.Module):
    """
    Random discrete 90-degree rotations (90, 180, 270).
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if torch.rand(1, device=x.device).item() < self.p:
            k = torch.randint(1, 4, (1,)).item() # 1, 2, or 3
            return torch.rot90(x, k, [2, 3]).contiguous()
        return x.contiguous()

class Contiguous(nn.Module):
    def forward(self, x):
        return x.contiguous()

class KorniaMultiCropAugmentation(nn.Module):
    """
    GPU-accelerated DINO augmentation pipeline using Kornia.
    Takes a single PyTorch tensor (B, C, H, W) on CUDA and outputs a tuple
    of global and local crop augmented tensors.
    """
    def __init__(
        self,
        global_crops_scale=(0.14, 1.0),
        local_crops_scale=(0.05, 0.14),
        global_crops_number=2,
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        super().__init__()
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        
        # 1. Global Crop Transforms
        self.global_transform = nn.Sequential(
            K.RandomResizedCrop(size=(global_crops_size, global_crops_size), scale=global_crops_scale),
            self._get_spatial_transforms(),
            self._get_color_transforms(),
            self._get_blur(kernel_size=5, sigma=(0.1, 2.0), p=0.5), # DINO uses blur with 0.5 prob on global
            Contiguous(),
            K.Normalize(mean=mean, std=std)
        )
        
        # 2. Local Crop Transforms
        self.local_transform = nn.Sequential(
            K.RandomResizedCrop(size=(local_crops_size, local_crops_size), scale=local_crops_scale),
            self._get_spatial_transforms(),
            self._get_color_transforms(),
            self._get_blur(kernel_size=3, sigma=(0.1, 2.0), p=0.1), # DINO uses less blur prob on local
            Contiguous(),
            K.Normalize(mean=mean, std=std)
        )
        
    def _get_spatial_transforms(self):
        return nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            RandomRot90(p=0.5)
        )
        
    def _get_color_transforms(self):
        return K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            p=0.8
        )
        
    def _get_blur(self, kernel_size, sigma, p):
        # Kornia RandomGaussianBlur takes a float or tuple for sigma
        # kernel_size must be an odd integer or tuple of odds.
        return K.RandomGaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=sigma, p=p)

    def forward(self, images: torch.Tensor):
        """
        Input:
            images: Tensor of shape (B, 3, H, W) on GPU.
        Returns:
            Tuple of Tensors containing global and local crops.
        """
        crops = []
        
        # Generate global crops
        for _ in range(self.global_crops_number):
            crops.append(self.global_transform(images))
            
        # Generate local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(images))
            
        return tuple(crops)
