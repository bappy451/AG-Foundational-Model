import torch

def test():
    N, C, H, W = 24, 3, 640, 640
    p = 32
    images = torch.rand(N, C, H, W).cuda() * 255.0
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, w = H // p, W // p
        patches = images.reshape(N, C, h, p, w, p)
        mean = patches.mean(dim=(1, 3, 5), keepdim=True)
        var = patches.var(dim=(1, 3, 5), unbiased=False, keepdim=True)
        norm_images = (patches - mean) / torch.sqrt(var + 1e-6)
        norm_images = norm_images.reshape(N, C, H, W)
        
        print("Images NaN?", torch.isnan(images).any().item())
        print("Mean NaN?", torch.isnan(mean).any().item())
        print("Var NaN?", torch.isnan(var).any().item())
        print("Var negative?", (var < 0).any().item())
        print("Norm_images NaN?", torch.isnan(norm_images).any().item())

test()
