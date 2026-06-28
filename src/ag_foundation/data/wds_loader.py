import sys
import webdataset as wds
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

if sys.platform == 'win32':
    from webdataset.gopen import gopen_schemes
    gopen_schemes['winfile'] = lambda url, mode='rb', bufsize=8192, **kw: open(url.replace('winfile://', ''), mode, buffering=bufsize)


class SizedWebDataset(IterableDataset):
    def __init__(self, pipeline, length):
        self.pipeline = pipeline
        self.length = length
        
    def __iter__(self):
        return iter(self.pipeline)
        
    def __len__(self):
        return self.length

def build_wds_dataloader(tar_urls, batch_size=64, num_workers=8, epoch_batches=None):
    """
    Build a WebDataset pipeline that streams TAR shards and decodes on CPU.
    """
    # A simple pipeline to load tarballs, shuffle them, and decode the images
    # to PyTorch float32 tensors (C, H, W).
    
    # We use basic torchvision transforms to just convert to tensor on CPU
    # The actual heavy augmentations happen on the GPU via Kornia.
    to_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    pipeline = (
        wds.WebDataset(tar_urls, resampled=True)
        .shuffle(1000)
        .decode("pil")
        .rename(image="jpg;png;jpeg;tif;tiff", handler=wds.warn_and_continue)
        .map_dict(image=to_tensor)
        .batched(batch_size, partial=False)
    )
    
    if epoch_batches is not None:
        pipeline = pipeline.with_epoch(epoch_batches)
        dataset = SizedWebDataset(pipeline, epoch_batches)
    else:
        dataset = pipeline
    
    loader = DataLoader(
        dataset,
        batch_size=None, # batching is handled inside WebDataset pipeline
        num_workers=num_workers,
        pin_memory=True, # crucial for fast CPU -> GPU transfer later
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader
