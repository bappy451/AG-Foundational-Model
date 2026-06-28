import torch

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
except ImportError:
    # Handle environments where DALI is not installed (like native Windows)
    pass

@pipeline_def
def _wds_pipeline(tar_urls, crop_size=512):
    """
    DALI pipeline definition that reads WebDataset tarballs,
    decodes images on the GPU via nvJPEG, and standardizes them.
    """
    # DALI WebDataset reader natively supports tarballs
    jpegs, keys = fn.readers.webdataset(
        paths=tar_urls,
        ext=["jpg", "png", "jpeg", "tif", "tiff"],
        missing_component_behavior="skip",
        random_shuffle=True,
        initial_fill=1000
    )
    
    # Decode directly into GPU memory
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    # Resize to base crop size on GPU
    images = fn.resize(
        images,
        resize_x=crop_size,
        resize_y=crop_size,
        interp_type=types.INTERP_LINEAR
    )
    
    # Convert HWC uint8 -> CHW float32 [0.0, 1.0] to match PyTorch's transforms.ToTensor()
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0]
    )
    
    return images

class DaliDictWrapper:
    """Wraps DALI's DALIGenericIterator to yield dictionaries matching the CPU WDS loader."""
    def __init__(self, dali_iterator, epoch_batches):
        self.dali_iterator = dali_iterator
        self.epoch_batches = epoch_batches
        
    def __iter__(self):
        for batch in self.dali_iterator:
            yield {"image": batch[0]["image"]}
            
    def __len__(self):
        return self.epoch_batches if self.epoch_batches else 0

def build_dali_wds_dataloader(tar_urls, batch_size=64, num_workers=4, epoch_batches=None, crop_size=512):
    """
    Builds the DALI PyTorch DataLoader.
    Expects DALI to be installed (Colab/Linux).
    """
    pipe = _wds_pipeline(
        tar_urls=tar_urls,
        crop_size=crop_size,
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0
    )
    
    iterator_size = epoch_batches * batch_size if epoch_batches else -1
    
    iterator = DALIGenericIterator(
        [pipe],
        output_map=["image"],
        size=iterator_size,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )
    
    return DaliDictWrapper(iterator, epoch_batches)
