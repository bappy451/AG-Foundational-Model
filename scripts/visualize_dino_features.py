import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add src to pythonpath so we can import ag_foundation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ag_foundation.models.official_vit import RemoteSensingViT

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 Dense Patch Feature Cosine Similarities")
    parser.add_argument("--image", type=str, help="Path to a single input image")
    parser.add_argument("--image_dir", type=str, help="Path to a directory of images to randomly sample from")
    parser.add_argument("--num_random", type=int, default=10, help="Number of random images to process if using --image_dir")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained DINOv3 best.pt checkpoint")
    parser.add_argument("--output", type=str, default="similarity_heatmap.png", help="Output path for the visualization (or output directory if --image_dir is used)")
    parser.add_argument("--x", type=int, default=112, help="X coordinate (pixel) of the target patch (default: center of 224x224)")
    parser.add_argument("--y", type=int, default=112, help="Y coordinate (pixel) of the target patch (default: center of 224x224)")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Must provide either --image or --image_dir")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading backbone from {args.checkpoint}...")
    model = RemoteSensingViT(
        image_size=224,
        model_name="B",
        pretrained_backbone=False,
        pretrained_source="dinov3",
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student_backbone."):
            new_state_dict[k.replace("student_backbone.", "")] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
    ])

    patch_size = model.patch_size[0]
    grid_size = model.grid_size[0]
    
    if args.x < 0 or args.x >= 224 or args.y < 0 or args.y >= 224:
        print("Error: Target coordinates (x, y) must be within [0, 223].")
        return

    patch_x = args.x // patch_size
    patch_y = args.y // patch_size
    target_patch_idx = patch_y * grid_size + patch_x

    # Collect images
    images_to_process = []
    if args.image_dir:
        import random
        dataset_path = Path(args.image_dir)
        all_images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
        if not all_images:
            print(f"No images found in {args.image_dir}")
            return
        num_to_pick = min(args.num_random, len(all_images))
        images_to_process = random.sample(all_images, num_to_pick)
        
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_to_process = [Path(args.image)]

    for idx, img_path in enumerate(images_to_process):
        try:
            raw_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            continue

        raw_img = raw_img.resize((224, 224))
        img_tensor = transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.forward_features(img_tensor)
            
        features = features.squeeze(0)
        features = F.normalize(features, p=2, dim=-1)
        target_feature = features[target_patch_idx].unsqueeze(0)
        similarities = torch.mm(features, target_feature.t()).squeeze(-1)
        similarities = similarities.view(grid_size, grid_size).cpu().numpy()
        
        sim_resized = Image.fromarray(similarities).resize((224, 224), resample=Image.BICUBIC)
        sim_resized = np.array(sim_resized)
        sim_resized = (sim_resized - sim_resized.min()) / (sim_resized.max() - sim_resized.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_img)
        axes[0].scatter(args.x, args.y, marker='x', color='red', s=100, linewidth=2)
        axes[0].set_title(f"Original Image\nTarget (X:{args.x}, Y:{args.y})")
        axes[0].axis("off")

        axes[1].imshow(raw_img)
        heatmap = axes[1].imshow(sim_resized, cmap='jet', alpha=0.5)
        axes[1].scatter(args.x, args.y, marker='x', color='red', s=100, linewidth=2)
        axes[1].set_title("DINO Feature Cosine Similarity")
        axes[1].axis("off")
        
        plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()

        if args.image_dir:
            out_path = Path(args.output) / f"vis_{idx:02d}_{img_path.name}"
        else:
            out_path = Path(args.output)

        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved successfully to {out_path}!")

if __name__ == "__main__":
    main()
