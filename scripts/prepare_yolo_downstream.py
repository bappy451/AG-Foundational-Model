import argparse
import os
import torch
from ultralytics import YOLO

def extract_yolo_weights(spark_ckpt_path: str, output_path: str, base_yolo_model: str = "yolo11l-obb.pt"):
    print(f"Loading SparK checkpoint from {spark_ckpt_path}...")
    ckpt = torch.load(spark_ckpt_path, map_location="cpu", weights_only=False)
    spark_state = ckpt.get("model_state_dict", ckpt)
    
    print(f"Loading official YOLO model ({base_yolo_model}) as base...")
    yolo = YOLO(base_yolo_model)
    dense_state = yolo.model.state_dict()
    
    layer_map = {
        "encoder.stem.": "model.0.",
        "encoder.stage1.0.": "model.1.",
        "encoder.stage1.1.": "model.2.",
        "encoder.stage2.0.": "model.3.",
        "encoder.stage2.1.": "model.4.",
        "encoder.stage3.0.": "model.5.",
        "encoder.stage3.1.": "model.6.",
        "encoder.stage4.0.": "model.7.",
        "encoder.stage4.1.": "model.8.",
        "encoder.stage4.2.": "model.9.",
    }
    
    mapped_count = 0
    new_state = dense_state.copy()
    
    for sparse_key, sparse_tensor in spark_state.items():
        if not sparse_key.startswith("encoder."):
            continue
            
        # Find mapping
        dense_key = None
        for sp_prefix, dense_prefix in layer_map.items():
            if sparse_key.startswith(sp_prefix):
                dense_key = sparse_key.replace(sp_prefix, dense_prefix, 1)
                dense_key = dense_key.replace(".bn.ln.", ".bn.")
                break
                
        if dense_key is None:
            continue
            
        if "bridge.weight" in dense_key:
            continue
            
        if dense_key in new_state:
            dense_tensor = new_state[dense_key]
            
            # Convert spconv weight (out_c, k, k, in_c) to PyTorch dense (out_c, in_c, k, k)
            if sparse_tensor.ndim == 4:
                sparse_tensor = sparse_tensor.permute(0, 3, 1, 2)
            
            if sparse_tensor.shape == dense_tensor.shape:
                new_state[dense_key] = sparse_tensor
                mapped_count += 1
            else:
                # Need to fuse bridge and conv!
                bridge_key = sparse_key.replace("conv.weight", "bridge.weight")
                if bridge_key in spark_state:
                    bridge_tensor = spark_state[bridge_key]
                    if bridge_tensor.ndim == 4:
                        bridge_tensor = bridge_tensor.permute(0, 3, 1, 2)
                        out_c = bridge_tensor.shape[0]
                        in_c_bridge = bridge_tensor.shape[1]
                    else:
                        print(f"Bridge tensor for {dense_key} is not 4D: {bridge_tensor.shape}")
                        continue
                    
                    in_c, _, k1, k2 = sparse_tensor.shape
                    
                    b_mat = bridge_tensor.view(out_c, in_c_bridge)
                    c_mat = sparse_tensor.reshape(in_c_bridge, -1)
                    
                    fused = torch.mm(b_mat, c_mat).view(out_c, in_c, k1, k2)
                    
                    if fused.shape == dense_tensor.shape:
                        new_state[dense_key] = fused
                        mapped_count += 1
                    else:
                        print(f"Shape mismatch after fusion for {dense_key}: {fused.shape} != {dense_tensor.shape}")
                else:
                    print(f"Shape mismatch for {dense_key}: {sparse_tensor.shape} != {dense_tensor.shape} and no bridge found.")
        else:
            # Silently ignore if not in dense_state
            pass
            
    print(f"Successfully mapped {mapped_count} tensors to dense YOLO.")
    
    yolo.model.load_state_dict(new_state)
    
    ckpt_to_save = {
        'epoch': -1,
        'model': yolo.model.half(),
        'optimizer': None,
        'train_args': {},
        'date': '',
        'version': '8.3.0',
    }
    
    torch.save(ckpt_to_save, output_path)
    print(f"Saved extracted YOLO weights to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r"E:\AG_Dataset\AG-Foundational-Model\runs\spark_yolo_full\spark_yolo_best.pt")
    parser.add_argument("--out", type=str, default=r"E:\AG_Dataset\AG-Foundational-Model\runs\spark_yolo_full\spark_yolo11l-obb.pt")
    args = parser.parse_args()
    
    extract_yolo_weights(args.ckpt, args.out)
