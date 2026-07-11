import os

files = [
    r'E:\AG_Dataset\AG-Foundational-Model\src\ag_foundation\training\cls_runner.py',
    r'E:\AG_Dataset\AG-Foundational-Model\src\ag_foundation\training\count_runner.py',
    r'E:\AG_Dataset\AG-Foundational-Model\src\ag_foundation\training\det_runner.py',
    r'E:\AG_Dataset\AG-Foundational-Model\src\ag_foundation\training\temporal_runner.py'
]

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    target = 'model_name=f"vit_base_patch16_{crop_size}" if model_cfg.get("model_name", "B") == "B" else "vit_small_patch16_224",'
    replacement = 'model_name=model_cfg.get("model_name", "B"),'
    
    if target in content:
        content = content.replace(target, replacement)
        with open(f, 'w') as file:
            file.write(content)
        print(f'Fixed {f}')
    else:
        print(f'Target not found in {f}')
