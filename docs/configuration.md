# Configuration Reference

YAML files use four top-level sections: `data`, `runtime`, `model`, and
`optimizer`. Command-line values override YAML values.

Paths inside YAML are resolved relative to the YAML file, not the shell's
working directory.

Unknown sections and keys are rejected rather than ignored. Numeric constraints
such as positive epochs/LR, crop divisibility, valid probabilities, DINO
temperatures, crop scales, and EMA ranges are validated before data loading or
pretrained-weight download begins.

## Data

| Key | Meaning |
| --- | --- |
| `data_root` | Directory, image, GeoTIFF, or ZIP root |
| `catalog_path` | Optional portable CSV catalog |
| `crop_size` | Square crop and ViT input size; divisible by the selected patch size |
| `channels` | Exact number of input bands |
| `batch_size` | Samples per optimizer step |
| `num_workers` | DataLoader worker processes |
| `prefetch_factor` | Batches prefetched per worker |
| `val_fraction` | Target sample fraction under a group-disjoint split |

## Runtime

| Key | Meaning |
| --- | --- |
| `output_dir` | Run artifact directory |
| `epochs` | Final target epoch |
| `seed` | Python, NumPy, Torch, split, and loader seed |
| `precision` | `fp32`, `fp16`, or `bf16` |
| `device` | `auto`, `cpu`, `cuda`, or `mps` |
| `warmup_epochs` | Epoch-level linear warmup before cosine decay |
| `resume` | Resume from `<output_dir>/last.pt` if present |
| `resume_from` | Explicit checkpoint path |
| `log_every` | Batch interval for console loss output |
| `save_visualizations` | Enable model-output figures |
| `visualization_every` | Figure interval in epochs |
| `visualization_samples` | Samples shown per figure |

## Shared Model Keys

| Key | Meaning |
| --- | --- |
| `model_name` | `S`, `B`, or `L` |
| `pretrained_backbone` | Load the selected official checkpoint family |
| `pretrained_source` | `imagenet`, `dinov2`, `dinov3`, or `mae` |
| `pretrained_cfg` | Optional `timm` pretrained variant for ImageNet-family runs |
| `gradient_checkpointing` | Trade compute for activation memory |
| `drop_rate` | Backbone dropout |
| `attn_drop_rate` | Attention dropout |
| `drop_path_rate` | Stochastic-depth rate |

## MIM Keys

| Key | Meaning |
| --- | --- |
| `mask_ratio` | Fraction of patches replaced by mask tokens |

## DINO Keys

| Key | Meaning |
| --- | --- |
| `dino_out_dim` | Prototype/logit dimension |
| `dino_hidden_dim` | Projection-head hidden width |
| `dino_bottleneck_dim` | Projection bottleneck width |
| `head_nlayers` | Projection-head linear depth |
| `num_global_crops` | Teacher/student global views |
| `num_local_crops` | Additional student-only views |
| `global_crop_scale` | Relative crop-area interval |
| `local_crop_scale` | Relative local crop-area interval |
| `student_temperature` | Student softmax temperature |
| `teacher_temperature` | Teacher softmax temperature |
| `teacher_momentum_start` | Initial EMA coefficient |
| `teacher_momentum_end` | Final EMA coefficient |
| `center_momentum` | Teacher-center EMA coefficient |

## Optimizer

| Key | Meaning |
| --- | --- |
| `learning_rate` | AdamW base learning rate |
| `weight_decay` | AdamW weight decay |

## Example

```yaml
data:
  data_root: ../data/five_band_tiles
  catalog_path: ../catalogs/five_band.csv
  crop_size: 224
  channels: 5
  batch_size: 64
  num_workers: 8
  prefetch_factor: 2
  val_fraction: 0.1

runtime:
  output_dir: ../runs/mim_vit_b_5band
  epochs: 100
  seed: 27
  precision: bf16
  device: auto
  warmup_epochs: 10
  resume: true
  resume_from: null
  log_every: 25
  save_visualizations: true
  visualization_every: 1
  visualization_samples: 4

model:
  model_name: B
  pretrained_backbone: true
  pretrained_source: mae
  pretrained_cfg: null
  mask_ratio: 0.75
  gradient_checkpointing: true
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1

optimizer:
  learning_rate: 0.0001
  weight_decay: 0.05
```

## CLI Overrides

```bash
python -m ag_foundation train-mim \
  --config configs/my_mim.yaml \
  --model-name L \
  --batch-size 32 \
  --epochs 200 \
  --precision bf16
```

Boolean flags support positive and negative forms, such as
`--pretrained-backbone` and `--no-pretrained-backbone`.

Patch-size reminders:

- ImageNet and MAE backbones use patch size 16.
- DINOv3 backbones use patch size 16.
- DINOv2 backbones use patch size 14.
