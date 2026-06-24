# Pretraining Data Pipeline: Dataset.yml Update, Data Loading & Evaluation Plan

## System Configuration Summary

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Core i9-14900KF (32 threads) |
| **GPU** | NVIDIA RTX 4090 (24 GB VRAM) |
| **RAM** | 64 GB (≈19 GB free) |
| **Disk E:** | 7.5 TB total, **2.7 TB free** |
| **CUDA** | 13.2 (Driver 595.95) |
| **Python Env** | `venv` (Python 3.12, PyTorch 2.12+CPU, timm 1.0.27) |
| **Alt Env** | `torchenv` (Python 3.12, PyTorch 2.6+CUDA 12.6, **no timm**) |

> [!IMPORTANT]
> The active `venv` conda environment has all project deps but **CPU-only PyTorch**. The `torchenv` env has CUDA PyTorch but **no timm**. For actual training, we need CUDA PyTorch + timm in the same environment. The data loading code and tests can run on CPU PyTorch in `venv`.

---

## Phase 1: Update Dataset.yml

The current `Dataset.yml` lists 31 candidate sources but is **incomplete** — the Pretraining folder contains 8–10 additional datasets not registered in the YAML, and there are duplicate ZIP files wasting ~16 GB.

### Datasets to ADD to Dataset.yml (present in Pretraining but not in YAML)

| Dataset | Size | Type | ID |
|---------|------|------|----|
| Agriculture-Vision-2021.tar.gz | 19.59 GB | Aerial/UAV field-level | `agriculture_vision_2021` |
| PlantSeg (Plant Disease Segmentation) | 1.59 GB | Disease segmentation | `plantseg` |
| DeepWeeds (Multiclass Weed Species) | 468 MB | Weed detection | `deepweeds` |
| PlantNet 300K | 29.48 GB | Species classification | `plantnet_300k` |
| UNL-CPPD | 7.20 GB | Plant phenotyping | `unl_cppd` |
| OPPD (Open Plant Phenotyping Database) | Extracted | Plant phenotyping | `oppd` |
| Corn Kernel Counting | 297 MB | Kernel counting | `corn_kernel_counting` |
| Longitudinal Nutrient Deficiency | 1.78 GB | Nutrient deficiency | `longitudinal_nutrient_deficiency` |

### Status Updates for Existing Entries
- All datasets physically present in the `Pretraining/` folder: status → `downloaded`
- `rice_plant` (rajkumar898): remains `candidate` (not found locally)
- Duplicate ZIPs flagged: `Plant Disease Expert.zip` and `Plant Leaves for Image Classification.zip` each appear twice

### Proposed Changes

#### [MODIFY] [Dataset.yml](file:///e:/AG_Dataset/AG-Foundational-Model/Dataset.yml)
- Add 8 new dataset entries with `status: downloaded`
- Update all 30 matched entries from `candidate` → `downloaded`
- Add `notes` field on duplicate entries to flag which to keep
- Keep `rice_plant` as `candidate`

---

## Phase 2: Pretraining Strategy

### Image Count Estimate (from audit + Pretraining folder analysis)

| Dataset Category | Est. Images | Used For |
|-----------------|-------------|----------|
| **plantnet_300K** | ~306,000 | Pretraining |
| **GeoPlant** | ~188,000 | Pretraining |
| **Plant Disease Expert** | ~200,000 | Pretraining (one copy) |
| **Plants leafs Dataset** | ~190,000 | Pretraining |
| **PlantVillage** | ~162,000 | Pretraining |
| **PlantifyDr** | ~125,000 | Pretraining |
| **Agriculture-Vision-2021** | ~94,000 (est.) | Pretraining |
| **Plant Leaves for Classification** | ~4,500 | Pretraining (one copy) |
| **Wheat Plant Diseases** | ~16,000 (est.) | Pretraining |
| **Ghana Crop Disease** | ~50,000 (est.) | Pretraining |
| **House Plant Species** | ~30,000 (est.) | Pretraining |
| **CottonWeedDet3** | ~20,000 (est.) | Pretraining |
| **Cotton Plant Disease** | ~12,000 (est.) | Pretraining |
| **V2 Plant Seedlings** | ~11,078 | Pretraining |
| **DeepWeeds** | ~17,509 | Pretraining |
| **Plants Type Datasets** | ~30,000 | Pretraining |
| **Weed Detection Soybean** | ~15,000 (est.) | Pretraining |
| **UNL-CPPD** | ~50,000 (est.) | Pretraining |
| **OPPD** | ~47,000 (est.) | Pretraining |
| Other smaller datasets | ~50,000 (est.) | Pretraining |
| **Total Estimated** | **~1.5–1.9 Million** | |

### Pretraining Data Split Strategy

```
┌─────────────────────────────────────────────────────┐
│                ALL DATASETS (~1.87M images)          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ PRETRAINING POOL (~85% of total, ~1.59M)     │   │
│  │                                               │   │
│  │  ├─ Train Split: 80% (~1.27M images)         │   │
│  │  └─ Val Split:   20% (~318K images)          │   │
│  │     (group-disjoint, existing splitting)      │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ HELD-OUT FOR EVALUATION (~15%, ~280K)         │   │
│  │                                               │   │
│  │  Reserved ENTIRE datasets (source-held-out):  │   │
│  │  • Paddy Doctor (~10K) — disease classif.     │   │
│  │  • Pumpkin Leaf Diseases (~3K) — unseen crop  │   │
│  │  • Pea Plant (~1K) — unseen crop              │   │
│  │  • Indian Medicinal Plants (~6K) — unseen     │   │
│  │  • Toxic Plant Classification (~10K)          │   │
│  │  • Edible Wild Plants (~7K)                   │   │
│  │  • Agriculture Crop Images (~1K) — crop class │   │
│  │  • PlantSeg (~15K est.) — segmentation eval   │   │
│  │  • Longitudinal Nutrient Deficiency           │   │
│  │  • Corn Kernel Counting                       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Evaluation Datasets (Held-Out — Never Seen During Pretraining)

| Dataset | Purpose | Task Type |
|---------|---------|-----------|
| **Paddy Doctor** | Disease classification on unseen crop+source | Classification |
| **Pumpkin Leaf Diseases** | Cross-crop disease transfer | Classification |
| **Pea Plant** | Unseen species transfer | Classification |
| **Indian Medicinal Plants** | Unseen domain transfer | Classification |
| **Toxic Plant Classification** | Domain generalization | Classification |
| **Edible Wild Plants** | Domain generalization | Classification |
| **Agriculture Crop Images** | Few-shot crop classification | Classification |
| **PlantSeg** | Segmentation evaluation | Segmentation |
| **Longitudinal Nutrient Deficiency** | Temporal/nutrient evaluation | Specialized |
| **Corn Kernel Counting** | Counting/regression evaluation | Regression |

### Evaluation Plan

1. **Linear Probing**: Freeze pretrained backbone, train linear classifier on held-out datasets → measures representation quality
2. **Few-Shot Classification**: 1-shot, 5-shot, 10-shot evaluation on held-out classification datasets
3. **Full Fine-Tuning**: Fine-tune entire model on held-out datasets → measures transferability
4. **kNN Evaluation**: Nearest-neighbor classification using frozen features → no training needed
5. **Cross-Domain Generalization**: Train on one disease dataset, test on another (e.g., train on PlantVillage diseases, test on Paddy Doctor)
6. **ImageNet Baseline Comparison**: Compare agricultural pretraining vs. vanilla ImageNet ViT on all downstream tasks

---

## Phase 3: Data Loading Implementation for Pretraining

### Current State
The existing `AgricultureImageDataset` + `get_dataloaders()` in [dataset.py](file:///e:/AG_Dataset/AG-Foundational-Model/src/ag_foundation/data/dataset.py) already supports:
- ZIP archive loading (including nested ZIPs) ✅
- Group-disjoint train/val splitting ✅
- Catalog-based loading via CSV ✅
- Multi-precision (fp32/fp16/bf16) ✅
- DistributedSampler ✅
- Persistent workers + pin_memory ✅

### What Needs to Be Built

#### 1. Multi-Source Pretraining DataLoader
The current dataloader takes a **single `data_root`** (one directory or one ZIP file). For pretraining with ~37 datasets across multiple ZIP files, we need a `MultiSourceDataset` that:
- Accepts a **list of data roots** (multiple ZIPs and directories)
- Builds a unified record list from all sources
- Supports per-source weighting/sampling to avoid domination by large datasets (e.g., plantnet_300K with 306K images vs. Pea Plant with ~1K)
- Tracks source provenance for analysis
- Supports excluding specific datasets (for held-out evaluation)

#### 2. Pretraining Catalog Generator
- Script to scan ALL pretraining ZIPs and generate a master catalog CSV
- Include columns: `path`, `group`, `source_dataset`, `is_pretraining` (vs held-out)
- Skip duplicate ZIPs automatically

#### 3. Optimized DataLoader Settings for RTX 4090 System

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_workers` | 8 | 32 cores available, 8 workers balances CPU/memory |
| `prefetch_factor` | 2 | Standard; keeps pipeline fed without excess memory |
| `pin_memory` | True | RTX 4090 CUDA → pinned memory transfer |
| `persistent_workers` | True | Avoids ZIP handle re-creation overhead |
| `batch_size` | 8 | RTX 4090 24GB VRAM at 224×224 crop with bf16 |
| `precision` | bf16 | RTX 4090 Ada Lovelace has excellent bf16 support |
| `gradient_accumulation_steps` | 4 | Effective batch size = 32 |

### Proposed Changes

#### [NEW] [multi_source_dataset.py](file:///e:/AG_Dataset/AG-Foundational-Model/src/ag_foundation/data/multi_source_dataset.py)
- `MultiSourcePretrainingDataset(Dataset)`: wraps multiple `AgricultureImageDataset` instances
- `get_pretraining_dataloaders()`: factory function for multi-source pretraining
- Source-balanced sampling via `WeightedRandomSampler`
- Exclusion list for held-out evaluation datasets

#### [NEW] [build_pretraining_catalog.py](file:///e:/AG_Dataset/AG-Foundational-Model/scripts/build_pretraining_catalog.py)
- Scans all ZIPs in `Pretraining/` directory
- Generates master catalog CSV with source tracking
- Skips duplicate ZIPs by size comparison
- Marks held-out datasets

#### [NEW] [pretraining_config.yaml](file:///e:/AG_Dataset/AG-Foundational-Model/configs/pretraining_full.yaml)
- Full pretraining configuration optimized for RTX 4090
- References multi-source data roots

#### [MODIFY] [dataset.py](file:///e:/AG_Dataset/AG-Foundational-Model/src/ag_foundation/data/dataset.py)
- No breaking changes to existing code
- Multi-source support is built as a separate module that composes the existing `AgricultureImageDataset`

#### [MODIFY] [__init__.py](file:///e:/AG_Dataset/AG-Foundational-Model/src/ag_foundation/data/__init__.py)
- Export new `MultiSourcePretrainingDataset` and `get_pretraining_dataloaders`

---

## Phase 4: Testing Plan

### New Tests to Write

#### [NEW] [test_multi_source_dataset.py](file:///e:/AG_Dataset/AG-Foundational-Model/tests/test_multi_source_dataset.py)
1. **test_multi_source_discovers_from_multiple_roots** — creates 3 separate ZIP sources, verifies unified discovery
2. **test_multi_source_excludes_held_out_sources** — verifies exclusion list works
3. **test_multi_source_weighted_sampling** — verifies balanced sampling across unequal sources
4. **test_multi_source_getitem_returns_source_info** — each sample includes `source_dataset` field
5. **test_multi_source_train_val_split** — group-disjoint split across multiple sources
6. **test_multi_source_catalog_generation** — catalog CSV has `source_dataset` column
7. **test_multi_source_dataloader_batching** — DataLoader produces correct batch shapes
8. **test_multi_source_handles_empty_zip** — graceful handling of empty archives
9. **test_multi_source_duplicate_detection** — skips ZIPs with identical sizes

### Existing Tests to Validate
- Run entire existing test suite to ensure no regressions

### Integration Test
- **test_real_pretraining_zip_scan** — scan a real small ZIP from `Pretraining/` (e.g., `Pea Plant dataset.zip` at 17 MB) to verify end-to-end loading from actual data

---

## Verification Plan

### Automated Tests
```bash
# Run all tests (existing + new)
python -m pytest tests/ -v

# Run only data loading tests
python -m pytest tests/test_data_loading.py tests/test_multi_source_dataset.py -v

# Run with real pretraining data (integration)
python -m pytest tests/test_multi_source_dataset.py -k "real_pretraining" -v
```

### Manual Verification
1. Generate master catalog from Pretraining directory
2. Verify catalog has correct image counts per source
3. Run a 1-epoch smoke test with multi-source data on CPU
4. Profile memory usage and throughput with `num_workers=8`
5. Verify held-out datasets are excluded from pretraining catalog

---

## Open Questions

> [!IMPORTANT]
> **Environment Setup**: The `venv` environment has CPU-only PyTorch. For data loading development and testing, this is fine (all data loading runs on CPU). However, for actual pretraining runs, you'll need CUDA PyTorch + timm in the same environment. Should I set up a new environment or install timm into `torchenv`?

> [!IMPORTANT]
> **Duplicate ZIPs**: Should I update the plan to physically delete the duplicate ZIPs (`Plant Disease Expert.zip` and `Plant Leaves for Image Classification.zip`) to free up ~16 GB, or just skip them in the catalog?

> [!IMPORTANT]
> **.tar.gz Support**: `Agriculture-Vision-2021.tar.gz` (19.59 GB) is a tar.gz, not a ZIP. The current loader only supports ZIP archives. Should I add tar.gz support to include this dataset in pretraining?

> [!IMPORTANT]
> **GeoPlant Dataset**: At 39.38 GB, GeoPlant is the largest dataset and may contain geospatial data (NPY/CSV format rather than standard images). Should I include it in the pretraining pool or reserve it for later when multispectral support is more mature?
