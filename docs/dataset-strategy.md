# Dataset Audit And Strategy

## Local Corpus Audit

Audit date: June 15, 2026.

The sibling `Pretraining/` directory currently contains two archives:

| Archive | Loader-visible images | Type | Derived groups |
| --- | ---: | --- | ---: |
| `plantnet_300K.zip` | 306,146 | JPG RGB | 3,243 |
| `ssharma2020:Plant-Seedlings-Dataset.zip` | 11,078 | PNG RGB | 24 |
| Total | 317,224 | RGB only | 3,267 |

PlantNet split:

- train: 243,917 images
- validation: 31,119 images
- test: 31,112 images

The Seedlings archive contains 5,539 images under class directories and another
5,539 under `nonsegmentedv2`, plus macOS metadata that the loader ignores.

## Duplicate Warning

CRC plus uncompressed-size signatures were used as a fast ZIP-index duplicate
proxy:

| Archive | Unique signatures | Files beyond first copy |
| --- | ---: | ---: |
| PlantNet | 306,137 | 9 |
| Seedlings | 5,538 | 5,540 |

CRC is not a cryptographic content hash, but the Seedlings result clearly shows
that the two directory trees are effectively mirrored. Do not use the full
11,078-image catalog for final pretraining without deduplication. The committed
64-image smoke catalog selects only top-level class groups and exists solely for
pipeline verification.

## `Dataset.yml`

The repository `Dataset.yml` is a valid structured registry of 31 candidate
public sources. Only the two
archives above are currently present in `Pretraining/`; a URL in the YAML file
must not be described as part of the trained corpus until it is downloaded,
licensed, audited, and added to a frozen manifest.

The listed sources emphasize:

- leaf disease classification
- plant species classification
- weeds and seedlings
- rice, wheat, cotton, chili, pea, and paddy imagery
- medicinal, edible, toxic, and house plants

## Current Coverage

Strengths:

- large species diversity from PlantNet
- close-range plant and leaf appearance
- seedlings and weeds
- many classification-oriented sources in the acquisition plan

Major gaps:

- no real GeoTIFF or multispectral data in the current `Pretraining/` directory
- limited farm/field-level imagery
- limited geographic and sensor metadata
- little temporal phenotyping
- weak orchard, fruit, counting, and yield coverage
- limited detection and segmentation annotations
- limited severity labels
- no image-text supervision
- no thermal or hyperspectral corpus
- uncertain license compatibility across candidate sources

## Priority Additions

1. Real multispectral and georeferenced agricultural imagery with documented bands
2. Field-condition detection and segmentation datasets
3. Multi-season and multi-location acquisitions
4. Orchard, fruit, counting, and yield-related imagery
5. Stress severity and continuous phenotyping labels
6. Image-text pairs from expert descriptions and extension material
7. External datasets reserved exclusively for transfer evaluation

## Curation Requirements

Before a publication-scale run, create a manifest with:

- source name and canonical URL
- download date and version
- license and redistribution status
- file count, byte size, and format distribution
- channel names and wavelength metadata
- spatial resolution and CRS where applicable
- country, farm, season, crop, and sensor when known
- duplicate and near-duplicate clusters
- train/validation/test assignment
- known label noise and exclusions

## Split Policy

The current loader offers group-disjoint splitting, but the group definition must
be curated. For final evaluation, prefer:

- farm-held-out
- geography-held-out
- season-held-out
- sensor-held-out
- source-dataset-held-out
- crop-held-out

PlantNet's folder structure includes split and species in the group path. It
must be remapped if the scientific split should operate at species, observation,
or geography level.

## Publication Positioning

The defensible contribution is not "many Kaggle datasets combined." It is:

- a provenance-aware, deduplicated, multimodal agricultural corpus
- sensor-adaptive continual pretraining from official ImageNet ViTs
- controlled comparison of reconstruction and self-distillation objectives
- broad transfer under realistic domain shift
- transparent data and evaluation governance
