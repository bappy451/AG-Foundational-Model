# Pretraining Dataset Audit

Generated at: `2026-06-17T04:45:12.174353+00:00`

Pretraining root: `/Users/abedinm/Desktop/AG Dataset/Pretraining`

## Executive Summary

- Local datasets scanned: **13**
- Image-like files counted: **412,454**
- Files counted across archives/directories: **412,466**
- Annotation/metadata files counted: **7**
- Manifest sources listed: **31**
- Manifest sources matched locally: **10**
- Manifest sources not found locally: **21**
- Local storage scanned: **35.86 GB**

## Format Mix

| Extension | Count |
| --- | ---: |
| .jpg | 400,359 |
| .png | 11,079 |
| .jpeg | 1,010 |
| .jfif | 6 |
| .csv | 4 |
| .hdf5 | 2 |
| .json | 2 |
| [no extension] | 2 |
| .txt | 1 |
| .xls | 1 |

## Dataset Inventory

| Local dataset | Matched manifest source | Images | Labels/classes inferred | Type summary | Modality summary |
| --- | --- | ---: | ---: | --- | --- |
| Agriculture crop images.zip | Agriculture crop images | 1,106 | 7 | crop-specific agronomy | standard RGB/image files, sampled RGB imagery |
| Chili Plant Disease Detection.zip | Chili Plant Disease Detection | 200 | 2 | plant disease / stress, crop-specific agronomy, detection / annotation | standard RGB/image files, sampled RGB imagery |
| Chili Plant Disease.zip | Chili Plant Disease Detection | 500 | 5 | plant disease / stress, crop-specific agronomy, detection / annotation | standard RGB/image files, sampled RGB imagery |
| Edible wild plants.zip | Edible wild plants | 6,878 | 63 | plant disease / stress, species / plant identification, weed / crop discrimination | standard RGB/image files, HDF5/NPZ container candidate (not expanded), sampled RGB imagery |
| Indian Medicinal Plant Image Dataset.zip | Indian Medicinal Plant Image Dataset | 5,945 | 40 | species / plant identification, weed / crop discrimination | standard RGB/image files, sampled RGB imagery |
| Plant Disease Detection.zip | Plant Disease Detection | 35,725 | 23 | plant disease / stress, detection / annotation | standard RGB/image files, sampled RGB imagery |
| plantnet_300K.zip | not listed / not matched | 306,146 | 1081 | species / plant identification | standard RGB/image files, sampled RGB imagery |
| Plants Type Datasets.zip | Plants Type Datasets | 30,000 | 30 | species / plant identification, crop-specific agronomy | standard RGB/image files, sampled RGB imagery |
| Rice Leaf Diseases Dataset.zip | Rice Leaf Diseases Dataset | 120 | 3 | plant disease / stress, crop-specific agronomy | standard RGB/image files, sampled RGB imagery |
| Rice Plant diseases dataset.zip | Rice Plant diseases dataset | 4,684 | 3 | plant disease / stress, crop-specific agronomy | standard RGB/image files, sampled RGB imagery |
| rice+leaf+diseases.zip | Rice Leaf Diseases Dataset | 120 | 3 | plant disease / stress, crop-specific agronomy | standard RGB/image files, sampled RGB imagery |
| ssharma2020:Plant-Seedlings-Dataset.zip | V2 Plant Seedlings Dataset | 11,078 | 12 | crop-specific agronomy, seedling / early growth, weed / crop discrimination | standard RGB/image files, sampled RGB imagery |
| Toxic Plant Classification.zip | Toxic Plant Classification | 9,952 | 5 | species / plant identification | standard RGB/image files, sampled RGB imagery |

## What Is Present

| Likely image/task type | Dataset count |
| --- | ---: |
| crop-specific agronomy | 8 |
| plant disease / stress | 7 |
| species / plant identification | 5 |
| detection / annotation | 3 |
| weed / crop discrimination | 3 |
| seedling / early growth | 1 |

## What Is Missing Or Underrepresented

- No TIFF/GeoTIFF or NPY multispectral files were found locally, so the current downloaded corpus appears RGB-only.
- No local dataset was clearly identified as geospatial, satellite, UAV, NIR, or remote-sensing imagery.
- Annotation/metadata files exist, but the corpus is still mostly class-folder style; convert labels into a shared schema before multi-task training.
- Several sources listed in the manifest are not downloaded locally yet, so the corpus is incomplete relative to the current plan.
- The corpus has useful disease and species coverage, but it is likely biased toward leaf close-ups and classification labels.

## Recommended Additions

- Add true multispectral and GeoTIFF pretraining data, ideally Sentinel-2/Landsat/UAV tiles with NIR and red-edge bands, because the current corpus is dominated by standard RGB formats.
- Add georeferenced field-scale imagery with location, date, crop stage, climate, soil, and management metadata so the foundation model learns agronomic context instead of only leaf appearance.
- Standardize the existing annotation files into a single schema and add segmentation/detection coverage for datasets that currently provide only class folders.
- Add external held-out benchmark datasets grouped by geography, crop, farm, and season to measure cross-domain generalization for CVPR or Computers and Electronics in Agriculture quality.
- Run duplicate and near-duplicate detection before pretraining, especially for augmented disease datasets, so SSL does not overfit repeated transformations.
- Track license, citation, source URL, and download date per dataset before publication; this is essential for a top-tier reproducible dataset release.

## Listed Sources Not Found Locally

- PlantVillage Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Plants leafs Dataset: https://www.kaggle.com/datasets/hadyahmed00/plants-leafs-dataset
- PlantifyDr Dataset: https://www.kaggle.com/datasets/lavaman151/plantifydr-dataset
- Plant Disease Expert: https://www.kaggle.com/datasets/sadmansakibmahi/plant-disease-expert
- Wheat Plant Diseases: https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
- Plant Pathogen Dataset: https://www.kaggle.com/datasets/kanishk3813/pathogen-dataset
- Rice Plant Dataset: https://www.kaggle.com/datasets/rajkumar898/rice-plant-dataset
- House Plant Species: https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species
- Plant Leaves for Image Classification: https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification
- Cotton plant disease: https://www.kaggle.com/datasets/dhamur/cotton-plant-disease
- Weed Detection in Soybean Crops: https://www.kaggle.com/datasets/fpeccia/weed-detection-in-soybean-crops, https://data.mendeley.com/datasets/3fmjm7ncc6/2
- Plant Stress Identification (Paddy Leaves): https://www.kaggle.com/datasets/ritikbompilwar/plantstressidentification
- Pea Plant dataset: https://www.kaggle.com/datasets/zunorain/pea-plant-dataset
- plant disease detection - dataset: https://www.kaggle.com/datasets/ironwolf437/plant-disease-detection-dataset
- Ghana Crop Disease Detection Dataset: https://www.kaggle.com/datasets/ohagwucollinspatrick/ghana-crop-disease
- Plant Disease Dataset: https://www.kaggle.com/datasets/rashidthihan/plant-disease-dataset
- CottonWeedDet3: https://www.kaggle.com/datasets/yuzhenlu/cottonweeddet3
- Paddy Doctor: Paddy Disease Classification: https://www.kaggle.com/competitions/paddy-disease-classification/data
- GeoPlant: Spatial Plant Species Prediction Dataset: https://www.kaggle.com/datasets/picekl/geoplant
- Pumpkin Leaf Diseases Dataset From Bangladesh Published: 24 June 2024 | Version 1 | DOI: 10.17632/wtxcw8wpxb.1: https://data.mendeley.com/datasets/wtxcw8wpxb/1
- Rice Leaf Diseases Dataset Published: 19 October 2023 | Version 1 | DOI: 10.17632/dwtn3c6w6p.1: https://data.mendeley.com/datasets/dwtn3c6w6p/1

## Local Datasets Not Matched To The Manifest

- plantnet_300K.zip (306,146 image-like files)

## Re-run Command

```bash
python scripts/analyze_pretraining_dataset.py \
  --pretraining-root ../Pretraining \
  --dataset-list ../Pretraining/Dataset.txt \
  --output-dir reports/pretraining_dataset_audit
```

The scanner does not extract archives. Image counts are based on file extensions, while dimensions, modes, 
and band counts are sampled from a small number of files per dataset.
