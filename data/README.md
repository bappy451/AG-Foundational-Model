# Local Data

Generated and downloaded datasets belong under this directory and are ignored by git.

Create the deterministic demo datasets with:

```bash
python -m ag_foundation create-demo-data --output-dir data/demo
```

The command creates:

- `data/demo/rgb/`
- `data/demo/multispectral/`
- `data/demo/dataset_summary.json`
