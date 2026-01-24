# UCSF PCNSL MRI Dataset Tutorials

This directory contains tutorials and utilities for working with the UCSF Primary CNS Lymphoma (PCNSL) MRI dataset, available on the [AWS Registry of Open Data](https://registry.opendata.aws/).

## Contents

- **[get-to-know-a-dataset-pcnsl.ipynb](get-to-know-a-dataset-pcnsl.ipynb)** - Interactive Jupyter notebook tutorial demonstrating how to access and work with the PCNSL dataset from AWS S3
- **[pcnsl_data_loader.py](pcnsl_data_loader.py)** - Python module with utilities for loading PCNSL neuroimaging data

## Dataset Overview

The PCNSL dataset contains MRI data from patients with primary CNS lymphoma, organized in BIDS (Brain Imaging Data Structure) format:

```
s3://ucsf-pcnsl/
├── sub-XXXX/
│   └── ses-YYYY/
│       └── anat/
│           ├── sub-XXXX_ses-YYYY_T1w.nii.gz
│           ├── sub-XXXX_ses-YYYY_ce-gadolinium_T1w.nii.gz
│           └── sub-XXXX_ses-YYYY_FLAIR.nii.gz
└── derivatives/
    └── pyalfe/
        └── sub-XXXX/
            └── ses-YYYY/
                ├── statistics/          # Lesion measurements (CSV)
                ├── skullstripped/       # Brain-extracted images
                └── masks/               # Lesion segmentation masks
```

### MRI Sequences

- **T1w**: T1-weighted structural image
- **ce-gadolinium_T1w**: Gadolinium-enhanced (post-contrast) T1-weighted image
- **FLAIR**: Fluid-attenuated inversion recovery image

### Derived Data

- **SummaryLesions**: Aggregate lesion statistics per subject
- **IndividualLesions**: Per-lesion measurements
- **radiomics**: PyRadiomics texture features

## Installation

### Using pip

```bash
pip install boto3 nibabel nilearn pandas numpy matplotlib
```

### Using Poetry

```bash
cd tutorials
poetry install
```

### Requirements

- Python >= 3.10
- boto3 >= 1.38.23
- nibabel >= 5.0.0
- nilearn >= 0.10.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

For Jupyter notebook support:
- jupyter >= 1.0
- ipykernel >= 6.0

## Quick Start

### Accessing Data from AWS S3

```python
import boto3
import nibabel as nib
import tempfile
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

# Connect to the public S3 bucket
bucket = "ucsf-pcnsl"
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Load a NIfTI file from S3
def load_nifti_from_s3(bucket, key, s3_client):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read()

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    img = nib.load(tmp_path)
    img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
    Path(tmp_path).unlink()
    return img

# Load a FLAIR image
subject = "sub-0001"
session = "ses-0001"
flair_key = f"{subject}/{session}/anat/{subject}_{session}_FLAIR.nii.gz"
flair_img = load_nifti_from_s3(bucket, flair_key, s3)

print(f"Image shape: {flair_img.shape}")
```

### Visualizing Images

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Display the FLAIR image
plotting.plot_anat(flair_img, title="FLAIR Image", display_mode='ortho')
plt.show()

# Load and overlay lesion mask
mask_key = f"derivatives/pyalfe/{subject}/{session}/masks/lesions_seg_comp/{subject}_{session}_FLAIR_lesions.nii.gz"
lesion_mask = load_nifti_from_s3(bucket, mask_key, s3)

plotting.plot_roi(
    lesion_mask,
    bg_img=flair_img,
    title="FLAIR with Lesion Overlay",
    alpha=0.5,
    cmap='hot'
)
plt.show()
```

### Loading Statistics

```python
import pandas as pd
import io

# Load summary lesion statistics
stats_key = f"derivatives/pyalfe/{subject}/{session}/statistics/SummaryLesions_FLAIR.csv"
response = s3.get_object(Bucket=bucket, Key=stats_key)
summary_stats = pd.read_csv(io.BytesIO(response['Body'].read()))

print(summary_stats)
```

## Running the Tutorial Notebook

1. Install Jupyter and register the kernel:
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=pcnsl-tutorial --display-name="PCNSL Tutorial"
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook get-to-know-a-dataset-pcnsl.ipynb
   ```

3. Select the "PCNSL Tutorial" kernel and run the cells.

## Resources

- **nibabel documentation**: https://nipy.org/nibabel/
- **nilearn documentation**: https://nilearn.github.io/
- **BIDS specification**: https://bids-specification.readthedocs.io/
- **PyRadiomics**: https://pyradiomics.readthedocs.io/
- **AWS Registry of Open Data**: https://registry.opendata.aws/

## License

This dataset is made available under the terms specified in the dataset's LICENSE file on AWS S3.

## Attribution

The tutorial notebook `get-to-know-a-dataset-pcnsl.ipynb` was developed with assistance from Claude (Anthropic). Claude contributed to:
- Notebook structure and dual data source support (local filesystem and AWS S3)
- Helper functions for loading NIfTI images and CSV files
- Visualization code using nilearn and matplotlib
- Statistical analysis and distribution plotting
- Documentation and explanatory markdown cells

## Citation

If you use this dataset in your research, please cite the associated publication (see dataset documentation on AWS for citation details).
