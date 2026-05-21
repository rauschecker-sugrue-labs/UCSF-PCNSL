# UCSF PCNSL MRI Dataset Tutorials

This directory contains tutorials and utilities for working with the UCSF Primary CNS Lymphoma (PCNSL) MRI dataset, available on the [AWS Registry of Open Data](https://registry.opendata.aws/).

## Contents

- **[get-to-know-a-dataset-pcnsl.ipynb](get-to-know-a-dataset-pcnsl.ipynb)** - Interactive Jupyter notebook tutorial demonstrating how to access and work with the PCNSL dataset
- **[figures_for_manuscript.ipynb](figures_for_manuscript.ipynb)** - Generates all publication tables and figures
- **[pcnsl_data_loader.py](pcnsl_data_loader.py)** - Python module with utilities for loading PCNSL neuroimaging and clinical data

## Dataset Overview

The PCNSL dataset contains derived MRI data from 150 patients with primary CNS lymphoma. Raw MRI files are not distributed — the dataset consists entirely of processed derivatives:

```
pcnsl-dataset_v1.0/
├── pyalfe/                              # Imaging derivatives
│   └── sub-XXXX/ses-YYYY/
│       ├── dicom_headers/               # Raw DICOM tag JSONs
│       ├── dcm2niix_sidecars/           # dcm2niix BIDS JSONs (acquisition params)
│       ├── masks/
│       │   └── lesions_seg_comp/        # Connected-component labeled lesion masks
│       ├── skullstripped/
│       │   ├── lesions_FLAIR_space/     # 4 sequences registered to FLAIR
│       │   └── lesions_T1Post_space/    # 4 sequences registered to T1Post (ce-gadolinium)
│       └── statistics/
│           ├── lesions_SummaryLesions/  # 2 CSVs (FLAIR + T1Post)
│           ├── lesions_IndividualLesions/
│           └── lesions_radiomics/       # PyRadiomics texture features
└── csvs_for_amazon_anonymized/          # Clinical CSVs
    ├── demographics.csv
    ├── biopsy_and_diagnosis_dates.csv
    ├── diagnosis_history.csv
    ├── medication_list_administered.csv
    ├── medication_list_ordered.csv
    ├── ucsf500_mutations.csv
    ├── data_dictionary_clinical.csv
    └── data_dictionary_imaging.csv
```

The same structure is available on S3 under `s3://ucsf-pcnsl/derivatives/pyalfe/`.

### Image Spaces

The skull-stripped images and lesion masks are registered to one of two reference spaces:

- **FLAIR space**: All sequences registered to the FLAIR image
- **T1Post space**: All sequences registered to the gadolinium-enhanced T1w image

### Derived Data

- **SummaryLesions**: Aggregate lesion statistics per subject
- **IndividualLesions**: Per-lesion measurements
- **radiomics**: PyRadiomics texture features

## Installation

This project uses `uv` for dependency management:

```bash
uv sync                        # Install core dependencies
uv sync --extra dev            # Include Jupyter support
```

Or install with pip:

```bash
pip install boto3 nibabel nilearn pandas numpy matplotlib seaborn great-tables tqdm
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

### Using the Data Loader (local data)

```python
from pcnsl_data_loader import AWSDataLoader

# Load from default pcnsl-dataset_v1.0 directory
loader = AWSDataLoader()

# List subjects
subjects = loader.list_subjects()
print(f"Found {len(subjects)} subjects")

# Load clinical data merged with imaging subject info
df = loader.load_merged_data()

# Load DICOM headers for all subjects
dicom_df = loader.load_dicom_headers()
```

### Using Convenience Functions

```python
from pcnsl_data_loader import (
    load_aws_demographics,
    load_aws_mutations,
    load_aws_dicom_headers,
    load_aws_dicom_geometry,
)

# Each returns a DataFrame with all subjects
demographics = load_aws_demographics()
mutations = load_aws_mutations()
dicom_headers = load_aws_dicom_headers()
geometry = load_aws_dicom_geometry()
```

### Accessing Data from AWS S3

```python
import boto3
import nibabel as nib
import tempfile
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

# Connect to the public S3 bucket (no authentication needed)
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

# Load a skull-stripped FLAIR image
subject = "sub-0001"
session = "ses-0001"
flair_key = f"derivatives/pyalfe/{subject}/{session}/skullstripped/lesions_FLAIR_space/{subject}_{session}_FLAIR_to_FLAIR_skullstripped.nii.gz"
flair_img = load_nifti_from_s3(bucket, flair_key, s3)

print(f"Image shape: {flair_img.shape}")
```

### Visualizing Images

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Display the skull-stripped FLAIR image
plotting.plot_anat(flair_img, title="FLAIR Image", display_mode='ortho')
plt.show()
```

## Running the Notebooks

```bash
jupyter notebook get-to-know-a-dataset-pcnsl.ipynb
jupyter notebook figures_for_manuscript.ipynb
```

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

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
