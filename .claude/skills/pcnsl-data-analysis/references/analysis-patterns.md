# Analysis Patterns

Self-contained code patterns for common PCNSL dataset analyses.

## 1. Demographics Table

```python
from pcnsl_data_loader import load_aws_clinical_imaging_merged
import pandas as pd

df = load_aws_clinical_imaging_merged()

# Century-correct 2-digit year dates
def fix_date(s):
    dt = pd.to_datetime(s, format='%m/%d/%y', errors='coerce')
    return dt.where(dt.dt.year <= 2025, dt - pd.DateOffset(years=100))

df['BirthDate'] = fix_date(df['BirthDate'])
df['DiagnosisDate'] = fix_date(df['DiagnosisDate'])
df['age_at_dx'] = (df['DiagnosisDate'] - df['BirthDate']).dt.days / 365.25

print(f"Age: {df['age_at_dx'].mean():.1f} +/- {df['age_at_dx'].std():.1f}")
print(df['Sex'].value_counts())
print(df['FirstRace'].value_counts())
```

## 2. Mutation Frequency (Top-N Genes)

```python
from pcnsl_data_loader import load_aws_mutations

muts = load_aws_mutations()
n_patients = muts['patientdurablekey'].nunique()

gene_stats = (muts.groupby('gene')['patientdurablekey'].nunique()
              .sort_values(ascending=False).head(10))
for gene, count in gene_stats.items():
    top_alts = muts[muts['gene'] == gene]['hgvsp'].value_counts().head(3).index.tolist()
    print(f"{gene}: {count}/{n_patients} ({100*count/n_patients:.0f}%) - {', '.join(top_alts)}")
```

## 3. Multi-Sequence Lesion Comparison (FLAIR vs T1Post)

```python
from pcnsl_data_loader import load_all_summary_statistics
import seaborn as sns, matplotlib.pyplot as plt

flair = load_all_summary_statistics(modality="FLAIR", processing=None)
t1 = load_all_summary_statistics(modality="T1Post", processing=None)
flair['modality'] = 'FLAIR'; t1['modality'] = 'T1Post'
combined = pd.concat([flair, t1], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ['total_lesion_volume', 'number_of_lesions', 'enhancement']):
    sns.stripplot(data=combined, x='modality', y=col, ax=ax, alpha=0.4, size=3)
    ax.set_title(col.replace('_', ' ').title())
plt.tight_layout(); plt.savefig('lesion_comparison.png', dpi=300)
```

## 4. Anatomical Distribution with Bootstrapped CI

```python
import numpy as np

regions = ['Frontal', 'Parietal', 'Occipital', 'Temporal', 'CorpusCallosum']
cols = [f'percentage_volume_in_{r}' for r in regions]

means, cis = [], []
for col in cols:
    vals = flair[col].dropna().values
    boot = [np.mean(np.random.choice(vals, len(vals))) for _ in range(1000)]
    means.append(np.mean(vals))
    cis.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

fig, ax = plt.subplots(figsize=(8, 4))
y = range(len(regions))
ax.barh(y, means, xerr=[[m-lo for m,(lo,hi) in zip(means,cis)],
                          [hi-m for m,(lo,hi) in zip(means,cis)]], capsize=3)
ax.set_yticks(y); ax.set_yticklabels(regions)
ax.set_xlabel('% Total Lesion Volume'); plt.tight_layout()
```

## 5. Radiomics Feature Comparison

```python
from pcnsl_data_loader import load_all_radiomics

rad_flair = load_all_radiomics(modality="FLAIR", processing=None)
rad_t1 = load_all_radiomics(modality="T1Post", processing=None)

# Compare first-order features
for feat in ['original_firstorder_Entropy', 'original_firstorder_Kurtosis',
             'original_firstorder_Energy']:
    f_val = pd.to_numeric(rad_flair[feat], errors='coerce')
    t_val = pd.to_numeric(rad_t1[feat], errors='coerce')
    print(f"{feat}: FLAIR={f_val.median():.2f}, T1Post={t_val.median():.2f}")
```

## 6. Scanner Variability

```python
from pcnsl_data_loader import load_aws_dicom_headers

dicom = load_aws_dicom_headers()
flair = dicom[dicom['sequence'] == 'FLAIR']

print("Manufacturers:", flair['Manufacturer'].value_counts().to_dict())
print("Field strengths:", flair['MagneticFieldStrength'].value_counts().to_dict())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
flair['Manufacturer'].value_counts().plot.bar(ax=axes[0], title='Manufacturer')
flair['MagneticFieldStrength'].value_counts().plot.bar(ax=axes[1], title='Field Strength (T)')
plt.tight_layout()
```

## 7. Acquisition Parameter Summary Table

```python
from pcnsl_data_loader import load_aws_dicom_headers
from great_tables import GT

dicom = load_aws_dicom_headers()
params = ['RepetitionTime', 'EchoTime', 'InversionTime', 'FlipAngle', 'SliceThickness']

summary = (dicom.groupby('sequence')[params]
           .agg(lambda x: f"{x.mean():.1f} +/- {x.std():.1f}"))
GT(summary.reset_index()).tab_header(title="MRI Acquisition Parameters")
```

## 8. NIfTI Visualization with nilearn

```python
from pcnsl_data_loader import AWSDataLoader
from nilearn import plotting
import matplotlib.pyplot as plt

loader = AWSDataLoader()
img, mask = loader.load_image_with_mask("sub-0001", processing=None, modality="FLAIR")

# Orthographic view
plotting.plot_anat(img, title="FLAIR - sub-0001", display_mode='ortho')

# Lesion overlay
plotting.plot_roi(mask, bg_img=img, title="Lesion Overlay",
                  alpha=0.5, cmap='hot', display_mode='ortho')

# Mosaic multi-slice
plotting.plot_roi(mask, bg_img=img, display_mode='mosaic',
                  cut_coords=8, cmap='Reds', alpha=0.5)
plt.show()
```

## 9. S3 Direct Access (no auth required)

```python
import boto3, nibabel as nib, tempfile, io, pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "ucsf-pcnsl"

# Load a NIfTI image
def load_nifti_s3(key):
    data = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        f.write(data); tmp = f.name
    img = nib.load(tmp)
    return nib.Nifti1Image(img.get_fdata(), img.affine, img.header)

flair = load_nifti_s3("sub-0001/ses-0001/anat/sub-0001_ses-0001_FLAIR.nii.gz")

# Load a CSV
def load_csv_s3(key):
    data = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return pd.read_csv(io.BytesIO(data))

stats = load_csv_s3("derivatives/pyalfe/sub-0001/ses-0001/statistics/"
                     "lesions_SummaryLesions/sub-0001_ses-0001_FLAIR_SummaryLesions.csv")
```
