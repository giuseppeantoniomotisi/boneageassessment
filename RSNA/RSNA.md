# RSNA dataset

**Overview**
- Dimension: 10.93 GB
- Number of images: 14036
- Number of labaled images: 14036
- Method of annotation: manual
- Downloadable ZIP file (https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017)
- Imaging file/structure set format: PNG
- Imaging Modality: X-ray

**Data use agreement/licensing**
- Non-commercial purpose
- References to dataset

All downloadable images were considered for the analysis, and a different split of the dataset was performed because the initial splits did not match the usual percentages stated in the literature. In particular, we decided to split the dataset in 70% for training, 20% for validation and 10% for test.

## Dataset manipulations
This section must be run *only* if you decide to download the raw RSNA dataset. Indeed, these are tedius operations and so we decided to upload our own dataset to speed up analysis.

Files in INFO_RSNA:
- \_\_init__.py

The \_\_init__ file is used to create folders so that the correct paths used by the application can be constructed. Specifically, it switches from the subdivision proposed by RSNA (training, validation-1, validation-2), to a hierarchical structure of the type:

```bash
-- desktop
  | -- boneageassesment
    |-- IMAGES 
      |-- labels
      |-- processed
        |-- all-images
        |-- test
        |-- train
        |-- val
```

- merge.py

- split.py

- balance.py

- checker.py

- main.py
