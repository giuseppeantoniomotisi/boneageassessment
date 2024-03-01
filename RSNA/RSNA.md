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

### How download RSNA dataset?
If you have decided to download the entire RSNA dataset, as shown in figure below, then you should find the following structure in the Download folder:
1. A folder named "boneage-training-dataset".
2. A file named "train.csv" containing the labels for the training files.
3. A folder named "Bone Age Validation Set" containing: "boneage-validation-dataset-1.zip", "boneage-validation-dataset-2.zip", and "Validation Dataset.csv".

<img src="https://github.com/giuseppeantoniomotisi/boneageassessment/raw/main/RSNA/images/download_rsna.png" alt="drawing" width="200"/ align="center">

So the initial structure of RSNA is:
```bash
-- downloads
  |-- boneage-training-dataset
  |-- train.csv 
  |-- Bone\ Age\ Validation\ Set
      |-- boneage-validation-dataset-1.zip
      |-- boneage-validation-dataset-2.zip
      |-- Validation Dataset.csv
```
### Hierarchical structure
Starting from this structure, we want to obtain a more organized hierarchical structure for the new dataset. The new dataset brings several advantages, including:
- No overlapping between the various datasets.
- The possibility to experiment with other subdivisions of the initial dataset.

Files in RSNA directory:
- `__init__.py`

`__init__.py` is used to create folders so that the correct paths used by the application can be constructed. Specifically, it switches from the subdivision proposed by RSNA (training, validation-1, validation-2), to a hierarchical structure of the type:

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

- `merge.py`

`merge.py` code defines functions and a class related to merging CSV files and images for
a dataset project.

- `split.py`

`split.py` facilitates the splitting of a dataset into training, validation, and
test sets. Additionally, it organizes images corresponding to these sets into separate directories.

- `balance.py`

`balancing.py` defines a class BalancingDataset that is designed to balance a dataset by augmenting
the training data with additional samples. The process involves creating a balanced CSV file and,
optionally, generating augmented images.

- `checker.py`

`checker.py` defines a class called Checker which is responsible for checking the contents of
specified folders and CSV files.

- `main.py`

## Usage
1. First download RSNA dataset. Check if all files are in Downloads directory.
2. Use the command cd to move inside the terminal and open ../boneageassessment/RSNA/

```bash
cd ../boneageassessment/RSNA/
```
3. Then use command line:

```bash
python3 main.py
```
4. Now dataset is in your desktop.
