# Bone Age Assessment
This project aims to determine through machine learning methods the bone age from digital radiographs of patients aged 0 to 228 months. The challenge and the dataset can be found on the [RSNA website](https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017).

### What is Pediatric Bone Age Assessment?
It is the standard method used from doctors in order to estimate the maturity of a child's skeletal system. It simply consists in taking an X-ray image of the wrist, hand and fingers of the subject. The wrist was chosen because its growth can represent the whole body bone development and the radiation damage to the human body is the least when taking X-rays. The traditional bone age recognition methods makes use of a bone age standard atlas, or alternatively of a scoring method. The atlas method consists in comparing the acquired X-ray image with the standard bone age atlas to infer the bone age. The scoring method requires the doctor to divide the development status of each bone in the hand into different grades, and then evaluate the corresponding grades and scores of different bones. The final score of each X-ray image is the sum of all scores and it can be used to infer the bone age via the median curve of bone maturity score. Of course both these methods are subjected to human error since the evaluation depends completely on the doctors' skills.

### Why it is so important? 
Bone age can reflect the level and maturity of human growth and development. Bone age assessment is widely used in clinical medicine, forensic medicine, sports medicine and other fields. In clinical medicine, skeletal development can lead to the diagnosis of endocrine, developmental and nutritional disorders. Through bone age, it is possible to determine the appropriate time for orthopaedic surgery (like teeth or nasal cavity) and provide a basis for predicting the adult height of the patient. In forensic science, bone age can estimate the real birth date of an individual and provide legal basis for criminal identification. Moreover, the information about bone maturity can guide the selection of athletes more scientifically.

### How ML methods could achieve better results?
As mentioned above, bone age assessment is a technique prone to human error, therefore, with the popularization and development of computer technology, machine learning-based bone age prediction has become a research hotspot in recent years. First of all, machine learning solves the problems of subjective factors linked to the doctors' interpretation, and at the same time reduces the prediction time. From 2007, a lot of ML algorithms for bone age assessment were developed, improving precision over and over and reaching an impressive result of a mean absolute error of 5.46 months. [5]
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/13196.png" width=50% height=50%>

### Challenge results
| Position | Team              | MAD (months) |
| :------: | :---------------: | :----------: |
| 1        | 16 Bit Inc.       | 4.265        |
| 2        | Ian Pan           | 4.350        |
| 3        | F. Kitamura       | 4.382        |
| 4        | H. Thodberg       | 4.505        |
| 5        | MD.ai             | 4.525        |
| 6        | amiper            | 4.527        |
| 7        | rsnahandchallenge | 4.527        |
| 8        | grin              | 4.802        |
| 9        | lbicfigraz        | 4.881        |
| 10       | jcrayan           | 4.907        |

## Usage
This project presents a non-trivial structure. Indeed, within the "baa" application, there are three modules:

1. `RSNA`: for preparing the raw RSNA dataset. For detailed instructions on usage, please refer to the [RSNA module documentation](https://github.com/giuseppeantoniomotisi/boneageassessment/blob/c073291d7423c9fc34a752ee4eb1b61f99ede6ef/baa/RSNA/RSNA.md).
2. `preprocessing`: for preprocessing the images provided by RSNA. To delve deeper into its usage guidelines, consult the [preprocessing module documentation](https://github.com/giuseppeantoniomotisi/boneageassessment/blob/c073291d7423c9fc34a752ee4eb1b61f99ede6ef/baa/preprocessing/preprocessing.md).
3. `age`: for determining bone age through Deep Learning methods. Explore detailed usage instructions in the  [age module documentation](https://github.com/giuseppeantoniomotisi/boneageassessment/blob/c073291d7423c9fc34a752ee4eb1b61f99ede6ef/baa/age/age.md).

Additionally, it features two fundamental functions:

4. `utils.py`: containing important bridging functions between various parts of the code.
5. `predictions.py`: a script capable of making new predictions on a new image using the best model.

Finally, the `boneageassessment.py` application orchestrates the smooth operation of the entire codebase through a straightforward logic. Below is a schematic representation of the project's structure:
```
..boneageassessment
├── LICENSE
├── README.md
├── baa
│   ├── RSNA
│   │   ├── RSNA.md
│   │   ├── __init__.py
│   │   ├── balancing.py
│   │   ├── checker.py
│   │   ├── images
│   │   ├── merge.py
│   │   ├── rsna.py
│   │   ├── split.py
│   │   ├── tests
│   │   │   └── rsna_test.py
│   │   └── tools_rsna.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── preprocessing.md
│   │   ├── preprocessing.py
│   │   ├── tools.py
│   │   └── tests
│   │       └── preprocessing_test.py
│   ├── age
│   │   ├── __init__.py
│   │   ├── age.md
│   │   ├── age.py
│   │   ├── age_macro.json
│   │   ├── model.py
│   │   ├── results
│   │   ├── weights
│   │   └── tests
│   │       └── age_test.py
│   ├── info.csv
│   ├── macro.json
│   ├── prediction.py
│   ├── utils.py
│   └── boneageassessment.py
├── dataset
│   └── IMAGES
│       ├── labels
│       ├── raw
│       └── processed
│           ├── validation
│           ├── test
│           └── train
├── documentation
│   ├── link_to_dataset.txt
│   ├── Bone_Age_Assessment_Report.pdf
│   └── images
└── requirements
    ├── install_reqs.bat
    ├── install_reqs.sh
    └── requirements.txt
```
### `boneageassessment.py`
This script provides a modular system for Bone Age Assessment (BAA). It integrates various modules such as data preprocessing, machine learning, and prediction. Users can configure the behavior of the system using a JSON configuration file.
>[!IMPORTANT]
>Before using this script, ensure that you have already installed the *dataset.zip*(only if RSNA flag is `false`) or *dataset_lite.zip*. 
>In any case, before use the script, please download *weights.zip* or *weights_essential.zip* files at following link to [Google Drive](https://drive.google.com/drive/folders/1zNPHIJymBkvtQQkJgFwa-7K-x1XH2qdp). Once downloaded, verify that both files dataset and weights are present in the 'Downloads' directory, and then please proceed to unzip them.

> [!Tip]
>Please check if there are not subdirectories with same name.
>Example:
>```
>..Downloads
>├── dataset
>│   └── IMAGES
>│       ├── labels
>│       ├── raw
>│       └── processed
>│           ├── validation
>│           ├── test
>│           └── train
>├── weights
>│   ├── best_weights.keras
>│   ├── case_a
>│   ├── case_b
>│   ├── case_d
>│   └── case_e



```json
{
 "RSNA": true, // false
 "Training and testing model": true, // false
 "Path to hyperparameters.json": "../baa/age/macro_age.json",
 "New prediction": true, // false
 "New image name": "image.png",
 "Path to new image": "../"
}
```
After selecting the opportune keys in `macro.json` script, you can easily run the application with the bash command:
```bash
cd <path to boneageassessment>
python3 baa/boneageassessment.py --macro baa/macro.json
```
Check if you are running the code in `boneageassessment` directory.

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/baa_workflow.png">
</p>

## Methods
### Dataset
- Dimensions: 14.2 GB
- All images: 45418 (31382+14036) images
- Training : 27170 (9824) images
- Validation : 2816 images
- Test : 1396 images
- Avaibility: free
>[!NOTE]
>The dataset is available only for academic and research purposes.

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/piechart_dataset.png" width=50% height=50%>
</p>

> [!IMPORTANT]
> This dataset splitting is not the one provided by RSNA. For more information, [RSNA documentation](https://github.com/giuseppeantoniomotisi/boneageassessment/blob/fe9983fb9211d5209c455c57918fba75577d5ce0/documentation/markdown/RSNA.md).

### Dataset
First of all, the dataset is preliminarily analyzed. The preliminary analysis is divided into:

- **Gender assessment**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/gender.png" width=50% height=50%>
</p>

- **Annual bone age distribution over all images and by gender**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/boneage_dist.png" width=50% height=50%>
</p>

- **Training, validation and test dataset distribution**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/321be465407ec1bce45a51403e7b12b13606c927/documentation/images/dataset_dist.png" width=50% height=50%>
</p>

- **Balanced vs unbalanced training dataset**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/321be465407ec1bce45a51403e7b12b13606c927/documentation/images/balanced_ds.png" width=50% height=50%>
</p>

### Augmentation and Preprocessing
As stated above, for the augmentation step we balanced the training dataset. To do this we took the training images and we performed a rotation of a random angle between -20 and 20 degrees around the vertical axis. Each age bin is augmented until is reached the amount of images corresponding to the maximum number of occurrences in the age histogram, so that the new histogram is basically a rectangle.
Once the augmentation step was done we wanted to perform a preprocessing step, that has two main purposes: one is to reduce the dataset size and the second is to clean as much as possible the images (a pseudo-segmentation), making the background black and our subjects (the hands) shiny, i.e. with a higher level of gray. This expedient would create a dataset that would be more readable for our deep learning networks. At first we tried the hand segmentation with a K-means algorithm: we selected three different image regions (background, bone and soft tissue) that should have been discriminated the hand from the background but this method didn't really worked, since it mistook some shady bone regions for background regions. We then switched to the Google mediapipe package [1] to try and detect hands in our images. Whenever the IA algorithm detects one hand, it saves 21 pixels in the image called landmarks. Each of these pixels corresponds to a specific hand zone, so by choosing the top left and the bottom right landmark we can make a bounding box for the hand and crop the image deleting what's outside the box. Then whether the hand is detected or not, the image is further processed with the following operations:
- **Image windowing**: an algorithm finds the leftmost peak (i.e. the darker one) in the image histogram and it puts to zero every pixel with an intensity lower than that corresponding to the bin at the right base of the peak.
- **Histogram equalization**: the image histogram gets equalized.
- **Squaring**: the image is squared, the greater dimension of the image is the chosen side of the square.
- **Resize**: finally the image gets resized to a 399x399 image.

After all these operations, most images get brighter in the hand region and darker in the outer region. The images are ready to be read from some deep learning network to predict the hand bone age.

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/pre_vs_orig.png">
</p>

### Model
Our model for estimating pediatric bone age in months integrates a CNN's feature extraction with linear regression. Initially, rVGG16 combined VGG16 with a regression head, but proved slow and lacked generalization. Dropout layers were added, offering minimal improvement. L2 regularization was then implemented to mitigate overfitting, resulting in enhanced performance. The model's [architecture](https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/model_graph_short_h.png), aimed to balance complexity and computational efficiency, leveraging pre-trained `ImageNet` weights for transfer learning potential.

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/vgg16_regression_model_l2.png" width=50% height=50%>
</p>

## References

[1] Google. Mediapipe | Google for Developers. [Link](https://developers.google.com/mediapipe).

[2] Radiological Society of North America. RSNA Pediatric Bone Age Challenge - Appendix (2017). [Link](https://bonexpert.com/refs/radiol.2018boneAgeChallengeAppendix.pdf).

[3] Radiological Society of North America. RSNA Pediatric Bone Age Challenge (2017). [Link](https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017).

[4] Cole TJ et al. “Ethnic and sex differences in skeletal maturation among the Birth to Twenty cohort in South Africa”. In: Arch Dis Child. 2015 Feb (2014). [doi](10.1136/archdischild-2014-306399).
  
[5] Liu Z-Q et al. “Bone age recognition based on mask R-CNN using xception regression model”. In: Front. Physiol. 14:1062034. (2023). [doi](10.3389/fphys.2023.1062034).

[6] Chao Chen et al. “Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment”. In: IEEE Journal of Biomedical and Health Informatics 26.3 (2022), pp. 1208–1218. [doi](10.1109/JBHI.2021.3095128).
  
[7] Paul Gavrikov. `visualkeras`. 2020. [Link](https://github.com/paulgavrikov/visualkeras).
