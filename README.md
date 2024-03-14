# Bone Age Assessment
This project aims to determine through machine learning methods the bone age from digital radiographs of patients aged 0 to 216 months. The challenge and the dataset can be found on the RSNA website at the following link: (https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pediatric-Bone-Age-Challenge-2017).

### What is Pediatric Bone Age Assessment?
It is the standard method used from doctors in order to estimate the maturity of a child's skeletal system. It simply consists in taking an X-ray image of the wrist, hand and fingers of the subject. The wrist was chosen because its growth can represent the whole body bone development and the radiation damage to the human body is the least when taking X-rays. The traditional bone age recognition methods makes use of a bone age standard atlas, or alternatively of a scoring method. The atlas method consists in comparing the acquired X-ray image with the standard bone age atlas to infer the bone age. The scoring method requires the doctor to divide the development status of each bone in the hand into different grades, and then evaluate the corresponding grades and scores of different bones. The final score of each X-ray image is the sum of all scores and it can be used to infer the bone age via the median curve of bone maturity score. Of course both these methods are subjected to human error since the evaluation depends completely on the doctors' skills.

### Why it is so important? 
Bone age can reflect the level and maturity of human growth and development. Bone age assessment is widely used in clinical medicine, forensic medicine, sports medicine and other fields. In clinical medicine, skeletal development can lead to the diagnosis of endocrine, developmental and nutritional disorders. Through bone age, it is possible to determine the appropriate time for orthopaedic surgery (like teeth or nasal cavity) and provide a basis for predicting the adult height of the patient. In forensic science, bone age can estimate the real birth date of an individual and provide legal basis for criminal identification. Moreover, the information about bone maturity can guide the selection of athletes more scientifically.

### How ML methods could achive bettere results?
As mentioned above, bone age assessment is a technique prone to human error, therefore, with the popularization and development of coputer technology, machine learning-based bone age prediction has become a research hotspot in recent years. First of all, machine learning solves the problems of subjective factors linked to the doctors' interpretation, and at the same time reduces the prediction time. From 2007, a lot of ML algorithms for bone age assessment were developed, improving precision over and over and reaching an impressive result of a mean absolute error of 5.46 months. [quotation]

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/13196.png" width=50% height=50%>

### Challange results
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

## Dataset
- Dimensions: 10.93 GB
- All images: 14036 images
- Training : 12,613 images
- Validation : 801 images
- Test : 626 images
- Avaibility: free

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/piechart_rsna_dataset.png" width=50% height=50%>
</p>

### Training dataset
First of all, the training dataset is preliminarily analyzed. The preliminary analysis is divided into:

- **Gender assessment**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/training_gender_counter.png" width=50% height=50%>
</p>

- **Annual bone age distribution over all images and by gender**

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/training_boneage.png" width=50% height=50%>
</p>

 <!--
- **Distribution of z score over all images and by gender**
**z score of boneage**
$$z_{score}(x_i, \bar{x}, \sigma_x) = \frac{x_i-\bar{x}}{\sigma_x}$$
The introduction of the z score is explained in that models perform better when features are normalized to have zero mean and unity standard deviation.

<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/dataset/images/training_zscore.png" width=50% height=50%>
</p>
-->

-**Violin plot**
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/training_violin.png" width=50% height=50%>
</p>

### Validation dataset
In agreement with what was done before:
- **Gender assessment**
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/validation_gender_counter.png" width=50% height=50%>
</p>

- **Annual bone age distribution over all images and by gender**
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/validation_boneage.png" width=50% height=50%>
</p>

<!--
- **Distribution of z score over all images and by gender**
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/dataset/images/validation_zscore.png" width=50% height=50%>
</p>
 -->

- **Violin plot**
<p align="center">
<img align="center" src="https://github.com/giuseppeantoniomotisi/boneageassessment/blob/main/documentation/images/validation_violin.png" width=50% height=50%>
</p>

## Methods

## Results

[! [Build Status](https: //travis-ci.org/joenccann/dillinger.svg?bra nch=main)](https://travis-ci.org/joemccann/dillinger)

## References
