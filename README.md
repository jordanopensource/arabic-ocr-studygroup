# Arabic OCR
## JOSA Deep Learning Study Group

![Arabic OCR](https://i.imgur.com/bOaKj4B.png)

Welcome to the JOSA Deep Learning study groups first homework! In this homework, we'll be classifying handwritten Arabic letters (all 28 of them) using whatever neural network you want to build.

## Requirements

Python 3 (3.5 or higher recommended)

## Python libraries
* numpy >= 1.13.3
* matplotlib >= 1.5.1
 
## How to start
* Unzip `dataset.zip`
* Read the code inside `starting-code.py`
* Run the python file using `python3 starting-code.py`

## Information about the dataset

> The data-set is composed of 16,800 characters written by 60 participants, the age range is between 19 to 40 years, and 90% of participants are right-hand. Each participant wrote each character (from ’alef’ to ’yeh’) ten times on two forms as shown in Fig. 7(a) & 7(b). The forms were scanned at the resolution of 300 dpi. Each block is segmented automatically using Matlab 2016a to determining the coordinates for each block. The database is partitioned into two sets: a training set (13,440 characters to 480 images per class) and a test set (3,360 characters to 120 images per class). Writers of training set and test set are exclusive. Ordering of including writers to test set are randomized to make sure that writers of test set are not from a single institution (to ensure variability of the test set).


Taken from [Arabic Handwritten Characters Dataset](https://www.kaggle.com/mloey1/ahcd1) on Kaggle. Dataset is licensed under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1.0/).
