# pytorch-covid19

> **This is merely an experiment done on a few images and has not been validated/checked by external health organizations or doctors. No clinical studies have been performed based on the approach which can validate it. This model has been done as a P.O.C. and nothing can be concluded/inferred from this result.**


## Directory structure

    .
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   ├── raw
    │   │   ├── covid
    │   │   ├── normal
    │   │   └── pneumonia
    │   └── raw.csv
    ├── models
    │   └── checkpoint.pth
    ├── reports
    │   ├── architecture.csv
    │   └── figures
    ├── scripts
    │   ├── activationmap.py
    │   ├── architectures.py
    │   ├── datagen.py
    │   ├── __init__.py
    │   ├── test.py
    │   ├── train.py
    │   └── utils.py
    ├── makedataset.py
    └── trainer.py
    ├── evaluate.py
    ├── README.md

# Dataset

1.  [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) : dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.).

2. [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) : dataset of chest X-ray images of normal patients and infected with Pneumonia ( bacterial and viral )

# CNN Model

**base model** : RestNet50, `input_shape=(256, 256)`, `pretrained=True` with modified `fc_layer`

# Scripts

### 1. `makedataset.py`

create csv file with `LABEL` and `IMAGE_PATH`

    path = "./data"
    sample_per_category = 500
    seed = 24
    split_frac = 0.20

output -

    ./data/2_class_test_df.csv
    ./data/2_class_train_df.csv
    ./data/3_class_test_df.csv
    ./data/3_class_train_df.csv
    ./data/raw.csv

---

### 2. `trainer.py`

    train_file = "data/3_class_train_df.csv"
    num_workers = 2
    val_split = 0.2
    batch_size = 32
    num_epochs = 20
    input_shape = (3, 256, 256)
    le = LabelEncoder()

output -

    ./models/checkpoint.pth

---

### 3. `evaluate.py`

    test_file = "data/3_class_test_df.csv"
    image_file = "data/raw/covid/covid_001.jpg"
    num_workers = 2
    batch_size = 1
    input_shape = (256, 256)
    le = LabelEncoder()


- `test_model(model,testloader,device,encoder=None)`

- `test_image(model,image,in_shape,transform,device,labelencoder=None,cam=None)`

---

# Sample Model Results

    [phase: test] total: 240, correct: 112, acc: 46.667

                precision    recall  f1-score   support

            0       0.00      0.00      0.00        38
            1       0.69      0.17      0.28       104
            2       0.44      0.96      0.60        98

        accuracy                           0.47       240
    macro avg       0.38      0.38      0.29       240
    weighted avg       0.48      0.47      0.37       240

    [phase: test] confusion matrix

    Predicted  0   1    2  All
    Actual
    0          0   4   34   38
    1          1  18   85  104
    2          0   4   94   98
    All        1  26  213  240

    {0: 'covid', 1: 'normal', 2: 'pneumonia'}

---

# Sample Image Results


1. Normal X-Ray
![Normal](reports/test_image01.png?style=center "normal xray activation map")

1. Covid19
![Covid19](reports/test_image00.png?style=center "Covid19 xray activation map")

