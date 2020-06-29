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

2. [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) : dataset of chest X-ray images of normal patients and infected with Pneumonia ( bacterial and viral ) and

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

