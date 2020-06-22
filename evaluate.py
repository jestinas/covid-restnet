# -*- coding: utf-8 -*-
# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

import torch
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from PIL import Image

from scripts.datagen import Datagen
from scripts.architectures import Rn50
from scripts.test import test_model, test_image

def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'

if __name__ == "__main__":

    test_file = 'data/3_class_test_df.csv'
    image_file = 'data/raw/normal/normal_001.jpeg'
    num_workers = 2
    batch_size = 1
    input_shape = (3, 256, 256)
    le = LabelEncoder()

    df = pd.read_csv(test_file)

    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),])

    test_set = Datagen(df, l_encoder=le, transforms=test_transforms)
    label_enc = test_set.get_le()
    device = get_device()

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,)

    model = Rn50(device=device, classes=3)
    model.load_state_dict(torch.load('./models/checkpoint.pth')['state_dict'])

    test_model(model=model,
        testloader=test_loader,
        device=device)

    input_image = Image.open(image_file).convert("RGB")
    test_image(model=model,
        image=input_image,
        transform=test_transforms,
        device=device,
        labelencoder=label_enc)
