# -*- coding: utf-8 -*-
# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from scripts.datagen import Datagen
from scripts.architectures import Rn50
from scripts.train import train_model

def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'

if __name__ == "__main__":

    train_file = 'data/3_class_train_df.csv'
    num_workers = 2
    val_split = 0.2
    batch_size = 32
    num_epochs = 20
    input_shape = (3, 256, 256)
    le = LabelEncoder()

    df = pd.read_csv(train_file)

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),])


    validation_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),])

    train_set = Datagen(df, l_encoder=le, transforms=train_transforms)
    validation_set = Datagen(df, l_encoder=le, transforms=validation_transforms)

    train_idx, val_idx = train_test_split(list(range(len(df))), test_size=val_split)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        # shuffle=True,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,)


    valid_loader = torch.utils.data.DataLoader(
        validation_set,
        # shuffle=False,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,)

    device = get_device()
    net = Rn50(device=device, classes=3)

    dataloaders = {"train": train_loader, "val": valid_loader}
    dataloader_len = {"train": len(train_idx), "val": len(val_idx)}

    criteration = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters())
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(
        model=net,
        device=device,
        criterion=criteration,
        optimizer=optimizer,
        dataloaders=dataloaders,
        dataloader_len=dataloader_len,
        input_shape=input_shape,
        num_epochs=num_epochs,)
        