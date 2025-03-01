import numpy as np
import pandas as pd
import glob
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import torchmetrics as tm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

def download_data():
    os.makedirs("../data/", exist_ok=True)
    os.system("wget -q -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip")
    os.system("wget -q -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip")
    os.system("wget -q -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/amer_sign2.png")
    os.system("wget -q -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/amer_sign3.png")
    os.system("wget -q -P ../data https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/american_sign_language.PNG")
    os.system("unzip -o -q ../data/sign_mnist_train.csv.zip -d ../data/")
    os.system("unzip -o -q ../data/sign_mnist_test.csv.zip -d ../data/")

class SignLanguageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        label = self.df.iloc[index, 0]
        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

transforms4train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
])

class LightningSignLanguageDataset(LightningDataModule):
    def __init__(self, train_dataset, test_dataset, batch_size=200):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class LightningConvNet(LightningModule):
    def __init__(self, stride=1, dilation=1, n_classes=25, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
                      padding=1, stride=stride, dilation=dilation),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                      padding=1, stride=stride, dilation=dilation),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )
        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, n_classes)
        self.train_roc = tm.AUROC(task="multiclass", num_classes=n_classes)
        self.train_fbeta = tm.FBetaScore(task="multiclass", num_classes=n_classes, beta=1.0, average='macro')
        self.valid_roc = tm.AUROC(task="multiclass", num_classes=n_classes)
        self.valid_fbeta = tm.FBetaScore(task="multiclass", num_classes=n_classes, beta=1.0, average='macro')

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_roc(preds, y)
        self.train_fbeta(preds, y)
        self.log("train_loss", loss)
        self.log("train/roc", self.train_roc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/fbeta", self.train_fbeta, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.valid_roc(preds, y)
        self.valid_fbeta(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/roc", self.valid_roc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/fbeta", self.valid_fbeta, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

def main():
    download_data()
    train_df = pd.read_csv('../data/sign_mnist_train.csv')
    test_df = pd.read_csv('../data/sign_mnist_test.csv')
    train_dataset = SignLanguageDataset(train_df, transform=transforms4train)
    test_dataset = SignLanguageDataset(test_df)
    data_module = LightningSignLanguageDataset(train_dataset, test_dataset, batch_size=200)
    model = LightningConvNet()
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, mode="min", verbose=True)
    trainer = Trainer(max_epochs=20, accelerator="gpu" if torch.cuda.is_available() else "cpu", callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, datamodule=data_module)
    final_checkpoint = "final_model.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    print("Последняя модель сохранена в:", final_checkpoint)
    print("Лучшая модель сохранена в:", checkpoint_callback.best_model_path)
    best_model = LightningConvNet.load_from_checkpoint(final_checkpoint)
    best_model.eval()
    sample, actual_label = test_dataset[1]
    print("Actual label:", actual_label)
    sample = sample.unsqueeze(0)
    with torch.no_grad():
        logits = best_model(sample)
        predicted_label = torch.argmax(logits, dim=1).item()
    print("Predicted label:", predicted_label)

if __name__ == "__main__":
    main()
