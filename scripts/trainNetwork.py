# %% [markdown]
"""
This is the neural network to load and train a neural network on time series classification data
"""
# %%
from ramanScript import ramanSpectra, loadSpectralData, splitDataBalanced

import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from resnet1d import ResNet1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 50
batch_size = 32

# Get data
allSpectra, phenotypes = loadSpectralData(experiment='esamInit')
X_train, X_test, y_train, y_test = splitDataBalanced(allSpectra, phenotypes)
print(X_train.shape, y_train.shape)
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

# %%
dataiter = iter(train_loader)
spectra, labels = dataiter.next()
# %%
# make model
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
kernel_size = 16
stride = 2
n_block = 48
downsample_gap = 6
increasefilter_gap = 12
model = ResNet1D(
    in_channels=1, 
    base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
    kernel_size=kernel_size, 
    stride=stride, 
    groups=32, 
    n_block=n_block, 
    n_classes=4, 
    downsample_gap=downsample_gap, 
    increasefilter_gap=increasefilter_gap, 
    use_do=True)
model.to(device)

# train and test
model.verbose = False
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
criterion = torch.nn.CrossEntropyLoss()
# %%
for epoch in tqdm(range(num_epochs), desc=f"epoch", leave=False):
    # train
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f"spectra", leave=False)):

        spectra, labels = tuple(t.to(device) for t in batch)
        # spectra = spectra.to(device)
        # labels = labels.to(device)

        outputs = model(spectra)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step(loss)