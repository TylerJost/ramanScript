# %%
import sys
sys.path.append('../../ramanScript')
from ramanScript import ramanSpectra, shuffleLists
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# %%
testSize = 0.15
# %% Getting data
experiment = 'esam2'
ramanData = np.load(f'../../data/{experiment}/{experiment}.npy', allow_pickle=True)
reportName = f'{experiment}WholeDenoisedFinal'
# %%
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
random.seed(1234)
random.shuffle(scans)
spectra, phenotypes = [], []
for scan in scans:
    cellIdx = np.where(scan.cellSpectra>0)[0]
    for cell in cellIdx:
        spectra.append(scan.spectraDenoised[cell,:])
        phenotypes.append(scan.phenotype)

uniquePheno = np.unique(phenotypes)
spectra = np.array(spectra)
unique, cts = np.unique(phenotypes, return_counts=True)
maxCt = int(np.min(cts)*(1-testSize))

spectra, phenotypes = shuffleLists([spectra, phenotypes])
spectra = np.array(spectra)

X_train, y_train, X_test, y_test = [], [], [], []

phenoCount = {pheno: 0 for pheno in uniquePheno}
phenoDict = {phenotype: n for n, phenotype in zip(range(len(uniquePheno)), uniquePheno)}
for signal, phenotype in zip(spectra, phenotypes):
    if phenoCount[phenotype] <= maxCt:
        X_train.append(signal)
        y_train.append(phenoDict[phenotype])

        phenoCount[phenotype] += 1
    else:
        X_test.append(signal)
        y_test.append(phenoDict[phenotype])


X_train = np.array(X_train)
X_test = np.array(X_test)
# %%
batch_size=32
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
# %%
dataiter = iter(train_loader)
spectra, labels = dataiter.next()
print(spectra.shape)
# %%
# Hyper-parameters
num_epochs = 50
# Model parameters
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
    n_classes=2, 
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
allLoss = []
for epoch in tqdm(range(num_epochs), desc=f"epoch", leave=False):
    # train
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f"spectra", position=0, leave=True)):

        spectra, labels = tuple(t.to(device) for t in batch)
        # spectra = spectra.to(device)
        # labels = labels.to(device)

        outputs = model(spectra)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        allLoss.append(float(loss.cpu().detach().numpy()))
    scheduler.step(loss)
    if epoch % 5  == 0:
        torch.save(model.state_dict(), f'../../models/{reportName}.pth')
        np.save(f'../../models/{reportName}.npy', np.array(allLoss))

# %%
# plt.plot(allLoss)

# %%
torch.save(model.state_dict(), f'../../models/{reportName}.pth')
np.save(f'../../models/{reportName}.npy', np.array(allLoss))