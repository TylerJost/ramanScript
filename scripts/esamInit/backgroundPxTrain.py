# %% [markdown]
"""
This is a script to train a neural network on background pixels. If the network is
able to predict on background alone it may signal that it is the imaging conditions
that create differences, not spectra identifying differences in cells. 
"""
# %%
from ramanScript import ramanSpectra, shuffleLists
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import json

from skimage.draw import polygon2mask

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %matplotlib inline
# %%
train = 0
# %% Gather data
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %% Set background pixels from COCO file
for scan in tqdm(scans):
    img = scan.makeImage()
    jsonData = f'../data/{scan.experiment}/annotations-background.json'
    with open(jsonData) as f:
        coco = json.load(f)
    imgIds = {}
    for cocoImg in coco['images']:
        imgIds[cocoImg['file_name']] = cocoImg['id']
    imgName = f'{scan.file.split(".")[0]}_{scan.ramanParams}_{scan.phenotype}.png'
    if imgName in imgIds.keys():
        annotations = [annotation for annotation in coco['annotations'] if annotation['image_id'] == imgIds[imgName] ]
    else:
        pass
        # return np.array([])
    fullMask = np.zeros(img.shape)
    c = 1
    for annotation in annotations:
        seg = annotation['segmentation'][0]
        # Converts RLE to polygon coordinates
        seg = np.array(seg).reshape(-1,2)
        # Necessary for mask conversion, otherwise coordinates are wrong
        seg[:,[0,1]] = seg[:,[1,0]]
        mask = polygon2mask(img.shape, seg)
        mask = mask.astype('uint8')
        # Set labels on full mask
        mask[mask == 1] = c
        fullMask += mask
        c += 1
    fullMask = np.zeros(img.shape)
    c = 1
    for annotation in annotations:
        seg = annotation['segmentation'][0]
        # Converts RLE to polygon coordinates
        seg = np.array(seg).reshape(-1,2)
        # Necessary for mask conversion, otherwise coordinates are wrong
        seg[:,[0,1]] = seg[:,[1,0]]
        mask = polygon2mask(img.shape, seg)
        mask = mask.astype('uint8')
        # Set labels on full mask
        mask[mask == 1] = c
        fullMask += mask
        c += 1

    scan.background = fullMask.ravel()
# %%
phenoDict = {'esamNeg': 0, 'esamPos': 1}
spectra, phenotypes = [], []
for scan in scans:
    isBg = scan.background>1
    spectra.append(scan.spectra[isBg])
    phenoNames = [scan.phenotype]*sum(isBg)
    phenoLabels = [phenoDict[name] for name in phenoNames]
    phenotypes.append(phenoLabels)

spectra = np.concatenate(spectra)
phenotypes = np.concatenate(phenotypes)

spectra, phenotypes = shuffleLists([spectra, phenotypes])
spectra = np.array(spectra)
phenotypes = np.array(phenotypes)

# Build train/test set
_, cts = np.unique(phenotypes, return_counts=True)
maxAmt = min(cts)
testSize = 0.1
nTrain = int(maxAmt*(1-testSize))
# Hold out test data
trainIdx, testIdx = [], []
for phenotype in set(phenotypes):
    isPhenotype = np.where(phenotypes == phenotype)[0]
    trainIdx.append(isPhenotype[0:nTrain])
    testIdx.append(isPhenotype[nTrain:])
trainIdx = np.concatenate(trainIdx)
testIdx  = np.concatenate(testIdx)

X_train = spectra[trainIdx,:]
y_train = phenotypes[trainIdx].astype('int')

X_test = spectra[testIdx, :]
y_test = phenotypes[testIdx].astype('int')
# %%
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size=32
print(X_train.shape, y_train.shape)
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
if train:
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
    reportName = f'{experiment}BackgroundPx'
    torch.save(model.state_dict(), f'../models/{reportName}.pth')
# %%
reportName = f'{experiment}BackgroundPx'
model.load_state_dict(torch.load(f'../models/{reportName}.pth', map_location=device))
# %%
probs = []
allLabels = []
scores = []
for i, batch in enumerate(tqdm(test_loader, desc=f"spectra", position=0, leave=True)):

    spectra, labels = tuple(t.to(device) for t in batch)
    # spectra = spectra.to(device)
    # labels = labels.to(device)

    outputs = model(spectra)
    probs.append(outputs.cpu().data.numpy())
    allLabels.append(labels.cpu().data.numpy())
    scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
    
probs = np.concatenate(probs)
allLabels = np.concatenate(allLabels)
scores = np.concatenate(scores)
# TODO: Fix switched labels(?)
allLabelsSwitch = ~allLabels+2
assert set(allLabels) == {0,1}
pred = np.argmax(probs, axis=1)
# %%
fpr, tpr, _ = roc_curve(allLabelsSwitch, scores[:,0])
roc_auc = roc_auc_score(allLabelsSwitch, scores[:,0])

plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.plot(fpr, tpr,linewidth=3)
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'AUC = {roc_auc:0.3f}')

plt.savefig('../figures/rocCurveBackground.png', dpi=600)
plt.show()
np.save('../results/backgroundPxROC.npy', [fpr, tpr, roc_auc])

# %%
