# %%
# %%
from ramanScript import ramanSpectra, loadSpectralData, splitDataBalanced
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size=32
# Get data
experiment = 'esamInit'
allSpectra, phenotypes = loadSpectralData(experiment=experiment)
X_train, X_test, y_train, y_test = splitDataBalanced(allSpectra, phenotypes)
print(X_train.shape, y_train.shape)
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
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
model.load_state_dict(torch.load(f'../models/{experiment}Final.pth', map_location=device))
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
allLabels = ~allLabels+2
assert set(allLabels) == {0,1}
pred = np.argmax(probs, axis=1)
# %%
fpr, tpr, _ = roc_curve(allLabels, scores[:,1])
roc_auc = roc_auc_score(allLabels, scores[:,1])

plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.plot(fpr, tpr,linewidth=3)
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'AUC = {roc_auc:0.3f}')

plt.savefig('../figures/rocCurve.png', dpi=600)
plt.show()
# %%
