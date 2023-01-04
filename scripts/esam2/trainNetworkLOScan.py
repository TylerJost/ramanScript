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
# %% Getting data
experiment = 'esam2'
ramanData = np.load(f'../../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %%
