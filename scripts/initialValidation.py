# %%
from ramanScript import ramanSpectra
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import signaltonoise

import umap
# %% Load data
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]

# %%
phenotypes, spectra = [], []

for scan in scans:
    isCell = np.where(scan.cellSpectra>0)[0]
    for cell in isCell:
        spectra.append(scan.spectra[cell])
        phenotypes.append(scan.phenotype)

# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(spectra)
# %%
phenoDict = {'esamPos': 'green', 'esamNeg': 'red'}
phenoColors = [phenoDict[phenotype] for phenotype in phenotypes]
plt.scatter(embedding[:,0], embedding[:,1], s=0.5, c=phenoColors)
# %% Trying to determine noisiness
# def signaltonoise(a, axis=0, ddof=0):
#     m = a.mean(axis)
#     sd = a.std(axis=axis, ddof=ddof)
#     return np.where(sd == 0, 0, m/sd)
# # %%
# scan = scans[0]

# isCell = np.where(scan.cellSpectra>0)[0]
# isBackground = np.where(scan.cellSpectra == 0)[0]

# n = 100
# plt.plot(scan.spectra[isCell[n]], 'b-')
# plt.plot(scan.spectra[isBackground[n]], 'm--')

# # %% 
# mag2db = lambda y : 20*np.log10(y)
# rssq = lambda x : np.sqrt(np.sum(np.abs(x)**2))
# snr = lambda x,y: mag2db(rssq(x)/rssq(y))
# snrCell, snrBack = [], []
# for cell in isCell:
#     snrCell.append(snr(scan.spectra[cell]))

# for bg in isBackground:
#     snrBack.append(snr(scan.spectra[bg]))

# plt.hist(snrCell, label='cell', alpha=0.5)
# plt.hist(snrBack, label='background', alpha=0.5)
# plt.legend()

# %%