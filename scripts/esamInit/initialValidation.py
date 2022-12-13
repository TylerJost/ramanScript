# %%
from ramanScript import ramanSpectra
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import umap
# %% Load data
experiment = 'esam2'
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
fig, ax = plt.subplots(figsize=(8,8))
esamNegIdx = np.array(phenotypes) == 'esamNeg'
plt.scatter(embedding[esamNegIdx,0], embedding[esamNegIdx,1], s=1.5, c='red', alpha=0.75, label='ESAM (-)')
plt.scatter(embedding[~esamNegIdx,0], embedding[~esamNegIdx,1], s=1.5, c='green', alpha=0.75, label='ESAM (+)')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('MDA-MB-231\nRaman Signal')
lgnd = plt.legend(loc='upper right')
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
plt.savefig('../figures/ramanUMAP.png', dpi=300)
plt.show()

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