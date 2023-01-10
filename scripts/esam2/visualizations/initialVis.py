# %%
import sys
sys.path.append('../../../ramanScript')
from ramanScript import ramanSpectra, getRamanData

import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd

from skimage.measure import label
from skimage.color import label2rgb
# %% Load data
experiment = 'esam2'
ramanData = np.load(f'../../../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %% Make labeled image
scan = scans[12]

img = scan.makeImage()
mask = scan.cellSpectra.reshape(scan.shape)
labelImage = label(mask)
imgOverlay = label2rgb(labelImage, image=img, bg_label=0, alpha=0.5)
plt.imshow(imgOverlay)
plt.axis('off')
plt.show()
# %% Make UMAP
# spectra, phenotypes = [], []
# for scan in scans:
#     isCell = np.where(scan.cellSpectra>0)[0]
#     for cell in isCell:
#         spectra.append(scan.spectra[cell])
#         phenotypes.append(scan.phenotype)
# %%
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(spectra)
# %%
# phenoDict = {'esamPos': 'green', 'esamNeg': 'red'}
# phenoColors = [phenoDict[phenotype] for phenotype in phenotypes]
# fig, ax = plt.subplots(figsize=(8,8))
# esamNegIdx = np.array(phenotypes) == 'esamNeg'
# plt.scatter(embedding[esamNegIdx,0], embedding[esamNegIdx,1], s=1.5, c='red', alpha=0.25, label='ESAM (-)')
# plt.scatter(embedding[~esamNegIdx,0], embedding[~esamNegIdx,1], s=1.5, c='green', alpha=0.25, label='ESAM (+)')

# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.title('MDA-MB-231\nRaman Signal')
# lgnd = plt.legend(loc='upper right')
# for handle in lgnd.legendHandles:
#     handle.set_sizes([50.0])

# plt.show()

# %%
# Spectra denoised
spectra, phenotypes, labelsScan = [], [], []
for scan in scans:
    isCell = np.where(scan.cellSpectra>0)[0]
    for cell in isCell:
        spectra.append(scan.spectraDenoised[cell])
        phenotypes.append(scan.phenotype)
        labelsScan.append(f'{scan.phenotype}-{scan.file}')
# %%
reducer = umap.UMAP()
embedding = reducer.fit_transform(spectra)
# %%
fontSize = 20
phenoDict = {'esamPos': 'green', 'esamNeg': 'red'}
phenoColors = [phenoDict[phenotype] for phenotype in phenotypes]
fig, ax = plt.subplots(figsize=(8,8))
esamNegIdx = np.array(phenotypes) == 'esamNeg'
plt.scatter(embedding[esamNegIdx,0], embedding[esamNegIdx,1], s=1.5, c='red',     alpha=.5, label='ESAM (-)')
plt.scatter(embedding[~esamNegIdx,0], embedding[~esamNegIdx,1], s=1.5, c='green', alpha=.5, label='ESAM (+)')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1', size=fontSize)
plt.ylabel('UMAP 2', size=fontSize)
plt.title('esam2\nRaman Signal', size=fontSize)
lgnd = plt.legend(loc='upper right')
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
plt.savefig('../../../figures/UMAP/esam2UMAPPheno.png', dpi=300)
plt.show()

# %%
umapDf = pd.DataFrame(embedding)
umapDf['label'] = labelsScan

# %%
# Plot so that each scan is a different color
cm = plt.cm.get_cmap('tab20')
colors = {label:color for label, color in zip(np.unique(labelsScan), list(cm.colors))}
grouped = umapDf.groupby('label')
fig, ax = plt.subplots(figsize=(8,8))
for key, group in grouped:
    plt.scatter(group[0], group[1], label=key, color=colors[key], s=2.5, alpha=0.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
# plt.xlim(-2,20)
# plt.axis('off')
plt.xlabel('UMAP 1', size=fontSize)
plt.ylabel('UMAP 2', size=fontSize)
plt.title('UMAP by Scan', size=fontSize)
plt.xticks([])
plt.yticks([])
lgnd = plt.legend(loc='upper right')
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
plt.savefig('../../../figures/UMAP/esam2UMAPScan.png', dpi=300)
plt.show()
# %%
