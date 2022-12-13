# %%
import sys
sys.path.append('../../')
from ramanScript.ramanScript import ramanSpectra, getRamanData

import numpy as np
import matplotlib.pyplot as plt
import umap

from skimage.measure import label
from skimage.color import label2rgb
# %% Load data
scans, spectraAxis = getRamanData('esam2')
# %% Make labeled image
scan = scans[1]

img = scan.makeImage()
mask = scan.cellSpectra.reshape(scan.shape)
labelImage = label(mask)
imgOverlay = label2rgb(labelImage, image=img, bg_label=0, alpha=0.5)
plt.imshow(imgOverlay)
plt.axis('off')
plt.show()
# %% Make UMAP
spectra, phenotypes = [], []
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
plt.scatter(embedding[esamNegIdx,0], embedding[esamNegIdx,1], s=1.5, c='red', alpha=0.25, label='ESAM (-)')
plt.scatter(embedding[~esamNegIdx,0], embedding[~esamNegIdx,1], s=1.5, c='green', alpha=0.25, label='ESAM (+)')

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

plt.show()

# %%
dataPath = '../../data'
experiment = 'esam2'
with open(f'{dataPath}/{experiment}/RamanAxisforMCR.txt') as f:
    axisInfo = f.read()

axisInfo = np.array([float(num) for num in axisInfo.split('\n')[1:-1]])

idx1720 = np.argmin(np.abs(axisInfo-1720))

idx2820 = np.argmin(np.abs(axisInfo-2820))

plt.plot(axisInfo[idx2820:idx1720], spectra[0][idx2820:idx1720])