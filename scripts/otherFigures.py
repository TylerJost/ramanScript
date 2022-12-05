# %%
from ramanScript import ramanSpectra, loadSpectralData, splitDataBalanced

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d

from skimage.measure import label
from skimage.color import label2rgb
from skimage import exposure
from skimage.io import imsave
# %%
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %%
scan = scans[0]
img = scan.makeImage()
# img = img.astype('float')
p2, p98 = np.percentile(img, (2, 98))
img = exposure.rescale_intensity(img, in_range=(p2, p98))

plt.imshow(img, cmap='gray')
plt.axis('off')
# plt.savefig('../figures/imgFilterRaw.png', dpi=600)
imsave('../figures/imgFilterRaw.png', img)
# %%
mask = scan.cellSpectra.reshape(scan.shape)
labelImage = label(mask)
imgOverlay = label2rgb(labelImage, image=img, bg_label=0, alpha=0.5)
plt.imshow(imgOverlay)
plt.axis('off')
imsave('../figures/imgFilterOverlay.png', imgOverlay)
# %%
backgroundPxROC = np.load('../results/backgroundPxROC.npy', allow_pickle=True)
leaveOutCellROC = np.load('../results/leaveOutCellROC.npy', allow_pickle=True)

aucBackground = backgroundPxROC[2]
aucCell = leaveOutCellROC[2]
# Interpolate Values
backgroundFunc = interp1d(backgroundPxROC[0], backgroundPxROC[1])
cellFunc = interp1d(leaveOutCellROC[0], leaveOutCellROC[1])

fpr = np.linspace(0,1,1000)
backgroundTpr = backgroundFunc(fpr)
backgroundTpr[0] = 0
backgroundTpr[-1] = 1
cellTpr = cellFunc(fpr)
cellTpr[0] = 0
cellTpr[-1] = 1
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.plot(fpr, backgroundTpr, c='#c77cff', label=f'Background (AUC={aucBackground:0.3f})', linewidth=3)
plt.plot(fpr, cellTpr, c='#00bfc4', label=f'Cell (AUC={aucCell:0.3f})', linewidth=3)
plt.legend(loc='lower right')
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.grid()

plt.savefig('../figures/trainingComparison.png', dpi=300)