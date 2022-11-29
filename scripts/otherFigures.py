# %%
from ramanScript import ramanSpectra, loadSpectralData, splitDataBalanced

import numpy as np
import matplotlib.pyplot as plt
import random

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
