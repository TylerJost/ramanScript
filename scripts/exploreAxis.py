# %%
from ramanScript import ramanSpectra
import matplotlib.pyplot as plt
import numpy as np

from skimage import io, exposure, data
# %%
isSquare = lambda x : int(np.sqrt(len(x))) == np.sqrt(len(x))
isAxis = lambda x : len(x) == len(axisInfo)
# %%
experiment = 'esamInit'
scansRaw = list(np.load(f'../data/{experiment}.npy', allow_pickle=True))

with open('../data/RamanAxisforMCR.txt') as f:
    axisInfo = f.read()
axisInfo = np.array([float(num) for num in axisInfo.split('\n')[1:-1]])
# %%
scans = []
for scan in scansRaw:
    if isSquare(scan.spectra) and isAxis(scan.spectra[0]):
        scans.append(scan)
# %
img = scans[4].makeImage()

plt.imshow(img, cmap='gray')
plt.axis('off')