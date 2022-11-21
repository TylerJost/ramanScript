# %%
from ramanScript import ramanSpectra
import matplotlib.pyplot as plt
import numpy as np

# from skimage import io, exposure, data
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

p2, p98 = np.percentile(img, (2, 98))
img = exposure.rescale_intensity(img, in_range=(p2, p98))

plt.imshow(img, cmap='gray')
plt.axis('off')
# %%
plt.plot(axisInfo, scan.spectra[80*100+88])
plt.xlabel('Wavenumber (1/cm)')
plt.ylabel('Normalized Intensity')
plt.show()

plt.plot(axisInfo, scan.spectra[0])
plt.xlabel('Wavenumber (1/cm)')
plt.ylabel('Normalized Intensity')
plt.show()
# %%
intensityRange = np.where(np.logical_and(axisInfo<3000, axisInfo>2800))[0]
spectra = scan.spectra[80*100+88]
plt.plot(spectra[intensityRange[0]:intensityRange[-1]])
arrayStart = np.where(axisInfo>=2800)[0][-1]