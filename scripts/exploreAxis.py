# %%
from ramanScript import ramanSpectra
import matplotlib.pyplot as plt
import numpy as np
# %%
experiment = 'esamInit'
scans = list(np.load(f'../data/{experiment}.npy', allow_pickle=True))

with open('../data/RamanAxisforMCR.txt') as f:
    axisInfo = f.read()

axisInfo = [float(num) for num in axisInfo.split('\n')[1:-1]]
# Remove non-correct scan
for i, scan in enumerate(scans):
    if len(scan.spectra[0]) != len(axisInfo):
        scans.pop(i)
    # For now also remove non-square images 
    # TODO: fix this
    if int(np.sqrt(len(scan.spectra))) != np.sqrt(len(scan.spectra)):
        scans.pop(i)
# %% Plot image
scan = scans[3]
for i in range(100):
    plt.plot(scan.spectra[i])
    plt.title(i)
    plt.show()
# %%
intensities = []
scan = scans[4]
for spectra in scan.spectra:
    intensities.append(sum(spectra[480:600]))
resize = int(np.sqrt(len(scan.spectra)))
plt.imshow(np.array(intensities).reshape((resize,resize)), cmap='gray')