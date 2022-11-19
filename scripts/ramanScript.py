# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from tqdm import tqdm
# %%
class ramanSpectra:
    """
    Stores raman spectra information
    """

    def __init__(self, fileName):
        self.fileName = fileName
        fSplit = fileName.split('/')
        self.file = fSplit[-1]
        self.ramanParams = fSplit[-2]
        self.phenotype = fSplit[-3]
        self.spectra = self.getData()

    def getData(self):
        # Read text file
        with open(self.fileName) as f:
            spectraRaw = f.read()
        spectraRaw = spectraRaw.split('\n')
        spectraRaw = spectraRaw[0:-1]
        spectraRaw = [spec.split('\t') for spec in spectraRaw]

        # Sometimes there is a blank line
        # To avoid this, set value to last value
        spectra = []
        for spec in spectraRaw[1:]:
            for i, val in enumerate(spec):
                if val != '':
                    spec[i] = float(val)
                else:
                    spec[i] = spec[i-1]
            spectra.append(spec)
        return spectra

    def imshow(self):
        pass
# %%
if __name__ == "__main__":
    experiment = 'esamInit'
    spectras = []
    for root, dirs, files in os.walk(f'../data/{experiment}'):
        if len(files)>0:
            for scan in files:
                if scan.endswith('.txt'):
                    fileName = os.path.join(root, scan)
                    spectras.append(ramanSpectra(fileName))

    print('Saving spectra')
    np.save(f'../data/{experiment}.npy', spectras)
# %%
