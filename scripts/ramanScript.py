# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from skimage import exposure

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
        self.spectraRaw, self.spectra = self.getData()

    def getData(self):
        """
        Read text file

        Input:
        Relative file path

        Output:
        Interpolated spectra as list
        """
        # Read text file
        with open(self.fileName) as f:
            spectraRaw = f.read()
        spectraRaw = spectraRaw.split('\n')
        spectraRaw = spectraRaw[0:-1]
        spectraRaw = [spec.split('\t') for spec in spectraRaw]

        # Sometimes there is a blank line
        # To avoid this, we interpolate the value
        spectra, spectraInterp = [], []
        for spec in tqdm(spectraRaw[1:], desc=self.fileName):
            specFloat = []
            for i, val in enumerate(spec):
                if val != '':
                    specFloat.append(float(val))
                else:
                    if i>0:
                        specFloat.append((specFloat[i-1]))
                    else:
                        specFloat.append(np.NaN)
            specFloat = np.array(specFloat)
            specFloatInterp = specFloat.copy()
            # Interpolate values
            nans, idxFun = self.nan_helper(specFloat)
            specFloatInterp[nans] = np.interp(idxFun(nans), idxFun(~nans), specFloat[~nans])
            # Normalize by max
            specFloatNorm = specFloatInterp/np.max(specFloatInterp)

            spectra.append(specFloatInterp)
            spectraInterp.append(specFloatNorm)
        spectra = np.array(spectra)
        spectraInterp = np.array(spectraInterp)
        return spectra, spectraInterp

    def makeImage(self):
        """
        Input: array of spectra
        Output: Square image of summed intensities in the 2800-3000 wavenumber area
        """
        isSquare = lambda x : int(np.sqrt(len(x))) == np.sqrt(len(x))
        itensity = 'NaN'
        if not isSquare(self.spectra):
            raise ValueError('Scan is not a square')
        
        intensities = []

        for spectra in self.spectraRaw:
            intensities.append(sum(spectra[57:115]))
        resize = int(np.sqrt(len(self.spectra)))

        intensityRaw = np.array(intensities).reshape((resize,resize))
        
        # Filter image (optimal so far)
        # Because the data is self contained I won't include the raw image
        p2, p98 = np.percentile(intensityRaw, (2, 98))
        intensityScaled = exposure.rescale_intensity(intensityRaw, in_range=(p2, p98))

        return intensityRaw

    def nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

        Source: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        """

        return np.isnan(y), lambda z: z.nonzero()[0]
# %%
def getRamanData(experiment='esamInit', keep='strict'):
    """
    Loads list of scans specified by class ramanSpectra
    Inputs: 
        - experiment: Experiment to be loaded (should be simply named {experiment}.npy)
        - keep: Flag for whether or not to filter for square images and spectra of length of given axis information
    Outputs:
        - scans: List of scans of class ramanSpectra
        - axisInfo: List that converts indices to wavenumber
    """
    isSquare = lambda x : int(np.sqrt(len(x))) == np.sqrt(len(x))
    isAxis = lambda x : len(x) == len(axisInfo)

    ramanData = f'../data/{experiment}.npy'

    # Load appropriate file
    if os.path.isfile(ramanData):
        scansRaw = list(np.load(ramanData, allow_pickle=True))
    else:
        raise FileNotFoundError(f'Cannot find experiment {experiment}')
    
    # Get axis info
    with open('../data/RamanAxisforMCR.txt') as f:
        axisInfo = f.read()
    
    axisInfo = np.array([float(num) for num in axisInfo.split('\n')[1:-1]])
    
    # Keep only good scans unless otherwise stated
    if keep == 'strict':
        scans = []
        for scan in scansRaw:
            if isSquare(scan.spectra) and isAxis(scan.spectra[0]):
                scans.append(scan)
    else:
        scans = scansRaw

    return scans, axisInfo
## %%
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