# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import random

from sklearn.model_selection import train_test_split

from skimage import exposure
from skimage.io import imsave
from skimage.draw import polygon2mask

from tqdm import tqdm
# %%
class ramanSpectra:
    """
    Stores raman spectra information. 

    Attributes
    ----
    - fileName: Original file name and relative path
    - experiment: Name of collection experiment
    - file: Initial collection name
    - ramanParams: Parameters used for raman spectroscopy
    - phenotype: Experimental condition
    - spectraRaw: Raw spectra values from raman spectroscopy
    - spectra: Normalized spectra
    - spectraDenoised: Spectra that removes wavenumbers 1725 - 2820 cm^-1 by default
    - shape: Size of image formed of spectra
    - cellSpectra: Reports which scans are of cells and which are of background pixels

    Methods
    ----
    getSpectra(): Mines experiment folder for spectroscopy data
    spectraDenoise(minWN, maxWN): Removes wavenumbers
    makeImage(improveContrast): Makes image from spectra data. Assumes length is a perfect square.

    """

    def __init__(self, fileName):
        self.fileName = fileName
        fSplit = fileName.split('/')
        self.experiment = fSplit[2]
        self.file = fSplit[-1]
        self.ramanParams = fSplit[-2]
        self.phenotype = fSplit[-3]
        self.spectraRaw, self.spectra = self.getSpectra()
        self.spectraDenoised = self.spectraDenoise()

        # Getting the annotation portion
        imgName = f'{self.file.split(".")[0]}_{self.ramanParams}_{self.phenotype}.png'
        imgDir = os.path.join(f'../data/{self.experiment}/images/')
        img = self.makeImage()
        self.shape = img.shape
        if img.size>0:
            if not os.path.exists(imgDir):
                os.makedirs(imgDir)
            imgPath = os.path.join (imgDir, imgName)
            imsave(imgPath, img)
            self.cellSpectra = self.getCellIdx(img)
        else:
            self.cellSpectra = np.array([])


    # Prints information
    def __str__(self):
        """
        Print information about cell as :
        experiment-phenotype-raman parameters-scan name
        """
        return f'{self.experiment}-{self.phenotype}-{self.ramanParams}-{self.file.split(".")[0]}'
    def __repr__(self):
        return self.__str__()

    def getCellIdx(self, img):
        # Get annotation data
        jsonData = f'../data/{self.experiment}/annotations.json'
        if not os.path.exists(jsonData):
            return []
        with open(jsonData) as f:
            coco = json.load(f)
        imgIds = {}
        for cocoImg in coco['images']:
            imgIds[cocoImg['file_name']] = cocoImg['id']
        imgName = f'{self.file.split(".")[0]}_{self.ramanParams}_{self.phenotype}.png'
        if imgName in imgIds.keys():
            annotations = [annotation for annotation in coco['annotations'] if annotation['image_id'] == imgIds[imgName] ]
        else:
            return np.array([])
        fullMask = np.zeros(img.shape)
        c = 1
        for annotation in annotations:
            seg = annotation['segmentation'][0]
            # Converts RLE to polygon coordinates
            seg = np.array(seg).reshape(-1,2)
            # Necessary for mask conversion, otherwise coordinates are wrong
            seg[:,[0,1]] = seg[:,[1,0]]
            mask = polygon2mask(img.shape, seg)
            mask = mask.astype('uint8')
            # Set labels on full mask
            mask[mask == 1] = c
            fullMask += mask
            c += 1
        return fullMask.ravel()

    def getSpectra(self):
        """
        Read text file

        Input:
            - Relative file path

        Output:
            - Interpolated spectra as list
        """
        # Read text file
        with open(self.fileName) as f:
            spectraRaw = f.read()
        spectraRaw = spectraRaw.split('\n')
        spectraRaw = spectraRaw[0:-1]
        spectraRaw = [spec.split('\t') for spec in spectraRaw]

    
        # Sometimes there is a blank line
        # To avoid this, we interpolate the value
        spectra, spectraNorm = [], []
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
            # Interpolate values (NaNs)
            nans, idxFun = self.nan_helper(specFloat)
            specFloatInterp[nans] = np.interp(idxFun(nans), idxFun(~nans), specFloat[~nans])
            # Normalize by max
            specFloatNorm = specFloatInterp/np.max(specFloatInterp)

            spectra.append(specFloatInterp)
            spectraNorm.append(specFloatNorm)
        spectra = np.array(spectra)
        spectraNorm = np.array(spectraNorm)
        return spectra, spectraNorm

    def spectraDenoise(self, minWN = 1727, maxWN = 2820):
        """
        Denoises spectra by removing wavenumbers 1725 - 2820 cm^-1 (default). 
        Max and min refer to absolute wave numbers, not index values. 
        Returned values are the normalized spectra, not raw numbers.
        
        Input:
            - self.spectra: Input spectra numpy array
        Output:
            - spectraDenoised: Numpy array with wavenumbers removed
        """
        with open(f'../data/{self.experiment}/RamanAxisforMCR.txt') as f:
            axisInfo = f.read()

        axisInfo = np.array([float(num) for num in axisInfo.split('\n')[1:-1]])

        idxMin = np.argmin(np.abs(axisInfo-minWN))

        idxMax = np.argmin(np.abs(axisInfo-maxWN))
        
        spectraDenoised = []
        for spectra in self.spectra:
            spectra = np.concatenate([spectra[0:idxMax], spectra[idxMin:]])
            spectraDenoised.append(spectra)
        spectraDenoised = np.array(spectraDenoised)
        
        return spectraDenoised

    def makeImage(self, improveContrast = True):
        """
        Input: array of spectra
        Output: Square image of summed intensities in the 2800-3000 wavenumber area
        """
        isSquare = lambda x : int(np.sqrt(len(x))) == np.sqrt(len(x))
        itensity = 'NaN'
        if not isSquare(self.spectra):
            # raise ValueError('Scan is not a square')
            return np.array([])
        
        intensities = []

        for spectra in self.spectraRaw:
            intensities.append(sum(spectra[57:115]))
        resize = int(np.sqrt(len(self.spectra)))

        intensityRaw = np.array(intensities).reshape((resize,resize))
        
        if improveContrast:
            # Filter image (optimal so far)
            # Because the data is self contained I won't include the raw image
            p2, p98 = np.percentile(intensityRaw, (2, 98))
            intensityRaw = exposure.rescale_intensity(intensityRaw, in_range=(p2, p98))

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
def getRamanData(experiment, dataPath = '../../data', keep='strict'):
    """
    Loads list of scans specified by class ramanSpectra
    Inputs: 
        - experiment: Experiment to be loaded (should be simply named {experiment}.npy)
        - dataPath: Path to plug in experiment data for relative data loading
        - keep: Flag for whether or not to filter for square images and spectra of length of given axis information
    Outputs:
        - scans: List of scans of class ramanSpectra
        - axisInfo: List that converts indices to wavenumber
    """
    isSquare = lambda x : int(np.sqrt(len(x))) == np.sqrt(len(x))
    isAxis = lambda x : len(x) == len(axisInfo)

    ramanData = f'{dataPath}/{experiment}/{experiment}.npy'

    # Load appropriate file
    if os.path.isfile(ramanData):
        scansRaw = list(np.load(ramanData, allow_pickle=True))
    else:
        raise FileNotFoundError(f'Cannot find experiment {experiment}')
    
    # Get axis info
    with open(f'{dataPath}/{experiment}/RamanAxisforMCR.txt') as f:
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
# %%
def loadSpectralData(experiment):
    # Load data
    ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
    scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
    phenotypes, spectra = [], []

    for scan in scans:
        isCell = np.where(scan.cellSpectra>0)[0]
        for cell in isCell:
            spectra.append(scan.spectra[cell])
            phenotypes.append(scan.phenotype)
    return np.array(spectra), phenotypes

def splitDataBalanced(spectra, phenotypes, testSize=0.1, seed=1234, balanced=True):
    # Encode as labels
    uniquePheno = set(phenotypes)
    nPheno = len(uniquePheno)
    phenoDict = {phenotype: n for n, phenotype in zip(range(nPheno), uniquePheno)}
    phenoLabels = [phenoDict[phenotype] for phenotype in phenotypes]

    random.seed(seed)
    l = list(zip(spectra, phenoLabels))
    random.shuffle(l)
    spectra, phenoLabels = zip(*l)
    spectra, phenoLabels = np.array(spectra), np.array(phenoLabels)

    if balanced:
        # Find the least commonly occuring phenotype
        minPheno = len(phenotypes)
        for phenotype in uniquePheno:
            nPheno = phenotypes.count(phenotype)
            if nPheno<minPheno:
                smallestPheno = phenotype
                minPheno = nPheno
        
        labelIdx = []
        for phenoLabel in set(phenoLabels):
            labelIdx += list(np.where(phenoLabels == phenoLabel)[0][0:minPheno])
        spectra = spectra[labelIdx,:]
        phenoLabels = phenoLabels[labelIdx]


    X_train, X_test, y_train, y_test = train_test_split(
        spectra, phenoLabels, test_size=testSize, random_state=seed)
    
    return X_train, X_test, y_train, y_test

def shuffleLists(l, seed=1234):
    random.seed(seed)
    l = list(zip(*l))
    random.shuffle(l)
    return list(zip(*l))
# %%
# spectra, phenotypes = getData('esamInit')
# X_train, X_test, y_train, y_test = splitDataBalanced(spectra, phenotypes)
# %%
if __name__ == "__main__":
    experiment = sys.argv[1]
    # experiment = 'esam2'
    print(f'Getting experiment {experiment}')
    spectras = []
    for root, dirs, files in os.walk(f'../data/{experiment}/scans'):
        if len(files)>0:
            for scan in files:
                if scan.endswith('.txt'):
                    fileName = os.path.join(root, scan)
                    print(f'\tProcessing {scan}\t')
                    spectras.append(ramanSpectra(fileName))
    print('Saving spectra')
    np.save(f'../data/{experiment}/{experiment}.npy', spectras)

# %%
