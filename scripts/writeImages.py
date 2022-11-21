# %%
from ramanScript import ramanSpectra, getRamanData
import matplotlib.pyplot as plt
import numpy as np

from skimage import exposure
from skimage.io import imsave
# %%
scans, axisInfo = getRamanData(experiment='esamInit')
# %%
for scan in scans:
    img = scan.makeImage()
    p2, p98 = np.percentile(img, (2, 98))
    intensityScaled = exposure.rescale_intensity(img, in_range=(p2, p98))
    imgName = f'{scan.file.split(".")[0]}_{scan.ramanParams}_{scan.phenotype}.png'
    print(f'Saving {imgName}')
    imsave(f'../data/images/{imgName}', intensityScaled)
# %%
scans[0].file.split('.')[0]+scans[0].ramanParams+scans[0].phenotype+'.png'
scan = scans[0]