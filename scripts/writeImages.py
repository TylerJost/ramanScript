# %%
from ramanScript import ramanSpectra, getRamanData
import matplotlib.pyplot as plt
import numpy as np
import os 
import json

from skimage import exposure
from skimage.io import imsave, imread
# %%
experiment = 'esamInit'
scans, axisInfo = getRamanData(experiment=experiment)
# %%
os.makedirs(f'../data/{experiment}/images/', exist_ok=True)
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
# %% Getting coco annotations
imgPath = os.path.join(f'../data/{experiment}/images/{imgName}')

jsonData = f'../data/{experiment}/annotations.json'
with open(jsonData) as f:
    coco = json.load(f)
# ['licenses', 'info', 'categories', 'images', 'annotations']

imgIds = {}
for cocoImg in coco['images']:
    imgIds[cocoImg['file_name']] = cocoImg['id']
img = imread(imgPath)
annotations = [annotation for annotation in coco['annotations'] if annotation['image_id'] == imgIds[imgName] ]
seg = annotations[1]['segmentation'][0]
seg = np.array(seg).reshape(-1,2)

plt.imshow(img)
plt.plot(seg[:,0], seg[:,1], c = 'red')
# %%
# %%
annotation = annotations[0]
segmentation = [int(pxl) for pxl in annotation['segmentation'][0]]
for px in segmentation:
    img.flat[px] = 0

# %%
height = img.shape[0]
width = img.shape[1]
rleNumbers = [int(num) for num in seg]
mask = rleToMask(rleNumbers, height, width)
plt.imshow(mask)