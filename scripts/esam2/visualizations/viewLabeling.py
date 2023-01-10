# %%
root = '../../../'
import os
import sys
sys.path.append(os.path.join(root, 'ramanScript'))
import numpy as np
from ramanScript import ramanSpectra
import matplotlib.pyplot as plt

from skimage.color import label2rgb
from skimage.measure import label
# %% Getting data
experiment = 'esam2'
ramanData = np.load(os.path.join(root,'data',experiment,f'{experiment}.npy'), allow_pickle=True)
# %% Plot grid of images
nImgs = len(ramanData)
# rowNum = int(np.sqrt(ramanData))

imgNum = 1
for scan in ramanData:
    img = scan.makeImage()
    labelImg = label(np.reshape(scan.cellSpectra, scan.shape))
    overlaid = label2rgb(labelImg, image=img, bg_label=0)
    plt.subplot(5,4,imgNum)
    plt.imshow(overlaid, cmap='gray')
    plt.axis('off')
    imgNum += 1
plt.savefig(os.path.join(root, 'figures', f'{experiment}Grid.png'), dpi=600)
plt.show()