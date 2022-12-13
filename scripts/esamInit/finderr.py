# %%
from ramanScript import ramanSpectra, loadSpectralData
import numpy as np
# %%
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %%
