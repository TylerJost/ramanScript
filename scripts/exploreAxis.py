# %%
from ramanScript import ramanSpectra
import matplotlib.pyplot as plt
import numpy as np
# %%
experiment = 'esamInit'
spectras = np.load(f'../data/{experiment}.npy', allow_pickle=True)
# %%
