# %%
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
fileNames = ['esam2leaveOutCellROC.npy', 'esam2leaveOutScanROC.npy', 'esam2testWholeROC.npy']
labels = ['Leave Out Cell', 'Leave Out Scan', 'Shuffle Whole']
resultsDir = '../../../results'

plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

for name, label in zip(fileNames, labels):
    filePath = os.path.join(resultsDir, name)
    fpr, tpr, roc = np.load(filePath, allow_pickle=True)
    plt.plot(fpr, tpr,linewidth=3, label=label)

plt.legend()
plt.title('esam2 ROC')
plt.savefig('../../../figures/esam2ROC.png', dpi=600)
plt.show()
# %%
