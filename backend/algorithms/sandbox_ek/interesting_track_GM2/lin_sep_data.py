'''
Plot linearly separable vs. non-linearly separable data
'''

import numpy as np
from matplotlib import pyplot as plt
import os

pwd = os.getcwd()
    
save_fig = os.path.join(pwd, 'figs', 'model_limitations')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
    
# Make some data
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, 500)
mean2 = [4, 4]
cov2 = [[1, 0], [0, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, 500)

# Plot linearly separable
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
fig.text(0.5, 0.04, 'X', ha='center')
fig.text(0.04, 0.5, 'Y', va='center', rotation='vertical')

ax1.plot(data1[:, 0], data1[:, 1], marker='.', linestyle='None',
                 color='b', label='Class 1')
ax1.plot(data2[:, 0], data2[:, 1], marker='.', linestyle='None',
                 color='r', label='Class 2')
ax1.grid()
ax1.set_title('Linearly Separable Data')

# Plot non-linearly separable
close_msk = np.linalg.norm(data1 - mean1, axis=1) < 1
far_msk = np.linalg.norm(data1 - mean1, axis=1) >= 1
x = data1[:, 0]
y = data1[:, 1]
ax2.plot(x[close_msk], y[close_msk], marker='.', linestyle='None',
                 color='b', label='Class 1')
ax2.plot(x[far_msk], y[far_msk], marker='.', linestyle='None',
                 color='r', label='Class 2')
ax2.grid()
ax2.set_title('Non-linearly Separable Data')
ax2.legend()

# Save 
file_name = 'lin_sep.png'
plt.savefig(os.path.join(save_fig, file_name), dpi=150)