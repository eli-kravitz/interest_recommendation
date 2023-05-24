'''
Plot priors and posteriors in all tested cases
'''

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import os
import pickle

# Make location to save figs
pwd = os.getcwd()
save_fig = os.path.join(pwd, 'figs', 'posteriors')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
    
# Get priors
u_o = [1, -1, -1, -1, -1, -1]
u_g = [0, 0, 0, 0, 0, 0, 0, 0]
mean1 = np.array([0] + u_o + u_g + [0, 0, 0])
var_int = [3]
var_obj = []
for i in range(len(u_o)):
    if u_o[i] == -1:
        var_obj.append(1)
    elif u_o[i] == 0:
        var_obj.append(3)
    else:
        var_obj.append(1)
var_geo = []
for i in range(len(u_g)):
    if u_g[i] == -1:
        var_geo.append(1)
    elif u_g[i] == 0:
        var_geo.append(3)
    else:
        var_geo.append(1)
var_x = [3, 3, 3]
var1 = var_int + var_obj + var_geo + var_x

u_o = [-1, 1, 1, 1, 1, 1]
u_g = [-1, -1, -1, -1, -1, -1, -1, -1]
mean2 = np.array([0] + u_o + u_g + [0, 0, 0])
var_int = [3]
var_obj = []
for i in range(len(u_o)):
    if u_o[i] == -1:
        var_obj.append(1)
    elif u_o[i] == 0:
        var_obj.append(3)
    else:
        var_obj.append(1)
var_geo = []
for i in range(len(u_g)):
    if u_g[i] == -1:
        var_geo.append(1)
    elif u_g[i] == 0:
        var_geo.append(3)
    else:
        var_geo.append(1)
var_x = [3, 3, 3]
var2 = var_int + var_obj + var_geo + var_x

title_str = ['Intercept',
             'Object 1',
             'Object 2',
             'Object 3',
             'Object 4',
             'Object 5',
             'Object 6',
             'Africa',
             'Asia',
             'Caribbean',
             'Central America',
             'Europe',
             'North America',
             'Oceania',
             'South America',
             'Max Altitude',
             'Max Intensity',
             'Max Speed'
             ]

file_loc = os.path.join(pwd, 'data')
files = ['trained_ideal.pkl', 
         'trained_gamma_uniform.pkl',
         'trained_conf_obj2.pkl',
         'trained_bad_user.pkl']

data = [[]] * len(files)
for (i, f) in enumerate(files):
    with open(os.path.join(file_loc, f), 'rb') as input_file:
        data[i] = pickle.load(input_file)
        
label_str = ['Ideal',
             '$\gamma$ = 0.167',
             'O1 Predicted as O2',
             'Poor User Inputs']
colors = ['g', 'm', 'r', 'c']

for i in range(len(title_str)):
    
    # Make figure
    plt.figure(figsize = (15, 5))
    
    # Find bounds for plotting
    min_x = -9999
    max_x = 9999
    for obj in data:
        mu = obj.p_theta.mean[i]
        sig = obj.p_theta.cov[i][i]
        min_x_test = min(mu - 3 * sig, mean1[i] - 3 * var1[i], mean2[i] - 3 * var2[i])
        max_x_test = max(mu + 3 * sig, mean1[i] + 3 * var1[i], mean2[i] + 3 * var2[i])
        if min_x_test > min_x:
            min_x = min_x_test
        if max_x_test < max_x:
            max_x = max_x_test
        
    # Plot everything
    x = np.arange(min_x, max_x, 0.01)
    if mean1[i] == mean2[i] and var1[i] == var2[i]:
        plt.plot(x, stats.norm.pdf(x, mean1[i], var1[i]), label='Priors',
                 color='k', alpha=0.15, lw=4)
    else:
        plt.plot(x, stats.norm.pdf(x, mean2[i], var2[i]),
                 label='Bad User Prior', color='k', alpha=0.35, lw=4, linestyle='--')
        plt.plot(x, stats.norm.pdf(x, mean1[i], var1[i]),
                 label='Other Priors', color='k', alpha=0.35, lw=4)
    for (j, obj) in enumerate(data):
        mu = obj.p_theta.mean[i]
        sig = obj.p_theta.cov[i][i]
        plt.plot(x, stats.norm.pdf(x, mu, sig), 
                 label='Posterior: ' + label_str[j], color=colors[j], 
                 alpha=0.35, lw=4)
    plt.grid()
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.title(title_str[i] + ' Weight')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    file_name = title_str[i].replace(' ', '_') + '.png'
    plt.savefig(os.path.join(save_fig, file_name), dpi=150, bbox_inches='tight')
