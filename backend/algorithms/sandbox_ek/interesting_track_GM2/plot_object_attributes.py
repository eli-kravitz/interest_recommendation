'''
Plot features by object type
'''

import numpy as np
import scipy.stats as stats
import glob
import pyproj
import random
from backend.io.CSV_Adapter import CSV_IO
from matplotlib import pyplot as plt
import os
from camp_interest_classifier import CAMPInterestClassifier
from datetime import datetime

# Inputs
n_each = 1
pwd = os.getcwd()
plt.rcParams.update({'font.size': 18})

targets = ['Object 1',
           'Object 2',
           'Object 3',
           'Object 4',
           'Object 5',
           'Object 6'
           ]

# Create ecf and lla objects
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

# Directory where track files exist
direc = '../../../../MOVINT_data/Transfer27102022/'

# Get all files and pick n_each of each model type randomly
files = glob.glob(direc + '/**/*.tmf', recursive=True)
idx = [i for i, s in enumerate(files) if 'stereo' in s]
files = [e for i, e in enumerate(files) if i in idx]
model_names = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']
ct = 0
for m in model_names:
    model_idx = [i for i, s in enumerate(files) if m in s]
    model_files = [e for i, e in enumerate(files) if i in model_idx]
    model_keep_idx = random.sample(range(1, len(model_files)), n_each)
    model_keep_files = [e for i, e in enumerate(model_files) if i in model_keep_idx]
    if ct == 0:
        keep_files = model_keep_files
        ct += 1
    else:
        keep_files = keep_files + model_keep_files

# Initialize csv reader
csv = CSV_IO('HMM')

# Store time to process track in Bayes net
all_data = [[]] * len(keep_files)
idx = 0
for file in keep_files:
    
    data = csv.extract_data(file)
    
    # Read in file
    with open(file) as f:
        lines = f.readlines()
    
    # Separate into separate lines
    ct = 0
    live_data = np.zeros((len(data), 3))
    for line in lines:
        
        full_line = False
        
        d = line.strip('\n')
        d = d.split()
        d = np.array(d)
        if d[0] == '#TRK':
            header_trk = d
        if d[0] == '#OBS':
            header_obs = d
        if d[0] == '#TPE':
            header_tpe = d
        if '#' in d[0]:
            continue
        
        if 'TRK' in d[0]:
            
            # Get velocity in ECF
            vel_msk = [i for i, s in enumerate(header_trk) if 'vel_ECF' in s]
            vel = d[vel_msk].astype(float)
            
            
            # Get position in ECF
            pos_msk = [i for i, s in enumerate(header_trk) if 'pos_ECF' in s]
            pos = d[pos_msk].astype(float)
            
        if 'OBS' in d[0]:
            
            # Get intensity
            int_msk = [i for i, s in enumerate(header_obs) if 'intensity_kwsr' in s]
            intensity = d[int_msk].astype(float)
            
        if 'TPE' in d[0]:
            
            full_line = True
        
        # Write to csv when we have all info
        if full_line:
            
            # Calculate speed, altitude, intensity
            speed = np.linalg.norm(vel)
            lon, lat, alt = pyproj.transform(ecef, lla, pos[0],
                                             pos[1], pos[2], radians=False)
            intensity = intensity[0]
            
            # Save data
            live_data[ct][0] = speed
            live_data[ct][1] = alt
            live_data[ct][2] = intensity
            
            ct += 1
            
    # Save all data
    all_data[idx] = live_data
    idx += 1
    
# Now plot everything

save_fig = os.path.join(pwd, 'figs', 'feature_plots')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)

fig = plt.figure(figsize=(15, 10))
colors = ['b', 'c', 'g', 'k', 'm', 'r']
for (i, d) in enumerate(all_data):
    x = np.arange(len(d[:, 0]))
    plt.plot(x, d[:, 0], label=targets[i], color=colors[i], linewidth=2)
plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Speed [m/s]')
plt.title('Speed Profile by Object Type')
file_name = 'speed_obj.png'
plt.savefig(os.path.join(save_fig, file_name), dpi=150, bbox_inches='tight')

fig = plt.figure(figsize=(15, 10))
colors = ['b', 'c', 'g', 'k', 'm', 'r']
for (i, d) in enumerate(all_data):
    x = np.arange(len(d[:, 1]))
    plt.plot(x, d[:, 1], label=targets[i], color=colors[i], linewidth=2)
plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Altitude [m]')
plt.title('Altitude Profile by Object Type')
file_name = 'altitude_obj.png'
plt.savefig(os.path.join(save_fig, file_name), dpi=150, bbox_inches='tight')

fig = plt.figure(figsize=(15, 10))
colors = ['b', 'c', 'g', 'k', 'm', 'r']
for (i, d) in enumerate(all_data):
    x = np.arange(len(d[:, 2]))
    plt.plot(x, d[:, 2], label=targets[i], color=colors[i], linewidth=2)
plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Intensity [kW/sr]')
plt.title('Intensity Profile by Object Type')
file_name = 'intensity_obj.png'
plt.savefig(os.path.join(save_fig, file_name), dpi=150, bbox_inches='tight')

    
    
    
