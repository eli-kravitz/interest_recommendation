'''
Plot scaled derived features
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

pwd = os.getcwd()

targets = ['Object 1',
           'Object 2',
           'Object 3',
           'Object 4',
           'Object 5',
           'Object 6'
           ]

model_names = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']

# Create ecf and lla objects
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

# Directory where track files exist
direc = '../../../../MOVINT_data/Transfer27102022/'

# Get all files
files = glob.glob(direc + '/**/*.tmf', recursive=True)
idx = [i for i, s in enumerate(files) if 'stereo' in s]
keep_files = [e for i, e in enumerate(files) if i in idx]

# Initialize csv reader
csv = CSV_IO('HMM')

max_alt = np.array([])
max_int = np.array([])
max_spd = np.array([])
ot = np.array([])
# Loop through and reset alpha
for file in keep_files:
    
    data = csv.extract_data(file)
    
    for m in model_names:
        if m in file:
            type_idx = [i for i, s in enumerate(model_names) if m in s][0]
            break
    
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
            
    # Get derived track features, x
    fx = np.array([max(live_data[:, 1]), max(live_data[:, 2]), 
                  max(live_data[:, 0])])
    
    # Change scale of these features so they're more similar to one-hot
    fx[0] = fx[0] / 80764
    fx[1] = fx[1] / 20800
    fx[2] = fx[2] / 4152
    
    # Now store everything to plot later
    max_alt = np.append(max_alt, fx[0])
    max_int = np.append(max_int, fx[1])
    max_spd = np.append(max_spd, fx[2])
    ot = np.append(ot, targets[type_idx])
    
save_fig = os.path.join(pwd, 'figs', 'feature_plots')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
    
# Sort everything so object type ordered along x axis
idx = np.argsort(list(ot))
    
# Plot scaled features by type
file_name = 'scaled_max_alt.png'
fig = plt.figure(figsize = (12, 6))
plt.plot(ot[idx], max_alt[idx], marker='.', linewidth=0, markersize=12)
plt.xlabel('Object Type')
plt.ylabel('Scaled Maximum Altitude')
plt.title('Scaled Maximum Altitude By Object Type')
plt.grid()
ax = plt.gca()
ax.set_ylim([0, 1])
plt.savefig(os.path.join(save_fig, file_name), dpi=150)

file_name = 'scaled_max_int.png'
fig = plt.figure(figsize = (12, 6))
plt.plot(ot[idx], max_int[idx], '.', linewidth=0, markersize=12)
plt.xlabel('Object Type')
plt.ylabel('Scaled Maximum Intensity')
plt.title('Scaled Maximum Intensity By Object Type')
plt.grid()
ax = plt.gca()
ax.set_ylim([0, 1])
plt.savefig(os.path.join(save_fig, file_name), dpi=150)

file_name = 'scaled_max_spd.png'
fig = plt.figure(figsize = (12, 6))
plt.plot(ot[idx], max_spd[idx], '.', linewidth=0, markersize=12)
plt.xlabel('Object Type')
plt.ylabel('Scaled Maximum Speed')
plt.title('Scaled Maximum Speed By Object Type')
plt.grid()
ax = plt.gca()
ax.set_ylim([0, 1])
plt.savefig(os.path.join(save_fig, file_name), dpi=150)