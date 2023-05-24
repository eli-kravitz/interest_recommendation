'''
Find alpha values to scale derived features
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

# Initialize alpha values
a1 = 0
a2 = 0
a3 = 0

# Loop through and reset alpha
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
            
    # Reset alpha if necessary
    a1_test = max(live_data[:, 1])
    a2_test = max(live_data[:, 2])
    a3_test = max(live_data[:, 0])
    
    if a1_test > a1:
        a1 = a1_test
    if a2_test > a2:
        a2 = a2_test
    if a3_test > a3:
        a3 = a3_test
    
print('a1 = ' + str(a1))
print('a2 = ' + str(a2))
print('a3 = ' + str(a3))