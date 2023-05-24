'''
Test all the features of CAMPInterestClassifier class in ideal case.
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
import pickle

# Inputs
n_each = 20
pwd = os.getcwd()
plt.rcParams.update({'font.size': 18})

targets = ['Object 1',
           'Object 2',
           'Object 3',
           'Object 4',
           'Object 5',
           'Object 6'
           ]

regions= ['africa',
          'asia',
          'caribbean',
          'central_america',
          'europe',
          'north_america',
          'oceania',
          'south_america'
          ]

# Build pmf p(G|O) assuming uniform prior
p_g_given_o = {0: np.ones(len(regions)) / len(regions),
               1: np.ones(len(regions)) / len(regions),
               2: np.ones(len(regions)) / len(regions),
               3: np.ones(len(regions)) / len(regions),
               4: np.ones(len(regions)) / len(regions),
               5: np.ones(len(regions)) / len(regions)
               }

# User info
user_id = 1
session_id = 1

# Get user inputs for geographic location and object type interest
# All stored as not interested (-1), no preference (0), interested (1)

# Interested in Object 1, otherwise disinterested
u_o = [1, -1, -1, -1, -1, -1]

# No preference in geography
u_g = [0, 0, 0, 0, 0, 0, 0, 0]

# Need this info to plot priors
mean = np.array([0] + u_o + u_g + [0, 0, 0])
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
var = var_int + var_obj + var_geo + var_x
    
# Initialize class
obj = CAMPInterestClassifier(p_g_given_o, user_id, session_id, u_o, u_g,
         regions, targets)

# Now, loop through files
    # Make dummy HMM prediction, p(O|x)
    # Assign geographic location
    # Get derived track features, x
    # Make full regression array, X
    # Assign interest, I
    # Laplace approximation to get p(theta|I,x,G)

# Create ecf and lla objects
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

# Directory where track files exist
direc = '../../../../MOVINT_data/Transfer27102022/'

# Get all files and pick n_each of each model type randomly
random.seed(1)
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
        keep_files_train = model_keep_files
        ct += 1
    else:
        keep_files_train = keep_files_train + model_keep_files

# Initialize csv reader
csv = CSV_IO('HMM')

for file in keep_files_train:
    
    data = csv.extract_data(file)

    ###########################
    #### Define HMM output ####
    ###########################
    
    # Assume HMM predicts correct object as 100% chance
    p_o_given_x = np.zeros(len(targets))
    
    for m in model_names:
        if m in file:
            type_idx = [i for i, s in enumerate(model_names) if m in s][0]
            break
        
    p_o_given_x[type_idx] = 1.0
    
    ##################################
    #### Find geographic location ####
    ##################################
    
    # Read in file
    with open(file) as f:
        lines = f.readlines()
    
    # Separate into separate lines
    ct = 0
    live_data = np.zeros((len(data), 5))
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
            live_data[ct][3] = lat
            live_data[ct][4] = lon
            
            ct += 1
    
    # Assign interest, assume perfect inputs
    if type_idx == 0:
        I = 1
    else:
        I = 0
    
    # Get how long this takes
    start = datetime.now()
    
    obj.update_p_theta(p_o_given_x, live_data, I)
    
# Place to save data
save_data = os.path.join(pwd, 'data')
if not os.path.isdir(save_data):
    os.mkdir(save_data)
    
# Save trained model
data_file = os.path.join(save_data, 'trained_ideal.pkl')
with open(data_file, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Place to save figures
save_fig = os.path.join(pwd, 'figs', 'ideal')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
    
# Now, with trained p(theta), go through and do some inference
# Get all files and pick n_each of each model type randomly
random.seed(100)
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
        keep_files_test = model_keep_files
        ct += 1
    else:
        keep_files_test = keep_files_test + model_keep_files

# Initialize csv reader
csv = CSV_IO('HMM')

stored_interest = np.zeros(shape=(n_each, len(targets)))
counter = np.zeros(len(targets), dtype=int)
for file in keep_files_test:
    
    data = csv.extract_data(file)

    ###########################
    #### Define HMM output ####
    ###########################
    
    # Assume HMM predicts correct object as 100% chance
    p_o_given_x = np.zeros(len(targets))
    
    for m in model_names:
        if m in file:
            type_idx = [i for i, s in enumerate(model_names) if m in s][0]
            break
        
    p_o_given_x[type_idx] = 1.0
    
    ##################################
    #### Find geographic location ####
    ##################################
    
    # Read in file
    with open(file) as f:
        lines = f.readlines()
    
    # Separate into separate lines
    ct = 0
    live_data = np.zeros((len(data), 5))
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
            live_data[ct][3] = lat
            live_data[ct][4] = lon
            
            ct += 1
    
    # Infer interest
    p_I = obj.infer_interest(p_o_given_x, live_data)
    
    # Store
    stored_interest[counter[type_idx]][type_idx] = p_I[1]
    counter[type_idx] += 1
    
# Plot box chart of average interest for each type
file_name = 'Interest_Box.png'
avg_interest = np.mean(stored_interest, axis=0)
fig = plt.figure(figsize = (15, 10))
plt.boxplot(stored_interest)
plt.xlabel('Object Type')
plt.ylabel('P(I=1)')
plt.title('Interest By Object Type in Ideal Case')
plt.grid()
plt.savefig(os.path.join(save_fig, file_name), dpi=150)

# Now go through and see how fast we learn

# Re-initialize class
obj = CAMPInterestClassifier(p_g_given_o, user_id, session_id, u_o, u_g,
         regions, targets)
        
# Sort keep_files_train such that an epoch contains one of each target type
keep_files_tmp = list(np.copy(keep_files_train))
keep_files_epoch = []
model = 1
while len(keep_files_tmp) > 0:
    for (i, f) in enumerate(keep_files_tmp):
        model_str = 'model' + str(model)
        if model_str in f:
            keep_files_epoch = keep_files_epoch + [f]
            model = model + 1
            keep_files_tmp.pop(i)
            break
    if model > 6:
        model = 1
        
# Now loop through keep_files_epoch and find interest probability for 
# keep_files_test after each epoch 

epoch = 0
epochs = np.arange(0, 21)

# Initialize dicts to store
epoch_dict = {ep: {} for ep in epochs}
epoch_dict_lc = {ep: {} for ep in epochs}
epoch_dict_uc = {ep: {} for ep in epochs}

start_flag = True
for file in keep_files_epoch:
    
    # Run inference before learning
    if start_flag:
        
        stored_interest = np.zeros(shape=(n_each, len(targets)))
        counter = np.zeros(len(targets), dtype=int)
        for file_test in keep_files_test:
            
            data = csv.extract_data(file_test)

            ###########################
            #### Define HMM output ####
            ###########################
            
            # Assume HMM predicts correct object as 100% chance
            p_o_given_x = np.zeros(len(targets))
            
            for m in model_names:
                if m in file_test:
                    type_idx = [i for i, s in enumerate(model_names) if m in s][0]
                    break
                
            p_o_given_x[type_idx] = 1.0
            
            ##################################
            #### Find geographic location ####
            ##################################
            
            # Read in file
            with open(file_test) as f_test:
                lines = f_test.readlines()
            
            # Separate into separate lines
            ct = 0
            live_data = np.zeros((len(data), 5))
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
                    live_data[ct][3] = lat
                    live_data[ct][4] = lon
                    
                    ct += 1
            
            # Infer interest
            p_I = obj.infer_interest(p_o_given_x, live_data)
            
            # Store
            stored_interest[counter[type_idx]][type_idx] = p_I[1]
            counter[type_idx] += 1
            
        epoch_dict[epoch] = np.mean(stored_interest, axis=0)
        
        ci_lc = np.array([])
        ci_uc = np.array([])
        for ci_ct in range(len(stored_interest[0])):
            ci_data = stored_interest[:, ci_ct]
            ci = stats.t.interval(alpha=0.95, df=len(ci_data)-1,
                  loc=np.mean(ci_data), scale=stats.sem(ci_data))
            ci_lc = np.append(ci_lc, ci[0])
            ci_uc = np.append(ci_uc, ci[1])
        epoch_dict_lc[epoch] = ci_lc
        epoch_dict_uc[epoch] = ci_uc
        
        epoch += 1
        start_flag = False
    
    data = csv.extract_data(file)

    ###########################
    #### Define HMM output ####
    ###########################
    
    # Assume HMM predicts correct object as 100% chance
    p_o_given_x = np.zeros(len(targets))
    
    for m in model_names:
        if m in file:
            type_idx = [i for i, s in enumerate(model_names) if m in s][0]
            break
        
    p_o_given_x[type_idx] = 1.0
    
    ##################################
    #### Find geographic location ####
    ##################################
    
    # Read in file
    with open(file) as f:
        lines = f.readlines()
    
    # Separate into separate lines
    ct = 0
    live_data = np.zeros((len(data), 5))
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
            live_data[ct][3] = lat
            live_data[ct][4] = lon
            
            ct += 1
    
    # Assign interest, assume perfect inputs
    if type_idx == 0:
        I = 1
    else:
        I = 0
    
    obj.update_p_theta(p_o_given_x, live_data, I)
    
    # Run test tracks through when epoch complete
    if 'model6' in file:
        
        stored_interest = np.zeros(shape=(n_each, len(targets)))
        counter = np.zeros(len(targets), dtype=int)
        for file_test in keep_files_test:
            
            data = csv.extract_data(file_test)

            ###########################
            #### Define HMM output ####
            ###########################
            
            # Assume HMM predicts correct object as 100% chance
            p_o_given_x = np.zeros(len(targets))
            
            for m in model_names:
                if m in file_test:
                    type_idx = [i for i, s in enumerate(model_names) if m in s][0]
                    break
                
            p_o_given_x[type_idx] = 1.0
            
            ##################################
            #### Find geographic location ####
            ##################################
            
            # Read in file
            with open(file_test) as f_test:
                lines = f_test.readlines()
            
            # Separate into separate lines
            ct = 0
            live_data = np.zeros((len(data), 5))
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
                    live_data[ct][3] = lat
                    live_data[ct][4] = lon
                    
                    ct += 1
            
            # Infer interest
            p_I = obj.infer_interest(p_o_given_x, live_data)
            
            # Store
            stored_interest[counter[type_idx]][type_idx] = p_I[1]
            counter[type_idx] += 1
            
        epoch_dict[epoch] = np.mean(stored_interest, axis=0)
        
        ci_lc = np.array([])
        ci_uc = np.array([])
        for ci_ct in range(len(stored_interest[0])):
            ci_data = stored_interest[:, ci_ct]
            ci = stats.t.interval(alpha=0.95, df=len(ci_data)-1,
                  loc=np.mean(ci_data), scale=stats.sem(ci_data))
            ci_lc = np.append(ci_lc, ci[0])
            ci_uc = np.append(ci_uc, ci[1])
        epoch_dict_lc[epoch] = ci_lc
        epoch_dict_uc[epoch] = ci_uc
        epoch += 1
    
# Plot learning
o1 = []
o2 = []
o3 = []
o4 = []
o5 = []
o6 = []
for k in epoch_dict.keys():
    o1.append(epoch_dict[k][0])
    o2.append(epoch_dict[k][1])
    o3.append(epoch_dict[k][2])
    o4.append(epoch_dict[k][3])
    o5.append(epoch_dict[k][4])
    o6.append(epoch_dict[k][5])
    
o1l = []
o2l = []
o3l = []
o4l = []
o5l = []
o6l = []
for k in epoch_dict.keys():
    o1l.append(epoch_dict_lc[k][0])
    o2l.append(epoch_dict_lc[k][1])
    o3l.append(epoch_dict_lc[k][2])
    o4l.append(epoch_dict_lc[k][3])
    o5l.append(epoch_dict_lc[k][4])
    o6l.append(epoch_dict_lc[k][5])
    
o1u = []
o2u = []
o3u = []
o4u = []
o5u = []
o6u = []
for k in epoch_dict.keys():
    o1u.append(epoch_dict_uc[k][0])
    o2u.append(epoch_dict_uc[k][1])
    o3u.append(epoch_dict_uc[k][2])
    o4u.append(epoch_dict_uc[k][3])
    o5u.append(epoch_dict_uc[k][4])
    o6u.append(epoch_dict_uc[k][5])
    
file_name = 'learning_rate.png'
fig = plt.figure(figsize = (15, 10))
plt.plot(epoch_dict.keys(), o1, color='b', linewidth=2, label='Object 1')
plt.fill_between(epoch_dict.keys(), o1l, o1u, color='b', alpha=0.1)
plt.plot(epoch_dict.keys(), o2, color='c', linewidth=2, label='Object 2')
plt.fill_between(epoch_dict.keys(), o2l, o2u, color='c', alpha=0.1)
plt.plot(epoch_dict.keys(), o3, color='g', linewidth=2, label='Object 3')
plt.fill_between(epoch_dict.keys(), o3l, o3u, color='g', alpha=0.1)
plt.plot(epoch_dict.keys(), o4, color='k', linewidth=2, label='Object 4')
plt.fill_between(epoch_dict.keys(), o4l, o4u, color='k', alpha=0.1)
plt.plot(epoch_dict.keys(), o5, color='m', linewidth=2, label='Object 5')
plt.fill_between(epoch_dict.keys(), o5l, o5u, color='m', alpha=0.1)
plt.plot(epoch_dict.keys(), o6, color='r', linewidth=2, label='Object 6')
plt.fill_between(epoch_dict.keys(), o6l, o6u, color='r', alpha=0.1)
plt.xlabel('Epoch')
plt.ylabel('Average P(I=1) Over ' + str(n_each) + ' Tracks')
plt.title('Interest by Epoch in Ideal Case')
plt.grid()
plt.legend()
plt.xticks(epochs)
plt.savefig(os.path.join(save_fig, file_name), dpi=150)
    
    