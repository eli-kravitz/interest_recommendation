'''
Test all the features of CAMPInterestClassifier class in realistic case. This
is intended to be more of a "deep dive" into how the model works in a 
scenario that is more realistic than anything else that was tested.
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
u_o = [0, 0, 0, 0, 0, 0]

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

freq = [1., 0.5, 0.25]    
strings = ['1', 'p5', 'p25']

# Create ecf and lla objects
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

# Place to save data
save_data = os.path.join(pwd, 'data')
if not os.path.isdir(save_data):
    os.mkdir(save_data)

# Place to save figures
save_fig = os.path.join(pwd, 'figs', 'realistic')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
    
if not os.path.exists(os.path.join(save_data, 'realistic_avg_1.pkl')):
    
    # Get all files and pick n_each of each model type randomly
    direc = '../../../../MOVINT_data/Transfer27102022/'
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
            
    # Speed things up for later by getting all the data for the keep_files_test
    # now instead of in the loop
    
    # Define model names
    model_names = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']
    
    # Initialize csv reader
    csv = CSV_IO('HMM')
    
    test_data = [[]] * len(keep_files_test)
    for (i, file) in enumerate(keep_files_test):
        
        data = csv.extract_data(file)
        
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
                
        test_data[i] = live_data
        
    # Get all training files
    files = glob.glob(direc + '/**/*.tmf', recursive=True)
    idx = [i for i, s in enumerate(files) if 'stereo' in s]
    keep_files_train = [e for i, e in enumerate(files) if i in idx]
            
    # Speed things up for later by getting all the data for the keep_files_train
    # now instead of in the loop
    
    train_data = [[]] * len(keep_files_train)
    for (i, file) in enumerate(keep_files_train):
        
        data = csv.extract_data(file)
        
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
                
        train_data[i] = live_data
    train_idx = list(range(0, len(train_data)))
    
    test_cases = 50
    n_seen = 100
    
    print('Beginning Learning/Inference')
    # Loop through everything
    for (fr_count, fr) in enumerate(freq):
        
        # Initialize matrices to save everything in
        plot_data = dict.fromkeys(targets)
        for key in plot_data.keys():
            plot_data[key] = np.zeros(shape=(test_cases, n_seen), dtype=float)
            
        dcg_data = np.zeros(shape=(test_cases, n_seen), dtype=float)
        dcg_data_rand = np.zeros(shape=(test_cases, n_seen), dtype=float)
        
        for row_count in range(test_cases):
    
            # Initialize class
            obj = CAMPInterestClassifier(p_g_given_o, user_id, session_id, u_o, u_g,
                     regions, targets)
                    
            # Now randomly shuffle training_indices
            random.shuffle(train_idx)
            
            for col_count in range(0, n_seen):
                
                # Get file
                file = keep_files_train[train_idx[col_count]]
                
                # Choose accuracy of classifier
                gamma = random.uniform(0.3, 0.9)
            
                ###########################
                #### Define HMM output ####
                ###########################
                
                # Assume HMM predicts correct object as gamma% chance
                p_o_given_x = np.ones(len(targets)) * (1 - gamma) / (len(targets) - 1)
                
                for m in model_names:
                    if m in file:
                        type_idx = [i for i, s in enumerate(model_names) if m in s][0]
                        break
                    
                p_o_given_x[type_idx] = gamma
                
                ##############################
                #### Get stored live data ####
                ##############################
                live_data = train_data[train_idx[col_count]]
                
                # Assign interest if we want to, assume perfect inputs
                label_rand = random.uniform(0.0, 0.99)
                if fr > label_rand:
                    if type_idx == 0:
                        I = 1
                    else:
                        I = 0
                    obj.update_p_theta(p_o_given_x, live_data, I)
                
                # Run test tracks through for inference each time
                stored_interest = np.zeros(shape=(n_each, len(targets)), dtype=float)
                type_counter = np.zeros(len(targets), dtype=int)
                for (ft_count, file_test) in enumerate(keep_files_test):
                    
                    # Choose accuracy of classifier
                    gamma = random.uniform(0.3, 0.9)
        
                    ###########################
                    #### Define HMM output ####
                    ###########################
                    
                    # Assume HMM predicts correct object as gamma% chance
                    p_o_given_x = np.ones(len(targets)) * (1 - gamma) / (len(targets) - 1)
                    
                    for m in model_names:
                        if m in file_test:
                            type_idx = [i for i, s in enumerate(model_names) if m in s][0]
                            break
                        
                    p_o_given_x[type_idx] = gamma
                    
                    ##############################
                    #### Get stored live data ####
                    ##############################
                    live_data = test_data[ft_count]
                    
                    # Infer interest
                    p_I = obj.infer_interest(p_o_given_x, live_data)
                    
                    # Store
                    stored_interest[type_counter[type_idx]][type_idx] = p_I[1]
                    type_counter[type_idx] += 1
                    
                # Store the averages
                to_save = np.mean(stored_interest, axis=0)
                for (save_ct, s) in enumerate(to_save):
                    plot_data[targets[save_ct]][row_count][col_count] = s
                    
                # Store discounted cumulative gain
                reshaped = np.ravel(stored_interest, order='F')
                rel = -1 * np.ones(len(keep_files_test))
                rel[0:20] = 1
                
                # Sort according to p(I=1)
                sort = np.argsort(reshaped)
                sort = np.flip(sort)
                reshaped = reshaped[sort]
                rel = rel[sort]
                dcg = 0.0
                for s in range(1, len(keep_files_test) + 1):
                    dcg = dcg + rel[s-1] / np.log2(s + 1)
                    
                dcg_data[row_count][col_count] = dcg
                
                # Store random discounted cumulative gain
                reshaped = np.ravel(stored_interest, order='F')
                rel = -1 * np.ones(len(keep_files_test))
                rel[0:20] = 1
                
                # Sort randomly
                sort = list(range(0, len(keep_files_test)))
                random.shuffle(sort)
                reshaped = reshaped[sort]
                rel = rel[sort]
                dcg = 0.0
                for s in range(1, len(keep_files_test) + 1):
                    dcg = dcg + rel[s-1] / np.log2(s + 1)
                    
                dcg_data_rand[row_count][col_count] = dcg
                
        # Save data
        data_file = os.path.join(save_data, 'realistic_avg_' + strings[fr_count] + '.pkl')
        with open(data_file, 'wb') as handle:
            pickle.dump(plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        data_file = os.path.join(save_data, 'realistic_dcg_' + strings[fr_count] + '.pkl')
        with open(data_file, 'wb') as handle:
            pickle.dump(dcg_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        data_file = os.path.join(save_data, 'realistic_dcg_rand_' + strings[fr_count] + '.pkl')
        with open(data_file, 'wb') as handle:
            pickle.dump(dcg_data_rand, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
# Probability of interest plots
for (ct, s) in enumerate(strings):
    
    with open(os.path.join(save_data, 'realistic_avg_' + s + '.pkl'), 'rb') as input_file:
        data = pickle.load(input_file)
        
    plt.figure(figsize=(12, 5))
    x = list(range(1, len(data['Object 1'][0]) + 1))
    for key in data.keys():
        data_single = data[key]
        for (i, d) in enumerate(data_single):
            if i == 0:
                if key == 'Object 1':
                    plt.plot(x, d, color='tab:blue', linewidth=2, label='Object 1', alpha=0.3)
                elif key == 'Object 2':
                    plt.plot(x, d, color='gray', linewidth=2, label='Other Objects', alpha=0.3)
                else:
                    plt.plot(x, d, color='gray', linewidth=2, alpha=0.3)
            else:
                if key == 'Object 1':
                    plt.plot(x, d, color='tab:blue', linewidth=2, alpha=0.3)
                else:
                    plt.plot(x, d, color='gray', linewidth=2, alpha=0.3)
    plt.grid()
    plt.ylim([0, 1])
    plt.title('Average Interest Over 20 Tracks of Each Object Type, f=' + str(freq[ct]))
    plt.xlabel('Number of Tracks Seen')
    plt.ylabel('P(I=1)')
    plt.legend(loc='upper right')
    
    # Save
    file_name = 'pI_avg_' + strings[ct] + '.png'
    plt.savefig(os.path.join(save_fig, file_name), dpi=150)
    
# DCG plot

# Calculate Ideal DCG
rel = -1 * np.ones(120)
rel[0:20] = 1
dcg_ideal = 0.0
for s in range(1, 121):
    dcg_ideal = dcg_ideal + rel[s-1] / np.log2(s + 1)

for (ct, s) in enumerate(strings):
    
    plt.figure(figsize=(12, 5))
    
    # With model sorting
    with open(os.path.join(save_data, 'realistic_dcg_' + s + '.pkl'), 'rb') as input_file:
        data = pickle.load(input_file)
        
    x = list(range(1, len(data[0]) + 1))
    for (i, d) in enumerate(data):
        if i == 0:
            plt.plot(x, d, color='tab:blue', linewidth=2, label='Model Sorting', alpha=0.3)
        else:
            plt.plot(x, d, color='tab:blue', linewidth=2, alpha=0.3)
                
    # With random sorting
    with open(os.path.join(save_data, 'realistic_dcg_rand_' + s + '.pkl'), 'rb') as input_file:
        data = pickle.load(input_file)
         
    for (i, d) in enumerate(data):
        if i == 0:
            plt.plot(x, d, color='gray', linewidth=2, label='Random Sorting', alpha=0.3)
        else:
            plt.plot(x, d, color='gray', linewidth=2, alpha=0.3)
             
    # Plot line with ideal DCG
    plt.axhline(y=dcg_ideal, color ='k', linestyle='--', linewidth=2, label='Ideal DCG')
     
    plt.grid()
    plt.title('Discounted Cumulative Gain, f=' + str(freq[ct]))
    plt.xlabel('Number of Tracks Seen')
    plt.ylabel('DCG')
    plt.legend(loc='lower right')
    
    # Save
    file_name = 'dcg_' + strings[ct] + '.png'
    plt.savefig(os.path.join(save_fig, file_name), dpi=150)
        
# Compare DCG after 100 tracks seen for each case
indices = [24, 49, 99]
bin_size = 0.5
for idx in indices:
    
    hist_data = [[]] * 4
    
    for (ct, s) in enumerate(strings):
        
        # With model sorting
        with open(os.path.join(save_data, 'realistic_dcg_' + s + '.pkl'), 'rb') as input_file:
            data = pickle.load(input_file)
            
        hist_data[ct] = data[:, idx]
                    
    # With random sorting
    with open(os.path.join(save_data, 'realistic_dcg_rand_1.pkl'), 'rb') as input_file:
        data = pickle.load(input_file)
             
    hist_data[3] = data[:, idx]
    
    colors = ['tab:blue', 'tab:green', 'tab:red', 'gray']
    labels = ['f=1.0', 'f=0.5', 'f=0.25', 'Random']
    plt.figure(figsize=(12, 9))
    for (i, h) in enumerate(hist_data):
        min_val = np.min(h)
        max_val = np.max(h)
        min_boundary = -1.0 * (min_val % bin_size - min_val)
        max_boundary = max_val - max_val % bin_size + bin_size
        n_bins = int((max_boundary - min_boundary) / bin_size) + 1
        bins = np.linspace(min_boundary, max_boundary, n_bins)
        plt.hist(h, color=colors[i], label=labels[i], alpha=0.2, bins=bins)
        
    # Plot vertical ideal line
    plt.axvline(x=dcg_ideal, color ='k', linestyle='--', linewidth=2, label='Ideal DCG')    
    
    plt.grid()
    plt.title('DCG Histogram, Tracks Seen = ' + str(idx + 1))
    plt.xlabel('DCG')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    
    # Save
    file_name = 'dcg_hist_seen' + str(idx + 1) + '.png'
    plt.savefig(os.path.join(save_fig, file_name), dpi=150)
        
        
        
        