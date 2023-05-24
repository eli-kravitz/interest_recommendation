import matplotlib.pyplot as plt

from backend.io.AbstractIOAdapter import AbstractIO
import numpy as np



class CSV_IO(AbstractIO):
    TRK = 'trackID parentID lifetime hit_count  report_time miss_count tot_hit_count status trackClass trackPhase svYear svDay    svSeconds       pos_ECF_x       pos_ECF_y       pos_ECF_z       vel_ECF_x       vel_ECF_y       vel_ECF_z       acc_ECF_x       acc_ECF_y       acc_ECF_z      cv_x_x     cv_y_x     cv_y_y     cv_z_x     cv_z_y     cv_z_z    cv_Vx_x    cv_Vx_y    cv_Vx_z   cv_Vx_Vx    cv_Vy_x    cv_Vy_y    cv_Vy_z   cv_Vy_Vx   cv_Vy_Vy    cv_Vz_x    cv_Vz_y    cv_Vz_z   cv_Vz_Vx   cv_Vz_Vy   cv_Vz_Vz    cv_Ax_x    cv_Ax_y    cv_Ax_z   cv_Ax_Vx   cv_Ax_Vy   cv_Ax_Vz   cv_Ax_Ax    cv_Ay_x    cv_Ay_y    cv_Ay_z   cv_Ay_Vx   cv_Ay_Vy   cv_Ay_Vz   cv_Ay_Ax   cv_Ay_Ay    cv_Az_x    cv_Az_y    cv_Az_z   cv_Az_Vx   cv_Az_Vy   cv_Az_Vz   cv_Az_Ax   cv_Az_Ay   cv_Az_Az'
    OBS = 'year day      seconds irradiance_wcmsq intensity_kwsr      los_ecf_x      los_ecf_y      los_ecf_z losSigma      eph_ecf_x      eph_ecf_y      eph_ecf_z  eph_ecf_Vx  eph_ecf_Vy  eph_ecf_Vz    flags satID      SNR cso_count gof_score simTag'

    def read(self, filename):
        track_data = []
        obs_data = []
        type_data = []

        data = []
        try:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line[0] == '#':
                        continue
                    line = line.split()
                    if line[0] == 'TRK':
                        # print("processing trk")
                        track_data.append(line)
                    elif line[0] == 'OBS':
                        # print('processing tpe')
                        obs_data.append(line)
                    elif line[0] == 'TPE':
                        # print('processing obs')
                        type_data.append(line)
        except Exception as e:
            print("CSV READ ERROR")
            print(e)

        TRK = self.TRK.strip().split()
        OBS = self.OBS.strip().split()
        keys = [*TRK, *OBS]
        values = []

        for t, o in zip(track_data, obs_data):
            values.append([*t[1:], *o[1:]])
        return keys, values

    def read_sat1(self, filename):
        track_data_1 = []
        track_data_2 = []
        obs_data_1 = []
        obs_data_2 = []
        tpe_data = []

        # try:
        with open(filename, 'r') as f:
            set_lines = f.readlines()
            for l in range(0, len(set_lines)):
                # print(l)
                line = set_lines[l]
                line = line.strip()
                if line[0] == '#':
                    continue
                line = line.split()
                if line[0] == 'TRK':
                    # print(line)
                    # print("processing trk")

                    sat_ID = set_lines[l + 1]
                    sat_ID = sat_ID.strip()
                    sat_ID = sat_ID.split()
                    # print(sat_ID)
                    ID = sat_ID[17]
                    if ID == '1':
                        track_data_1.append(line)
                    elif ID == '2':
                        track_data_2.append(line)
                    # track_data_1.append(line)
                elif line[0] == 'OBS':
                    ID = line[17]
                    if ID == '1':
                        obs_data_1.append(line)
                    elif ID == '2':
                        obs_data_2.append(line)
                    # obs_data_1.append(line)
                    # print('processing tpe')
                elif line[0] == 'TPE':
                    # print('processing obs')
                    tpe_data.append(line)

        return track_data_1, obs_data_1, tpe_data

    def write(self, data):
        pass

    def extract_data(self, filename):
        """Function is used to extract necessary data from .tmf files
        X,Y,and Z velocity data is extracted and normalized to find speed
        Intensity values are left raw

        Inputs:
            track data- lists of lists in strings
            obs data - lists of lists in strings
        Outputs:
            n x 3 numpy array """
        raw_data = self.read(filename)

        # Extract as string
        data_length = np.shape(raw_data[1])[0]
        x_str = [raw_data[1][i][16] for i in range(0, data_length)]
        y_str = [raw_data[1][i][17] for i in range(0, data_length)]
        z_str = [raw_data[1][i][18] for i in range(0, data_length)]
        # Convert to numpy array to float
        x_array = np.array(x_str)
        x = x_array.astype(float)
        y_array = np.array(y_str)
        y = y_array.astype(float)
        z_array = np.array(z_str)
        z = z_array.astype(float)
        # Calculate total speed
        speed = np.linalg.norm([x, y, z], axis=0)
        # Make covariance matrix to find total speed uncertainty
        speed_std_vec = np.zeros((data_length,))
        for t in range(data_length):
            # Extract data
            cv_Vxx = raw_data[1][t][31]
            cv_Vyx = raw_data[1][t][35]
            cv_Vyy = raw_data[1][t][36]
            cv_Vzx = raw_data[1][t][40]
            cv_Vzy = raw_data[1][t][41]
            cv_Vzz = raw_data[1][t][42]
            # Create matrix
            cv_mat = np.array([[cv_Vxx, cv_Vyx, cv_Vzx], [cv_Vyx, cv_Vyy, cv_Vzy], [cv_Vzx, cv_Vzy, cv_Vzz]])
            # Normalize matrix using Frobenius norm
            speed_std = np.sqrt(np.linalg.norm(cv_mat, 'fro'))
            # speed_std_vec.append(speed_std)
            speed_std_vec[t] = speed_std

        intensity_str = [raw_data[1][i][71] for i in range(0, data_length)]
        intensity_array = np.array(intensity_str)
        intensity = intensity_array.astype(float)
        data = np.column_stack((speed, intensity, speed_std_vec))
        return data

