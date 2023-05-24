'''
Class contains all methods to update regression weights given labeled tracks,
as well as all methods to infer interest of unlabeled tracks.

The two main methods are:
    update_p_theta - update p(theta) with p(theta|I,G,x) when user provides
                     track interest/lack of interest
    infer_interest - infer p(I|G,x,theta) for a given track
'''

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import camp_helper_functions as helpers

class CAMPInterestClassifier():
    
    def __init__(self, p_g_given_o, user_id, session_id, u_o, u_g,
                 regions, targets):
        
        '''
        Description:
            Initialize CAMP interest classifier with relevant parameters
            based on user parameters
        
        Inputs:
            p_g_given_o: dict(dtype = np.array(dtype=float))
                pmf for geographic location for each object type
            user_id: int
                unique user ID
            session_id: int
                unique session ID
            u_o: list(dtype = int)
                user object type preference
            u_g: list(dtype = int)
                user geographic location preference
            regions: list(dtype=str)
                list of possible geographic regions
            targets: list(dtype=str)
                list of possible object types
        
        Outputs:
            N/A
        '''
        
        self.p_g_given_o = p_g_given_o
        self.user_id = user_id
        self.session_id = session_id
        self.u_o = u_o
        self.u_g = u_g
        self.regions = regions
        self.targets = targets
        self.p_theta = self.build_p_theta()
        
    def build_p_theta(self):
        
        '''
        Description:
            Build p(theta) from user inputs
        
        Inputs:
            N/A
        
        Outputs:
            p_theta: scipy.stats.multivariate_normal
                pdf for theta
        '''
        
        # theta = [intercept, o_1,...,o_n, g_1,...,g_n, f(x)_1,...,f(x)_n]
        # Assume f(x) = [max(alt), max(intensity), max(speed)]
        # Have larger variance for no preference, smaller when preference given
        mean = np.array([0] + self.u_o + self.u_g + [0, 0, 0])
        var_int = [3]
        var_obj = []
        for i in range(len(self.u_o)):
            if self.u_o[i] == -1:
                var_obj.append(1)
            elif self.u_o[i] == 0:
                var_obj.append(3)
            else:
                var_obj.append(1)
        var_geo = []
        for i in range(len(self.u_g)):
            if self.u_g[i] == -1:
                var_geo.append(1)
            elif self.u_g[i] == 0:
                var_geo.append(3)
            else:
                var_geo.append(1)
        var_x = [3, 3, 3]
        var = var_int + var_obj + var_geo + var_x
        cov = np.diag(var)
        p_theta = stats.multivariate_normal(mean=mean, cov=cov)
        
        return p_theta
    
    def update_p_theta(self, p_o_given_x, x, I):
        
        '''
        Description:
            Update p(theta) given classifier prediction, track features, and
            labeled interest/lack of interest
        
        Inputs:
            p_o_given_x: np.array(dtype=float)
                pmf for object type from classifier
            x: np.ndarray(size=(n_observations, 3), dtype=float)
                [speed, altitude, intensity, lat, lon]
            I: int
                binary interest value
        
        Outputs:
            N/A
        '''
        
        # Get derived track features
        fx = self.get_derived_track_features(x)
        
        # Find geographic location
        G = self.get_geographic_location(x)
        
        # Now do Laplace approximation to approximate the posterior
        # p(theta|I,O,x) as a Gaussian
        
        # Do gradient descent to get optimal theta given data
        theta_star = self.optimize_theta(p_o_given_x, fx, G, I)
        
        # Numerically get Hessian evaluated at theta*
        H = self.get_hessian(p_o_given_x, fx, G, I, theta_star)
        
        # Finally, p(theta|x,G,I) ~ N(theta*, inv(H))
        # Make posterior new prior
        self.p_theta = stats.multivariate_normal(mean=theta_star,
                         cov=np.linalg.inv(H))
        
    def infer_interest(self, p_o_given_x, x):
        
        '''
        Description:
            Infer track interest
        
        Inputs:
            p_o_given_x: np.array(dtype=float)
                pmf for object type from classifier
            x: np.ndarray(size=(n_observations, 3), dtype=float)
                [speed, altitude, intensity, lat, lon]
        
        Outputs:
            p_I: np.array(dtype=float)
                p(I) for I = 0 and I = 1
        '''
        
        # Get derived track features
        fx = self.get_derived_track_features(x)
        
        # Find geographic location
        G = self.get_geographic_location(x)
        
        # Get E[theta]
        theta = self.p_theta.mean
        
        # Get p(I) - proportional
        p_I = np.zeros(2)
        for I in range(0, 2):
            tmp_sum = 0.
            for i in range(len(p_o_given_x)):
                first = p_o_given_x[i]
                second = self.p_g_given_o[i][np.where(G == 1)[0][0]]
                O = np.zeros(len(p_o_given_x))
                O[i] = 1.
                X = np.concatenate((np.array([1]), O, G, fx))
                third = 1 / (1 + np.exp(-np.dot(theta, X)))
                if I == 0:
                    third = 1 - third
                tmp_sum = tmp_sum + first * second * third
            p_I[I] = tmp_sum
        
        # Normalize because p(I=0) + p(I=1) = 1
        p_I = p_I / sum(p_I)
        
        return p_I
        
    def get_derived_track_features(self, x):
        
        '''
        Description:
            Get derived track features 
            f(x) = [max(alt), max(intensity), max(speed)]
        
        Inputs:
            x: np.ndarray(size=(n_observations, 3), dtype=float)
                [speed, altitude, intensity, lat, lon]
        
        Outputs:
            fx: np.array(dtype=float)
                derived track features
        '''
        
        fx = np.array([max(x[:, 1]), max(x[:, 2]), max(x[:, 0])])
        
        # Change scale of these features so they're more similar to one-hot
        # Scaled by maximum expected value of each feature
        
        # These are the max values from all delivered files
        # TODO: put in database somewhere?
        a1 = 80764
        a2 = 20800
        a3 = 4152
        
        fx[0] = fx[0] / a1
        fx[1] = fx[1] / a2
        fx[2] = fx[2] / a3
        
        return fx
    
    def get_geographic_location(self, x):
        
        '''
        Description:
            Get geographic location of track
        
        Inputs:
            x: np.ndarray(size=(n_observations, 5), dtype=float)
                [speed, altitude, intensity, lat, lon]
        
        Outputs:
            G: np.array(dtype=int)
                one-hot encoded geographic location
        '''
        
        latlon = np.array([x[-1, 3], x[-1, 4]])
        
        dist = np.inf
        ct = 0
        for r in self.regions:
            
            lat, lon = helpers.get_lat_lon_centers(r)
            d = helpers.get_geo_distance(latlon, np.array([lat, lon]))
            if d < dist:
                dist = d
                idx = ct
                
            ct += 1
            
        G = np.zeros(len(self.regions), dtype=int)
        G[idx] = 1
        
        return G
    
    def eval_E_theta(self, theta, p_o_given_x, fx, G, I):
        
        '''
        Description:
            Evaluates E(theta)
        
        Inputs:
            theta: np.array(dtype=theta)
                realization of theta
            p_theta: scipy.stats.multivariate_normal
                pdf for theta
            p_o_given_x: np.array(dtype=float)
                pmf for object type from classifier
            p_g_given_o: dict(dtype = np.array(dtype=float))
                pmf for geographic location for each object type
            fx: np.array(dtype=float)
                derived track features
            G: np.array(dtype=int)
                one-hot encoded geographic location
            I: int
                binary interest value
        
        Outputs:
            val: float
                function value at given point
        '''
        
        # Evaluate function, noting that this is specific to E(theta) used in
        # this problem
        
        val = -np.log(self.p_theta.pdf(theta))
        tmp_sum = 0.
        for i in range(len(p_o_given_x)):
            first = p_o_given_x[i]
            second = self.p_g_given_o[i][np.where(G == 1)[0][0]]
            O = np.zeros(len(p_o_given_x))
            O[i] = 1.
            X = np.concatenate((np.array([1]), O, G, fx))
            third = 1 / (1 + np.exp(-np.dot(theta, X)))
            if I == 0:
                third = 1 - third
            tmp_sum = tmp_sum + first * second * third
        if tmp_sum == 0:
            tmp_sum = 1e-20
        val = val - np.log(tmp_sum)
        
        return val
    
    def optimize_theta(self, p_o_given_x, fx, G, I):
        
        '''
        Description:
            Optimize theta using gradient descent
        
        Inputs:
            p_o_given_x: np.array(dtype=float)
                pmf for object type from classifier
            fx: np.array(dtype=float)
                derived track features
            G: np.array(dtype=int)
                one-hot encoded geographic location
            I: int
                binary interest value
        
        Outputs:
            theta_star: np.array(dtype=float)
                optimized theta*
        '''
        
        # Optimize with scipy
        
        theta_star = minimize(self.eval_E_theta, self.p_theta.mean, method='BFGS',
                   args=(p_o_given_x, fx, G, I),
                   tol=1e-4, options={'disp': False})
        theta_star = theta_star.x
        
        return theta_star
    
    def get_hessian(self, p_o_given_x, fx, G, I, theta):
        
        '''
        Description:
            Get Hessian numerically
        
        Inputs:
            p_o_given_x: np.array(dtype=float)
                pmf for object type from classifier
            fx: np.array(dtype=float)
                derived track features
            G: np.array(dtype=int)
                one-hot encoded geographic location
            I: int
                binary interest value
            theta: np.array(dtype=float)
                where to evaluate Hessian
        
        Outputs:
            H: np.ndarray(dtype=float)
                Hessian matrix evaluated at theta
        '''
        
        # Step size for central_differences
        h = 1e-5
        
        # Initialize Hessian
        H = np.zeros(shape=self.p_theta.cov.shape)
        
        # Populate Hessian
        for i in range(self.p_theta.cov.shape[0]):
            for j in range(self.p_theta.cov.shape[1]):
                if i > j:
                    
                    # Move on, will populate later since matrix is symmetric
                    continue
                    
                elif i == j:
                    
                    # Calculate along diagonal
                    theta_diff = np.zeros(len(theta))
                    theta_diff[i] = h
                    upper = self.eval_E_theta(theta + theta_diff, 
                                         p_o_given_x, fx, G, I)
                    middle = self.eval_E_theta(theta, p_o_given_x, 
                                         fx, G, I)
                    lower = self.eval_E_theta(theta - theta_diff, 
                                         p_o_given_x, fx, G, I)
                    H[i][j] = (upper - 2 * middle + lower) / h**2
                    
                else:
                    
                    # Calculate non-diagonal entries
                    theta_diff = np.zeros(len(theta))
                    theta_diff[i] = h
                    theta_diff[j] = h
                    upper_upper = self.eval_E_theta(theta + theta_diff, 
                                               p_o_given_x, fx, G, I)
                    theta_diff[i] = h
                    theta_diff[j] = -h
                    upper_lower = self.eval_E_theta(theta + theta_diff, 
                                               p_o_given_x, fx, G, I)
                    theta_diff[i] = -h
                    theta_diff[j] = h
                    lower_upper = self.eval_E_theta(theta + theta_diff, 
                                               p_o_given_x, fx, G, I)
                    theta_diff[i] = -h
                    theta_diff[j] = -h
                    lower_lower = self.eval_E_theta(theta + theta_diff, 
                                               p_o_given_x, fx, G, I)
                    
                    # Populate both elements
                    H[i][j] = H[j][i] = (upper_upper - upper_lower - \
                               lower_upper + lower_lower) / (4 * h**2)
                    
        return H
