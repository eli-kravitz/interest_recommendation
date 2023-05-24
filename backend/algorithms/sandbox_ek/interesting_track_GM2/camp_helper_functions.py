'''
Includes helper functions for CAMP
'''

import numpy as np

def get_lat_lon_centers(region):
    
    '''
    Description:
        Initialize coordinates of geographic regions
    
    Inputs:
        region: str
            str in {'africa',
                    'asia',
                    'caribbean',
                    'central_america',
                    'europe',
                    'north_america',
                    'oceania',
                    'south_america'
                    }
    
    Outputs:
        latitude: float
            latitude at center of region
        longitude: float
            longitude at center of region
    '''
    
    if region == 'africa':
        return -8.78, 34.51
    elif region == 'asia':
        return 34.05, 100.62
    elif region == 'caribbean':
        return 21.47, -78.66
    elif region == 'central_america':
        return 12.77, -85.60
    elif region == 'europe':
        return 54.53, 15.26
    elif region == 'north_america':
        return 47.12, -101.30
    elif region == 'oceania':
        return -22.74, 140.02
    elif region == 'south_america':
        return -8.78, -55.49
    else:
        print('Unknown region, returning 0, 0')
        return 0, 0
    
def get_geo_distance(latlon1, latlon2):
    
    '''
    Description:
        Initialize coordinates of geographic regions
    
    Inputs:
        latlon1: np.array(dtype=float64)
            latitude and longitude of first location
        latlon2: np.array(dtype=float64)
            latitude and longitude of second location
    
    Outputs:
        dist: float
            distance between locations
    '''
    
    dist = np.sqrt(sum((latlon1 - latlon2)**2))
    
    return dist
    