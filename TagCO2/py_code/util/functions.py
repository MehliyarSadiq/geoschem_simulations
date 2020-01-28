"""
    Some useful functions
        Authors: Mehliyar Sadiq
        History: 2019-11-15, added area_latlon
"""

import numpy as np

#Approximate the area of a spatial grid square from the latitudes and longitudes of the diagonal vertices
def area_latlon(lat1, lon1, lat2, lon2):
    """
    This function calculates the area (in km^2) of a spatial grid square, given the latitudes and longitudes of the two diagonal vertices of the grid square.
    lat/lon is in angle; lat: [-90:90]; lon:[-180:180].
    lat1/lon1 and lat2/lon2 are thus the diagonal vertices of the square grid.
    """
    lat1 = lat1/180*np.pi
    lat2 = lat2/180*np.pi
    lon1 = lon1/180*np.pi
    lon2 = lon2/180*np.pi
    A = np.absolute(6371.009**2*(np.sin(lat2)-np.sin(lat1))*(lon2-lon1))
    return A
