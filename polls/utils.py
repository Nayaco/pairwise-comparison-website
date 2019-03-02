#-*-coding:utf-8-*-
import sys
import math
import numpy as np
import pandas as pd

matplotlib.rcParams['toolbar'] = 'None'

class BmapConfig():
    """The configuration of map plotting   
        
        including shape file, range, color.
    """

    lon1 = 120.85
    lon2 = 121.98
    lat1 = 30.68
    lat2 = 31.88
    
    n_lat = 133
    n_lon = 107

    step_lat = (lat2 - lat1) / n_lat
    step_lon = (lon2 - lon1) / n_lon
