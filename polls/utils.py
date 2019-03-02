#-*-coding:utf-8-*-
import sys
import math
import numpy as np
import pandas as pd

home_dir = '/Users/tianqing'

matplotlib.rcParams['toolbar'] = 'None'

class BmapConfig():
    """The configuration of map plotting   
        
        including shape file, range, color.
    """

    GADM2_FILE_DIR  = home_dir+'/Downloads/GIS/gadm36_CHN_shp/gadm36_CHN_2'
    GADM3_FILE_DIR  = home_dir+'/Downloads/GIS/gadm36_CHN_shp/gadm36_CHN_3'
    CHINA_SHAPE_DIR = home_dir+'/Downloads/GIS/china-shape-new/chinaout'
    shape_file = [GADM2_FILE_DIR, GADM3_FILE_DIR, CHINA_SHAPE_DIR]
    info_name  = ['NL_NAME_1', 'NL_NAME_1', '所属省']
    info_name_dist  = ['NL_NAME_3', 'NL_NAME_3', 'NAME']
    proid_name = ['上海|上海', '上海|上海', '上海市']
    
    lon1 = 120.85
    lon2 = 121.98
    lat1 = 30.68
    lat2 = 31.88
    
    n_lat = 133
    n_lon = 107

    step_lat = (lat2 - lat1) / n_lat
    step_lon = (lon2 - lon1) / n_lon
    
    def blend_color(self, color1, color2, f_list):
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return [[r1 + (r2 - r1) * f, g1 + (g2 - g1) * f, b1 + (b2 - b1) * f]\
                for f in f_list]