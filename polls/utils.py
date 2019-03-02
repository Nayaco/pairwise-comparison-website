#-*-coding:utf-8-*-
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

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

class BmapPlotter(BmapConfig):
    """
    讲一下Plot的shape_file
    """
    def __init__(self, c1=[1, 1, 1], c2=[0.9, 0, 0],
                 levels=np.array([0, 0.25, 0.5, 0.75, 1]),
                 fig_size=(9, 9), shp_file=[0]):
        self.color1 = c1
        self.color2 = c2
        self.level_list = levels
        self.color_level = [ 1/(len(levels)-1) * i for i in range(len(levels))]
        self.fig = plt.figure(figsize=fig_size)
        self.ax1 = self.fig.add_axes([0.1,0.1,0.8,0.8])
        
        # draw bmap
        self.bmap = Basemap(projection='stere', llcrnrlon=self.lon1, 
                       llcrnrlat=self.lat1, urcrnrlon=self.lon2, 
                       urcrnrlat=self.lat2, lat_0=(self.lat1+self.lat2)/2, 
                       lon_0=(self.lon1+self.lon2)/2, ax=self.ax1, resolution='l')
        for shp_id in shp_file:
            shp_info = self.bmap.readshapefile(self.shape_file[shp_id],'states',drawbounds=False)
            for info, shp in zip(self.bmap.states_info, self.bmap.states):
                proid = info[self.info_name[shp_id]]
                if proid == self.proid_name[shp_id]:
                    poly = Polygon(shp,facecolor='none',edgecolor='k', lw=0.2)
                    self.ax1.add_patch(poly)
            self.bmap.drawcountries()
   
    def draw_density(self, density, norm_flag=False):
        x = np.linspace(self.lon1, self.lon2, self.n_lon)  
        y = np.linspace(self.lat1, self.lat2, self.n_lat)  
        X, Y = np.meshgrid(x, y)  
        X_bmap, Y_bmap = self.bmap(X, Y)
        Max = np.max(density) if norm_flag else np.max(1)
        density = density / Max
        
        plt.contourf(
            X_bmap, Y_bmap, density, len(self.level_list), 
            colors= self.blend_color(self.color1, self.color2, self.color_level),
            levels = np.max(density) * self.level_list, 
            alpha=0.8, antialiased=True) 
        plt.colorbar().draw_all()
        plt.title('Shanghai')
        # plt.show()
        plt.savefig('polls/static/images/resourve-value.png')
        return

    def draw_position(self, position, norm_flag=False):
        density = np.zeros([self.n_lat, self.n_lon])
        error_cnt = 0
        for i in range(len(position[:, 1])):
            lat_index = math.floor((position[i, 0] - self.lat1  ) / self.step_lat)
            lon_index = math.floor((position[i, 1] - self.lon1  ) / self.step_lon)
            if lon_index < self.n_lon and lon_index >= 0 \
                and lat_index >= 0 and lat_index < self.n_lat:
                density[lat_index][lon_index] += 1
            else:
                error_cnt += 1
        self.draw_density(density, norm_flag)
        return density, error_cnt