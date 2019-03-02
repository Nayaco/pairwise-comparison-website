import numpy as np
import tensorflow as tf
from .env import ranking_agent
from .citymodel import model
from .utils import BmapConfig, BmapPlotter
import datetime

def get_pair(n):
    # 随机选两个不相同的pair    
    return np.random.permutation(n)[:2]

def calc_params(compare_list, pref_list):
    my_model = model()
    num = len(pref_list)
    dataset = [np.zeros([num * 2, 2 * my_model.D]), \
           np.zeros([num * 2, 2]), np.zeros([num * 2, 1])]

    for i in range(num):
        pref  = pref_list[i]
        pair_index = compare_list[i][0]
        pairs = my_model.F[pair_index, :]
        dataset[0][2 * i] = np.append(pairs[0], pairs[1])
        dataset[1][2 * i] = [1, 0] if pref else [0, 1]
        dataset[2][2 * i] = 1 if pref else -1
        dataset[0][2 * i + 1] = np.append(pairs[1], pairs[0])
        dataset[1][2 * i + 1] = [1, 0] if not pref else [0, 1]
        dataset[2][2 * i + 1] = 1 if not pref else -1
    # training
    lr = 5e-3
    rankAgent = ranking_agent(lr, my_model.D)
    epoch_size = 1000
    batch_size = num
    train_flag = True
    save_dir   = None
    verbose    = True
    rankAgent.train(dataset, epoch_size, batch_size, my_model.F, 
        my_model.v, train_flag, save_dir, verbose) # training

    all_score = rankAgent.scores
    all_score = (all_score - np.min(all_score)) / (np.max(all_score) - np.min(all_score))

    np.save('data/all_score'+str(datetime.datetime.now()), all_score)
    
    # plot一下
    map_config = BmapConfig()
    level_num = 9
    plotter = BmapPlotter(c1=[0.9, 0.9, 1], c2=[0, 0, 0.9],
                     levels=np.linspace(start=0, stop=1, num=level_num + 1),
                     fig_size=(9, 9), shp_file=[0])
    isMainland = np.load('data/isMainland.npy')

    facility_value = - np.ones([map_config.n_lat, map_config.n_lon])
    facility_value[(isMainland[0], isMainland[1])] = np.reshape(all_score, [len(all_score), ])
    # plot
    # plotter.draw_density(facility_value)

    return 