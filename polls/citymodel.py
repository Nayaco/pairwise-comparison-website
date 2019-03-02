import numpy as np
import pandas as pd

class model():
    def __init__(self):
        self.F = np.load('data/spatial_capital/spatial_capital.npy')  # TODO!!!!
        self.N = len(self.F)  # number of items
        self.D = len(self.F[0])   # dimension of attribute vector
        self.v = np.load('data/spatial_capital/density_value.npy')
    def get_preference(self, v_1, v_2):
        pref = v_1 > v_2
        wrong_prob = 0 #.4 * np.exp(-np.abs(v_1 - v_2)) # 这些都可以改的
        if np.random.rand() < wrong_prob:
            pref = not pref
        return pref

    def get_pairs_random(self, n):
        return np.random.permutation(n)[:2]
    def get_accuracy_bycnt(self, real_v, gen_v):
        cnt = 0
        n = len(real_v)
        for i in range(n - 1):
            for j in range(i+1, n):
                if (real_v[i] - real_v[j]) * (gen_v[i] - gen_v[j]) >=0:
                    cnt += 1
        return cnt / (n * (n-1)/2)
  
    def get_accuracy_bycorr(self, real_v, gen_v):
        df = pd.DataFrame({'real rank':real_v,
             'res rank':gen_v})
        return df.corr()['real rank']['res rank']
    def get_accuracy_by_rankcorr(self, real_v, gen_v):
        real_v_rank = np.argsort(real_v)
        gen_v_rank  = np.argsort(gen_v)

        a = np.zeros([real_v.shape[0]])
        b = np.zeros([real_v.shape[0]])
        for index in range(len(real_v_rank)):
            a[real_v_rank[index]] = index
            b[gen_v_rank[index]]   = index
        df = pd.DataFrame({'real rank':a,
             'res rank':b})
        return df.corr()['real rank']['res rank']



