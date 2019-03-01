import numpy as np
from ARAC import ARAC, Params

verbose = True

L = 1000  # number of entities to be compared
N = 1    # number of compares a worker makes
M = 2000  # total numebr of workers
K = 1     # number of annotators. For simplicity, it's just 1 here

Lambda = 1 # represents the tradeoff be- tween exploration and exploitation

alpha_init = 2
beta_init  = 1
eta_init   = 1
mu_init    = 25
sigma_init = 25/25
kappa      = 10e-4

# for test
layers = np.array([0.5, 1.0])

alpha = np.ones([K,1]) * alpha_init #？？？？ 确定是L？
beta  = np.ones([K,1]) * beta_init
eta   = np.ones([K,1]) * eta_init
mu    = np.ones(L) * mu_init / 2
emu   = np.exp(mu)
sigma = np.ones([L,1]) * sigma_init # sigma squared

history = np.zeros([L, L])

params = Params( L, N, M, K, layers, Lambda, kappa,\
        mu, sigma, alpha, beta, eta, history)
model = ARAC(params)

model.TestLearning(verbose)
model.SaveErrHistory()
model.PlotErrHistory()
model.SaveParams()
