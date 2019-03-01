import numpy as np
from .ARAC import ARAC, Params

def init():

	L = 1000  # number of entities to be compared
	N = 1    # number of compares a worker makes
	M = 1  # total numebr of workers
	K = 1     # number of annotators. For simplicity, it's just 1 here

	Lambda = 1 # represents the tradeoff be- tween exploration and exploitation

	alpha_init = 10
	beta_init  = 1
	eta_init   = 1
	mu_init    = 25
	sigma_init = 25/3
	kappa      = 10e-4

	layers = np.array([0.25, 0.7,1.0])


	mu    = np.ones(L) * mu_init / 2
	emu   = np.exp(mu)
	sigma = np.ones([L,1]) * sigma_init # sigma squared

	history = np.zeros([L, L])

	params = Params( L, N, M, None, layers, Lambda, kappa,\
	        mu, sigma, None, None, None, history)

	model = ARAC(params, load_from_prev=False)
	# model.StartCrowdsourcing(verbose=True)

def test():
	model = ARAC(load_from_prev=True)
	model.StartCrowdsourcing(verbose=True)

def get_pair():
	model = ARAC(load_from_prev=True)
	return model.get_pairs()[0]

def update_params(pairs, pref):
	model = ARAC(load_from_prev=True)
	model.StartRealCrowdsourcing(pairs, pref)
	# and it saved parameters

def get_mu():
	model = ARAC(load_from_prev=True)
	return model.get_mu()

def RecalcParams(compare_list, pref_list):
	L = 1000  # number of entities to be compared
	N = 1    # number of compares a worker makes
	M = 1  # total numebr of workers
	K = 1     # number of annotators. For simplicity, it's just 1 here

	Lambda = 1 # represents the tradeoff between exploration and exploitation

	alpha_init = 10
	beta_init  = 1
	eta_init   = 1
	mu_init    = 25
	sigma_init = 25/3
	kappa      = 10e-4

	layers = np.array([0.25, 0.7,1.0])


	mu    = np.ones(L) * mu_init / 2
	emu   = np.exp(mu)
	sigma = np.ones([L,1]) * sigma_init # sigma squared

	history = np.zeros([L, L])

	params = Params( L, N, M, None, layers, Lambda, kappa,\
	        mu, sigma, None, None, None, history)

	model = ARAC(params, load_from_prev=False, recalc=True)
	for index in range(len(compare_list)):
		pairs = compare_list[index]
		pref = pref_list[index]
		model.StartRealCrowdsourcing(pairs, pref, recalc=True)
	model.SaveParams()

init()
print('after initialization')
# for i in range(100):
# 	test()
