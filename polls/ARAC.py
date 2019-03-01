import numpy as np
import copy
import glob
import datetime
from django.utils import timezone
import logging

import gc
import os

KLGauss = lambda mu_0, mu_1, sigma_0,sigma_1:\
     sigma_1/sigma_0 + (mu_1-mu_0) * ((mu_1-mu_0)/sigma_1) \
     - len(mu_0) - np.log(sigma_0/sigma_1)

class Params():
    def __init__(self, L=0, N=0, M=0, K=0, layers=0, Lambda=0, kappa = 0, \
            mu=0, sigma=0, alpha=0, beta=0, eta=0, history=None):
        self.L = L     # number of entities to be compared
        self.N = N     # number of compares a worker makes
        self.M = M     # total numebr of workers
        self.K = K     # number of annotators. For simplicity, it's just 1 here
        
        self.Lambda = Lambda # tradeoff parameter

        self.mu    = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta  = beta
        self.eta   = eta
        self.emu   = np.exp(mu)
        self.history = history
        self.kappa = kappa

        self.mu_a = 0
        self.mu_b = 0
        self.sigma_a = 0
        self.sigma_b = 0
        self.pr = 0

        self.layers = layers # for test data generation
        # 这里check一下维数对不对



class ARAC():
    def __init__(self, params=None, load_from_prev=False, recalc=False):    
        logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                        filename='new.log',
                        filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        #a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        #日志格式
                        ) 
        
        if load_from_prev == False:
            self.params = params
            self.err_history = []
            # self.InitEntropyMatrix()
            try:
                self.EntropyMatrix=np.load('data/history/InitEntropy.npy')
            except:
                self.InitEntropyMatrix()
                logging.error('no entropyinit file found, reconstruct')
            else:
                logging.info('load entropyinit success')
            if recalc == False:
                self.SaveParams()
        else:
            # load from latest saved file
            list_entropy = glob.glob('data/history/entropy*')
            list_entropy.sort(reverse=True)
            if len(list_entropy) > 20:
                try:
                    for i in range(1, 6):
                        os.remove(list_entropy[-i])
                        os.remove(list_entropy[-i].replace('entropy', 'params'))
                    print("delete previous entropy file success")
                    logging.info('delete previous entropy file success')
                except:
                    print("delete previous entropy file failed")
                    logging.error('delete previous entropy file failed')
                    pass

            entropy_file = list_entropy[0]
            params_file  = entropy_file.replace('entropy', 'params')
            self.params = np.load(params_file)[0]
            self.EntropyMatrix = np.load(entropy_file)
            # print('OK')
            # print(self.params.L, len(self.params.mu))


    def get_mu(self):
        return self.params.mu

    def CalcEntropy(self, a, b):
        new_params = self.UpdateParams(\
            a,b,[self.params.history[a,b], self.params.history[b,a]])
        
        # div_a_ab=   KLGauss(params.mu(a),new_params.mu_a,params.sigma(a),new_params.sigma_a);
        div_a_ab = 0 + KLGauss(new_params.mu_a, self.params.mu[a], \
            new_params.sigma_a, self.params.sigma[a])
    
        div_b_ab = KLGauss(new_params.mu_b, self.params.mu[b],\
            new_params.sigma_b, self.params.sigma[b])
       # div_b_ab =0 + KLGauss(params.mu(b),new_params.mu_b,params.sigma(b),new_params.sigma_b);
        div_k_ab = 0
        pr_ab = new_params.pr
        
        new_params = self.UpdateParams(\
            b,a,[self.params.history[b,a],self.params.history[a,b]])
        #idecies confuse
        
        # div_a_ba=   KLGauss(params.mu(a),new_params.mu_a,params.sigma(a),new_params.sigma_a);
        div_a_ba = 0 + KLGauss(new_params.mu_a, self.params.mu[b],\
            new_params.sigma_a, self.params.sigma[b])
        
        div_b_ba = KLGauss(new_params.mu_b, self.params.mu[a],\
            new_params.sigma_b, self.params.sigma[a])
        # div_b_ba =0+ KLGauss(params.mu(b),new_params.mu_b,params.sigma(b),new_params.sigma_b);
        div_k_ba = 0
        pr_ba = new_params.pr
        
        
        div =pr_ab * (div_a_ab + div_b_ab + div_k_ab) + pr_ba * (div_a_ba + div_b_ba + div_k_ba)
        return div
        
    def InitEntropyMatrix(self):
        self.EntropyMatrix = np.zeros([self.params.L, self.params.L])
        for a in range(self.params.L):
            for b in range(self.params.L):
                if a == b :
                    continue
                div = self.CalcEntropy(a, b)

                self.EntropyMatrix[a][b] = div
                self.EntropyMatrix[b][a] = div

    def UpdateEntropyMatrix(self, a, b):
        if a == b:
            return
        div = self.CalcEntropy(a, b)
        self.EntropyMatrix[a][b] = div
        self.EntropyMatrix[b][a] = div

    def get_pairs(self):
        # get pairs with top N entropy
        index = np.argsort(self.EntropyMatrix.reshape([1, self.params.L**2]))[0][-self.params.N:]
        a_list, b_list = index // self.params.L, index % self.params.L
        return [[a_list[i], b_list[i]] for i in range(self.params.N)]

    def UpdateParams(self, a, b, this_history):
        
        def get_C():
            C1 = np.exp( this_history[0] ) / (np.exp(this_history[0]) + np.exp(this_history[1]))
            #+ 1/2* (params.sigma(a)+params.sigma(b))*params.emu(a)*params.emu(b)*\
            #(params.emu(b)-params.emu(a))/(params.emu(a)+params.emu(b))^3;
            #C2 = 1-C1
            #C = (C1 * self.params.alpha[0] + C2 * params.beta(1))/(params.alpha(1)+params.beta(1))
            return  C1
        self.params.kappa = 10e-4
        term =1 -self.params.emu[a]/(self.params.emu[a]+self.params.emu[b])
        # Eq 12
        new_params = Params()
        new_params.mu_a = self.params.mu[a] + self.params.sigma[a] * term
        # Eq 13
        new_params.mu_b = self.params.mu[b] - self.params.sigma[b] * term

        term = - self.params.emu[a]*self.params.emu[b]\
            /(self.params.emu[a]+self.params.emu[b]) ** 2

        # Eq 14
        new_params.sigma_a =  self.params.sigma[a]* max(1+ self.params.sigma[a]*term, self.params.kappa);
        # Eq 15
        new_params.sigma_b = self.params.sigma[b]* max(1+ self.params.sigma[b]*term, self.params.kappa);

        new_params.pr = get_C()

        return new_params

    def get_preference(self, Xt, a, b):
        val_at_a = Xt[a]
        val_at_b = Xt[b]
        
        layers  = self.params.layers * np.max(Xt)
        # print(layers, val_at_a, val_at_b)
        a_layer = np.where(layers >= val_at_a)[0][0]
        b_layer = np.where(layers >= val_at_b)[0][0]
        
        pref = val_at_a > val_at_b
        if a_layer == b_layer:
            
            p = 0.5 # np.exp(-(val_at_a+val_at_b)) # probability of wrong answer. 这里正则化一下
            r = np.random.rand()
            if r<p :
                pref = not pref
        
        return pref     

    def TestLearning(self, verbose = False):

        # X  = np.array([i for i in range(self.params.L)]) # 从1开始
        Xt = np.load('polls/out_list_1000_density.npy')
        ixXt = np.argsort(Xt)  ###### 这里等会儿把1全部剪掉

        for i in range(self.params.M):
            pairs = self.get_pairs() # 自己有参数##########TODO
            # 这里可以和random sample对比一下

            for j in range(self.params.N): # get N sample
                a = pairs[j][0]
                b = pairs[j][1]
                pref = self.get_preference(Xt, a, b) # ???? Xt是啥
                if pref == False:
                    a, b = b, a
                self.params.history[a, b] = self.params.history[a, b] + 1
                new_params = self.UpdateParams(a, b, [self.params.history[a, b], self.params.history[b, a]])

                # update
                
                self.params.mu[a] = new_params.mu_a
                self.params.mu[b] = new_params.mu_b
                
                self.params.sigma[a] = new_params.sigma_a
                self.params.sigma[b] = new_params.sigma_b
                
                self.params.emu[a] = np.exp(self.params.mu[a])
                self.params.emu[b] = np.exp(self.params.mu[b])

                # update entropy list
                # print(self.EntropyMatrix)
                # tmp = copy.copy(self.EntropyMatrix)
                for i_a in range(self.params.L):
                    # for i_b in range(self.params.L):
                    #     if i_a == i_b:
                    #         continue
                    self.UpdateEntropyMatrix(i_a, a)
                    self.UpdateEntropyMatrix(i_a, b)
                # print(a, b, '-----')
                # print(np.where(tmp - self.EntropyMatrix != 0)[0], '\n',np.where(tmp - self.EntropyMatrix != 0)[1])
                # exit()
                # for i_b in range(self.params.L):
                #     self.UpdateEntropyMatrix(i_b, b)
            
                # Get Error:
            # ix = np.argsort(self.params.mu)


            # self.err_history.append(sum(abs(ix - ixXt)))
            # if verbose and i % (self.params.M // 100) == 0:
            #     print('err', np.mean(abs(ix - ixXt)))
            if self.params.M <= 100:
                continue
            if verbose and i % (self.params.M // 100) == 0:
                cnt_correct = 0
                cnt_all = 0
                # Wilcoxon- Mann-Whitney statistics:
                for index_i in range(self.params.L):
                    for index_j in range(self.params.L):
                        if index_i == index_j:
                            continue
                        cnt_all += 1
                        if (self.params.mu[index_i] - self.params.mu[index_j]) * (Xt[index_i] - Xt[index_j]) > 0:
                            cnt_correct += 1
                acc = cnt_correct / cnt_all
                self.err_history.append((i, acc))
                print('acc', acc)

    def StartCrowdsourcing(self, verbose=True):
        X  = np.array([i for i in range(self.params.L)]) # 从1开始
        Xt = np.random.permutation(self.params.L)
        ixXt = np.argsort(Xt)  ###### 这里等会儿把1全部剪掉
        # get_pairs()
        pairs = self.get_pairs()
        # get_preference()
        for j in range(self.params.N): # get N sample
            a = pairs[j][0]
            b = pairs[j][1]
            pref = self.get_preference(Xt, a, b) # ------ Set as user action -----
            if pref == False:
                a, b = b, a
            self.params.history[a, b] = self.params.history[a, b] + 1
            new_params = self.UpdateParams(a, b, [self.params.history[a, b], self.params.history[b, a]])

        # load current parameters
            list_entropy = glob.glob('./history/entropy*')
            list_entropy.sort(reverse=True)
            entropy_file = list_entropy[0]
            params_file  = entropy_file.replace('entropy', 'params')
            self.params = np.load(params_file)[0]
            self.EntropyMatrix = np.load(entropy_file)
            
        # update the latest parameters

            self.params.mu[a] = new_params.mu_a
            self.params.mu[b] = new_params.mu_b
            
            self.params.sigma[a] = new_params.sigma_a
            self.params.sigma[b] = new_params.sigma_b
            
            self.params.emu[a] = np.exp(self.params.mu[a])
            self.params.emu[b] = np.exp(self.params.mu[b])

            # update entropy list
            # print(self.EntropyMatrix)
            # tmp = copy.copy(self.EntropyMatrix)
            for i_a in range(self.params.L):
                # for i_b in range(self.params.L):
                #     if i_a == i_b:
                #         continue
                self.UpdateEntropyMatrix(i_a, a)
                self.UpdateEntropyMatrix(i_a, b)
            
        # save parameter
            self.SaveParams()
            cnt_correct = 0
            cnt_all = 0
            # Wilcoxon- Mann-Whitney statistics:
            for index_i in range(self.params.L):
                for index_j in range(self.params.L):
                    if index_i == index_j:
                        continue
                    cnt_all += 1
                    if (self.params.mu[index_i] - self.params.mu[index_j]) * (Xt[index_i] - Xt[index_j]) > 0:
                        cnt_correct += 1
            acc = cnt_correct / cnt_all
            print('acc', acc)
    def StartRealCrowdsourcing(self, pairs, pref, verbose=True, recalc=False):
        # X  = np.array([i for i in range(self.params.L)]) # 从1开始
        # Xt = np.random.permutation(self.params.L)
        # ixXt = np.argsort(Xt)  ###### 这里等会儿把1全部剪掉
        # get_pairs()
        # pairs = self.get_pairs()
        # get_preference()

        # 已经输入了pairs和preference

        Xt = np.load('polls/out_list_1000_density.npy') # as a refernce
        ixXt = np.argsort(Xt) 

        for j in range(self.params.N): # get N sample
            print(pairs)
            a = pairs[j][0]
            b = pairs[j][1]

            # pref = self.get_preference(Xt, a, b) # ------ Set as user action -----

            if pref == False:
                a, b = b, a
            self.params.history[a, b] = self.params.history[a, b] + 1
            new_params = self.UpdateParams(a, b, [self.params.history[a, b], self.params.history[b, a]])

            if recalc:
                pass
            else:
                # load current parameters
                # not recalculating, normal routine
                list_entropy = glob.glob('data/history/entropy*')
                list_entropy.sort(reverse=True)
                entropy_file = list_entropy[0]
                params_file  = entropy_file.replace('entropy', 'params')
                self.params = np.load(params_file)[0]
                self.EntropyMatrix = np.load(entropy_file)
            
        # update the latest parameters

            self.params.mu[a] = new_params.mu_a
            self.params.mu[b] = new_params.mu_b
            
            self.params.sigma[a] = new_params.sigma_a
            self.params.sigma[b] = new_params.sigma_b
            
            self.params.emu[a] = np.exp(self.params.mu[a])
            self.params.emu[b] = np.exp(self.params.mu[b])

            # update entropy list
            # print(self.EntropyMatrix)
            # tmp = copy.copy(self.EntropyMatrix)
            for i_a in range(self.params.L):
                # for i_b in range(self.params.L):
                #     if i_a == i_b:
                #         continue
                self.UpdateEntropyMatrix(i_a, a)
                self.UpdateEntropyMatrix(i_a, b)
            
        # save parameter
            if recalc:
                pass
            else:
                self.SaveParams()
                cnt_correct = 0
                cnt_all = 0
                # Wilcoxon- Mann-Whitney statistics:
                for index_i in range(self.params.L):
                    for index_j in range(self.params.L):
                        if index_i == index_j:
                            continue
                        cnt_all += 1
                        if (self.params.mu[index_i] - self.params.mu[index_j]) * (Xt[index_i] - Xt[index_j]) > 0:
                            cnt_correct += 1
                acc = cnt_correct / cnt_all
                print('acc', acc)
        
        del Xt
        gc.collect()

    def SaveErrHistory(self):
        np.save('history'+str(datetime.datetime.now()), self.err_history)
    def PlotErrHistory(self):
        import matplotlib.pyplot as plt
        plt.plot([item[0] for item in self.err_history], [item[1] for item in self.err_history])
        plt.show()
    def SaveParams(self):
        time = str(datetime.datetime.now())
        np.save('data/history/params' + time, np.array([self.params]))
        np.save('data/history/entropy' + time, self.EntropyMatrix)  