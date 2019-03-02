"""
This is the environment settings for ranking spatial capital
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random 

def get_accuracy_bycnt(real_v, gen_v):
    cnt = 0
    n = len(real_v)
    for i in range(n - 1):
        for j in range(i+1, n):
            if (real_v[i] - real_v[j]) * (gen_v[i] - gen_v[j]) >=0:
                cnt += 1
    return cnt / (n * (n-1)/2)
def get_accuracy_bycorr(real_v, gen_v):
    df = pd.DataFrame({'real rank':real_v,
         'res rank':gen_v})
    return df.corr()['real rank']['res rank']
def get_accuracy_by_rankcorr(real_v, gen_v):
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

def calc_ROC_param(prediction_value, labels, t): 
    
    index_0 = np.where(prediction_value >= t)[0] # >= threshold means labels[0] == 1
    
    index_1 = np.where(prediction_value < t)[0]
    
    TP = len(np.where(labels[index_0, 0] == 1)[0]) # 预测为正的那些里面，有多少个对的
    FN = len(np.where(labels[index_1, 0] == 1)[0]) # 预测为错的里面，有多少个错的
    TN = len(np.where(labels[index_1, 0] == 0)[0]) # 预测为错的里面，有多少对的
    FP = len(np.where(labels[index_0, 0] == 0)[0]) # 预测为正的里面，有多少错的
    return TP, TN, FP, FN           


class environment():
    def __init__(self,bag_len):
        self.bag_len = bag_len

    def reset(self, bag_pairs_ebd, index_pair_ids, bag_reward): 
        '''
        bag_pairs_ebd : N_bag * D(imension)
        bag_reward    : reward of selecting this bag
        index_pair_ids: 前面bag_pairs_ebd的全局编号 
        '''
        self.K = 50 # expected 一个bag里面最多选几个
        self.penalty_selected_num = lambda x, eps:x if x >= eps else - (self.bag_len - eps)/eps * x + self.bag_len
        self.relu = lambda x, eps:x if x >= eps else eps
        self.lambda_penalty = 0.5
        
        self.index_pair_ids = index_pair_ids
        self.bag_reward     = bag_reward       # 这是啥？？？
        self.dimension      = len(bag_pairs_ebd[0])
        self.bag_len        = len(bag_pairs_ebd)  #这个bag里面所有Pair的数量
        self.bag_pairs_ebd  = bag_pairs_ebd # 所有pair的attributes
        self.current_step   = 0
        self.num_selected   = 0
        self.list_selected  = []

        # 当前环境下的vector
        self.vector_current = self.bag_pairs_ebd[self.current_step, :] 

        # initialize
        # 我们现在解空间超大的，这个avg会不会不管用。。。或者partition之前k-means一下
        self.vector_mean    = np.array([0.0 for x in range(self.dimension)],dtype=np.float32)
        self.vector_sum     = np.array([0.0 for x in range(self.dimension)],dtype=np.float32)

        # current state包含了pairs的embedding与整个bag的avg embedding
        current_state       = [self.vector_current,self.vector_mean] # 
        return current_state

    def step(self, action):
        if action == 1:
            self.num_selected +=1
            self.list_selected.append(self.current_step)
            
        # 设计了一下Current reward!!!!
        
        
        current_reward_immediate = self.bag_reward[self.current_step]
        penalty = self.relu(self.num_selected, self.K) / self.bag_len
        current_reward_immediate = (current_reward_immediate )#- self.lambda_penalty * penalty)
        
        self.vector_sum = self.vector_sum + action * self.vector_current

        if self.num_selected == 0:
            self.vector_mean = np.array([0.0 for x in range(self.dimension)],dtype=np.float32)
        else:
            self.vector_mean = self.vector_sum / self.num_selected

        self.current_step +=1

        if (self.current_step < self.bag_len):
            self.vector_current = self.bag_pairs_ebd[self.current_step]

        current_state = [self.vector_current, self.vector_mean] # 只看vector了是吗？ 其实还是得有id信息的，不然没法反推
        
        return current_state, current_reward_immediate
        # do an action. return current_state

    def reward(self):
        assert (len(self.list_selected) == self.num_selected) 
        
        reward = [self.bag_reward[x] for x in self.list_selected] # reward??
        reward = np.array(reward)
        
        reward = np.mean(reward) # - self.lambda_penalty * self.penalty_selected_num(self.num_selected, self.K) / self.bag_len       
        return reward
        # reward咋填的？

def get_action(prob):
    tmp = prob[0] 
    result = np.random.rand()
    if result>0 and result< tmp:
        return 1
    elif result >=tmp and result<1:
        return 0

def decide_action(prob): 
    tmp = prob[0]
    if tmp>=0.5:
        return 1
    elif tmp < 0.5:
        return 0

class agent():
    def __init__(self, lr, s_size): 
        """
            The RL agent
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32) # state 的size???
        self.input = self.state_in 

        # load pre-train vectors 

        all_vars_init = np.load('./rankmodel/all_vars_init.npy')
        num_hidden = s_size * 2
        self.hidden = layers.fully_connected(self.input,num_hidden,
#                                              tf.nn.relu, 
                                             weights_initializer= tf.constant_initializer(all_vars_init[0]),#tf.truncated_normal_initializer(mean=0.01, stddev=0.01),
                                                biases_initializer = tf.constant_initializer(all_vars_init[1]),
                                             trainable = False,
                                            )
        self.hidden2 = layers.fully_connected(self.hidden, 2 * num_hidden,
        #                                              tf.nn.relu, 
                                             weights_initializer= tf.constant_initializer(all_vars_init[2]),# tf.truncated_normal_initializer(mean=0.01, stddev=0.01),
                                             biases_initializer = tf.constant_initializer(all_vars_init[3]),
                                              trainable = False
                                             )
        self.prob = tf.reshape(layers.fully_connected(self.hidden2,1,
                                                      tf.nn.sigmoid,
                                                      weights_initializer=tf.constant_initializer(all_vars_init[4]), #tf.truncated_normal_initializer(mean=0, stddev=0.01),
                                                        biases_initializer = tf.constant_initializer(all_vars_init[5]),
                                                     ),[-1]) 
        # compute loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        # the probability of choosing 0 or 1 policy function
        self.pi  = self.action_holder * self.prob + (1 - self.action_holder) * (1 - self.prob)
        # loss
        self.loss = - tf.reduce_mean( tf.log(self.pi) * self.reward_holder  )
        # minimize loss
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)
        self.tvars = tf.trainable_variables()
            
        #manual update parameters 这个又是啥？？？
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)
        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)
        #compute gradient
        self.gradients = tf.gradients(self.loss, self.tvars)
        #update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))


def getBatch(dataset, n):
    sample_list = random.sample(range(len(dataset[0])), n)
    return sample_list

class ranking_agent():
    """
        Ranking model.
        train() is used to compute the parameters
    """
    def __init__(self, lr, D):
        self.g_rank = tf.Graph()
        self.sess = tf.Session(graph=self.g_rank)

        with self.g_rank.as_default():
            with self.sess.as_default():
                self.X = tf.placeholder("float", shape = [None, D * 2]) 

                self.X_1 = self.X[:, 0:D] # tf.placeholder("float", shape = [None, D * 1]) 
                self.X_2 = self.X[:, D:]  # tf.placeholder("float", shape = [None, D * 1]) 
                self.y_= tf.placeholder("float", shape = [None, 2]) # 0 or 1
                self.pref_holder = tf.placeholder("float", shape = [None, 1]) # -1 or 1

                self.Theta = tf.Variable(tf.truncated_normal([D, 1], mean=0, stddev=0.1, )) # small bias
                self.b     = tf.Variable(tf.constant(.01, shape = [1]))# small bias

                # get the score of all value
                self.all_F = tf.placeholder("float", shape = [None, D]) 
                self.score_all = tf.matmul(self.all_F, self.Theta) + self.b
                self.score_all_sigmoid = tf.sigmoid(tf.matmul(self.all_F, self.Theta) + self.b)
                
                self.score_1 = tf.sigmoid(tf.matmul(self.X_1, self.Theta) + self.b) # N * 1  提问，这里有没有必要Sigmoid？
                self.score_2 = tf.sigmoid(tf.matmul(self.X_2, self.Theta) + self.b) # N * 1
                self.y = tf.nn.softmax(tf.concat([self.score_1, self.score_2], 1)) # N * 2 
                self.score_1_out = tf.matmul(self.X_1, self.Theta) + self.b # N * 1  提问，这里有没有必要Sigmoid？
                self.score_2_out = tf.matmul(self.X_2, self.Theta) + self.b # N * 1
                self.y_out = tf.nn.softmax(tf.concat([self.score_1_out, self.score_2_out], 1))

                self.entropy = tf.reduce_sum(- tf.log(self.y) *  self.y, axis=1) #这个来衡量一个pair的比较的准确度
                # self.entropy1 = tf.reduce_sum(- tf.log(self.y) *  self.y, axis = 1)

                self.cross_entropy = -tf.reduce_mean(self.y_ * tf.log(self.y)) # 这个是用于loss的cross_entropy
                self.correct_prediction_bin = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1)) # binary comparison accuracy
                self.accuracy_bin = tf.reduce_mean(tf.cast(self.correct_prediction_bin,"float"))

                self.zeros = tf.placeholder("float", shape = [None, 1])
                self.ranking_loss = -tf.reduce_mean(\
                        tf.pow( tf.maximum(self.pref_holder - self.pref_holder, (self.score_1 - self.score_2) * self.pref_holder) , 2) ) # 0 * 那啥么

                self.optimizer = tf.train.AdamOptimizer(lr)

                self.loss = self.cross_entropy # + 1 * self.ranking_loss
                # this loss is not 科学吧？为啥要reduce_sum? reduce_mean不好吗

                self.train_step = self.optimizer.minimize(self.loss)

                self.all_rank_vars = tf.trainable_variables() # all vars

                # gradients
                self.rank_gradients = tf.gradients(self.ranking_loss, self.all_rank_vars)

                self.gradient_holders = []
                for idx, var in enumerate(self.all_rank_vars):
                    placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                    self.gradient_holders.append(placeholder)       
                self.rank_update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.all_rank_vars))

                # 保存的一些参数：
                self.last_loss = -100000
                self.last_acc  = 0
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
    def getEntropy(self, all_pair_attr):
        return self.entropy.eval(session=self.sess, feed_dict={self.X:np.array(all_pair_attr)})
    
    def train(self, dataset, epoch_size, batch_size, F, v, train_flag=False, save_dir=None, verbose=False):
        """
            v这个东西有点神奇，有可能没有的（但在我们这里，考虑v=人口密度）
        """
        with self.g_rank.as_default():
            with self.sess.as_default():          
                if train_flag == True:
                    for epoch in range(epoch_size):
                        for i in range(len(dataset[0]) // batch_size): 
                            batch_index = getBatch(dataset, batch_size)        
                            feed_dict = {self.X:dataset[0][batch_index], 
                                         self.y_:dataset[1][batch_index], 
                                         self.pref_holder:dataset[2][batch_index]}
                            grads = self.sess.run(self.rank_gradients, feed_dict=feed_dict)
                            feed_dict = dictionary = dict(zip(self.gradient_holders, grads)) #这样可以吗
                            self.sess.run(self.rank_update_batch, feed_dict=feed_dict)

                        if (epoch + 1) % (epoch_size//10) == 0:
                            train_accuracy_all = self.accuracy_bin.eval(session = self.sess,
                                                           feed_dict = {self.X:dataset[0], 
                                                                        self.y_:dataset[1],
                                                                        self.pref_holder:dataset[2]})
                            self.scores = self.score_all.eval(session=self.sess, feed_dict={self.all_F:F})
                            acc_stat =get_accuracy_bycorr(real_v=v, gen_v=np.reshape(self.scores, [self.scores.shape[0]]))
                            acc_rank =get_accuracy_by_rankcorr(real_v=v, gen_v=np.reshape(self.scores, [self.scores.shape[0]]))
                            if verbose:
                                print("epoch %d, training_accuracy %g, score corr %g, global ranking corr %g"\
                                      %(epoch, train_accuracy_all, acc_stat, acc_rank))
                    feed_dict = {self.X:dataset[0], 
                                 self.y_:dataset[1],
                                 self.pref_holder:dataset[2]}
                    loss_val = self.loss.eval(session = self.sess,
                                               feed_dict=feed_dict)
                    train_accuracy = self.accuracy_bin.eval(session = self.sess,
                                               feed_dict=feed_dict)
                    self.last_loss = loss_val
                    self.last_acc  = train_accuracy
                    if save_dir != None:
                        self.saver.save(self.sess, save_path=save_dir) 
                    if verbose:
                        print('train_acc:', train_accuracy, 'loss:', loss_val)
                else:
                    tf.train.Saver().restore(self.sess, save_path=save_dir)
                    Theta_pre = self.sess.run(self.Theta)
                    feed_dict = {self.X:dataset[0], 
                                 self.y_:dataset[1],
                                 self.pref_holder:dataset[2]}
                    loss_val = self.loss.eval(session = self.sess,
                                               feed_dict=feed_dict)
                    train_accuracy = self.accuracy_bin.eval(session = self.sess,
                                               feed_dict=feed_dict)
                    self.last_loss = loss_val
                    self.last_acc  = train_accuracy
                    if verbose:
                        print(train_accuracy, loss_val)
    def getROC(self, save_dir, dataset):
        with self.g_rank.as_default():
            with self.sess.as_default(): 
                # self.sess.run(tf.global_variables_initializer())
                # saver = tf.train.Saver()
                # tf.train.Saver().restore(self.sess, save_path=save_dir)
                # Use this model to get ROC
                feed_dict = {self.X:dataset[0], 
                             self.y_:dataset[1],
                             self.pref_holder:dataset[2]}
                prediction_value = self.y_out.eval(session = self.sess,
                                               feed_dict=feed_dict)
                labels = dataset[1] # pref
                fpr,tpr,threshold = roc_curve(labels[:, 0], prediction_value[:, 0])
                roc_auc = auc(fpr,tpr) ###计算auc的值

                lw = 2
                plt.figure(figsize=(10,10))
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.show()
        return prediction_value, labels, roc_auc
    def getPR(self, save_dir, dataset):
        pass 