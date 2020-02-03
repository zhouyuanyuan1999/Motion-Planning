
# coding: utf-8

# In[10]:


import gym
from gym import spaces
import numpy as np


class Planner:

    '''
    Initialization of all necessary variables to generate a policy:
        discretized state space
        control space
        discount factor
        learning rate
        greedy probability (if applicable)
    '''
    def __init__(self, env, epsilon, density, gamma, alpha):
        
        #get control space
        self.controls = np.arange(env.action_space.n)
        
        #Initialize discreteized state space 
        obs = env.observation_space
        self.obs = obs
        x = np.linspace(obs.low[0],obs.high[0],density)
        y = np.linspace(obs.low[1],obs.high[1],density)
        x, y = np.meshgrid(x,y)
        x = x.flatten()
        x = x.reshape(x.shape[0],1)
        y = y.flatten()
        y = y.reshape(y.shape[0],1)
        self.states = np.hstack((x,y)).reshape(density,density,2)
        
        #get epsilon for greedy
        self.epsilon_i = epsilon
        self.epsilon = epsilon
        
        #get discount
        self.gamma = gamma
        
        #alpha
        self.alpha = alpha
        
        #initalize Q space, initalize to all zeros
        self.Q = np.zeros((density,density,self.controls.shape[0]))
        
        #N(x): store number of visits to states
        #self.N = np.zeros((density,density))
        
        #policy
        self.policy = np.zeros((density,density))
        
        #density
        self.density = density
        
        #used to plot Q(x,u)
        self.q_s1 = []
        self.q_s2 = []
        self.q_s3 = []
        

    '''
    Learn and return a policy via model-free policy iteration.
    '''
    def __call__(self, mc=False, on=True):
        return self._td_policy_iter(on)

    #This function find Q value position/index given state and action
    def find_Q_idx (self, s, a):
        sum_ = np.sum(np.absolute(self.states-s.reshape(2,)),axis=2)
        argmin = np.argmin(sum_)
        idx = np.array([np.floor(argmin/sum_.shape[0]), np.mod(argmin,sum_.shape[0])]).astype(int)
        return np.array([idx[0],idx[1],a])
        
    #Update Q function
    def q_update(self,s,a,r_,s_,a_):
        idx = self.find_Q_idx(s,a)
        #alpha = self.N[idx[0],idx[1]] / (1 + self.N[idx[0],idx[1]])
        idx_ = self.find_Q_idx(s_,a_)
        q = self.Q[idx[0],idx[1],idx[2]]
        q_ = np.max(self.Q[idx_[0],idx_[1]])
        alpha = self.alpha
        self.Q[idx[0],idx[1],idx[2]] = q + alpha*(r_ + self.gamma*q_ - q)
        return np.absolute(alpha*(r_ + self.gamma*q_ - q))
    
    #Update Q function at terminal state
    def q_update_terminal(self,s,a,r_):
        idx = self.find_Q_idx(s,a)
        #alpha = self.N[idx[0],idx[1]] / (1 + self.N[idx[0],idx[1]])
        q = self.Q[idx[0],idx[1],idx[2]]
        alpha = self.alpha
        self.Q[idx[0],idx[1],idx[2]] = q + alpha*(r_ - q)
        return np.absolute(alpha*(r_ - q))
    
        
    #Update policy given Q
    def update_policy(self):
        self.policy = np.argmax(self.Q, axis=2)
    
    #Pick epsilon greedy action
    def greedy_action(self,x):
        if np.random.rand() < (1-self.epsilon):
            idx = self.find_Q_idx(x,0)
            return self.policy[idx[0],idx[1]]
        else:
            return np.random.randint(3)
    
    #single q-learning episode
    def q_episode(self, render = False):
        #track the change of Q
        delta = 0
        t = 0
        s = env.reset()
        done = False
        self.update_policy()
        while True:
            a = self.greedy_action(s)
            if render:
                env.render()
            s_, r_, done, _ = env.step(a)
                
            a_ = self.greedy_action(s_)
            if done == True and t < 199:
                print('update terminal')
                delta += self.q_update_terminal(s,a,r_)
            else:
                delta += self.q_update(s,a,r_,s_,a_)
            s = s_
            
            if (done == True or t == 199):
                print("Success "+str(t))
                return delta
                break
            t = t+1
        return delta
            
    def test(self, render = False):
        self.Q = np.load('off_q.npy')
        t = 0
        s = env.reset()
        s[1] = s[1]*10
        print(s)
        done = False
        self.update_policy()
        while True:
            a = self.greedy_action(s)
            if render:
                env.render()
            s_, r_, done, _ = env.step(a)
                
            a_ = self.greedy_action(s_)
            s = s_
            
            if (done == True or t == 199):
                print("Success "+str(t))
                break
            t = t+1

    '''
    TO BE IMPLEMENT
    TD Policy Iteration
    Flags: on : on vs. off policy learning
    Returns: policy that minimizes Q wrt to controls
    '''
    def _td_policy_iter(self,train_on=True):
        if train_on == False:
            for i in range(10):
                print('ep'+str(i))
                self.epsilon = 0
                self.Q = np.load('off_q.npy')
                self.test(True)
            return
            
        #Q
        print(self.states)
        print(self.controls)
        print(self.controls.shape[0])
        
        #used to plot
        s1 = np.array([0,0])
        s2 = np.array([-1,0.05])
        s3 = np.array([0.25,-0.05])
        s1 = self.find_Q_idx (s1, 0)[:2]
        s2 = self.find_Q_idx (s2, 0)[:2]
        s3 = self.find_Q_idx (s3, 0)[:2]
        self.s1_idx = s1.copy()
        self.s2_idx = s2.copy()
        self.s3_idx = s3.copy()
        
        i = 0
        delta = 100
        while (i<10000) and delta > 0.01:
            self.epsilon = self.epsilon_i/(i+1)
            self.q_s1.append(self.Q[s1[0],s1[1]].copy())
            self.q_s2.append(self.Q[s2[0],s3[1]].copy())
            self.q_s3.append(self.Q[s2[0],s3[1]].copy())
            delta = self.q_episode(False)
            print("ep"+str(i)+' '+str(delta))
            #print("ep"+str(i))
            i = i+1
        print('total episodes: ' + str(i))
        np.save('off_q.npy',self.Q)
        print(delta)
        self.epsilon = 0
        self.q_episode(False)
        
        #Z = np.max(self.Q.reshape(density*density,3), axis = 1)
#         Z = -np.max(self.Q, axis = 2)
#         obs = self.obs
#         pos = np.linspace(obs.low[0],obs.high[0],self.density)
#         v = np.linspace(obs.low[1]*10,obs.high[1]*10,self.density)
#         pos, v = np.meshgrid(pos,v) 
#         ax = plt.axes(projection='3d')
#         ax.plot_wireframe(pos, v, Z)
#         plt.savefig('off_q_graph.png')


if __name__ == '__main__':
    epsilon = 100
    density = 30
    gamma = 1
    alpha = 0.1
    env = gym.make('MountainCar-v0')
    planner = Planner(env,epsilon,density,gamma,alpha)
    planner._td_policy_iter(True)

#Used to Plot, No need to care
# import matplotlib.pyplot as plt
# s1 = -1*np.asarray(planner.q_s1)
# s2 = -1*np.asarray(planner.q_s2)
# s3 = -1*np.asarray(planner.q_s3)
# print(s1.shape)
# t = np.arange(s1.shape[0])
# print(t.shape)
# print(s1.shape)
# for i in range(3):
#     plt.plot(t,s1[:,i])
#     plt.title('Q('+str([0,0,i])+')')
#     plt.savefig(str(i)+'_tdq_s1.png')
#     plt.show()
# for i in range(3):
#     plt.plot(t,s2[:,i])
#     plt.title('Q('+str([-1,0.05,i])+')')
#     plt.savefig(str(i)+'_tdq_s2.png')
#     plt.show()
# for i in range(3):
#     plt.plot(t,s3[:,i])
#     plt.title('Q('+str([0.25,-0.05,i])+')')
#     plt.savefig(str(i)+'_tdq_s3.png')
#     plt.show()
# plt.imshow(planner.policy)
# plt.colorbar()
# plt.title('td off-policy')
# plt.savefig('td_q_off_policy.png')
# plt.show()

