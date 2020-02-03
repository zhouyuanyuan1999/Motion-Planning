
# coding: utf-8

# In[39]:


import gym
from gym import spaces
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
        self.pos = x
        y = np.linspace(obs.low[1]*10,obs.high[1]*10,density)
        self.vel = y
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
    def find_Q_idx1 (self, s, a):
        sum_ = np.sum(np.absolute(self.states-s.reshape(2,)),axis=2)
        argmin = np.argmin(sum_)
        idx = np.array([np.floor(argmin/sum_.shape[0]), np.mod(argmin,sum_.shape[0])]).astype(int)
        return np.array([idx[0],idx[1],a])
        
    def find_Q_idx (self, s, a):
        p = np.argmin(np.absolute(self.pos - s[0]))
        v = np.argmin(np.absolute(self.vel - s[1]))
        return np.array([v,p,a])
    
    #Update Q function for sarsa
    def sarsa_update(self,s,a,r_,s_,a_):
        idx = self.find_Q_idx(s,a)
        #alpha = self.N[idx[0],idx[1]] / (1 + self.N[idx[0],idx[1]])
        idx_ = self.find_Q_idx(s_,a_)
        q = self.Q[idx[0],idx[1],idx[2]]
        q_ = self.Q[idx_[0],idx_[1],idx_[2]]
        alpha = self.alpha
        self.Q[idx[0],idx[1],idx[2]] = q + alpha*(r_ + self.gamma*q_ - q)
        return np.absolute(alpha*(r_ + self.gamma*q_ - q))
    
    #Update Q function at terminal state
    def sarsa_update_terminal(self,s,a,r_):
        idx = self.find_Q_idx(s,a)
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
    
    #single SARSA episode
    def sarsa_episode(self, render = False):
        #track the change of Q
        delta = 0
        
        t = 0
        s = env.reset()
        s[1] = s[1]*10
        done = False
        self.update_policy()
        while True:
            a = self.greedy_action(s)
            if render:
                env.render()
            s_, r_, done, _ = env.step(a)
            s_[1] = s_[1]*10
            a_ = self.greedy_action(s_)
            if done == True and t < 199:
                print('update terminal')
                delta += self.sarsa_update_terminal(s,a,r_)
            else:
                delta += self.sarsa_update(s,a,r_,s_,a_)
            s = s_
            
            if (done == True or t == 199):
                print("Success "+str(t))
                return delta
                break
            t = t+1
            
    
    def test(self, render = False):
        self.Q = np.load('sarsa_q.npy')
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
            s_[1] = s_[1]*10
                
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
                self.Q = np.load('sarsa_q.npy')
                self.test(True)
            return
            
        #SARSA
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
        while (i<10000) and delta > 0.001:
            self.epsilon = self.epsilon_i/(i+1)
            #delta = self.sarsa_episode(False)
            #used to plot
            self.q_s1.append(self.Q[s1[0],s1[1]].copy())
            self.q_s2.append(self.Q[s2[0],s3[1]].copy())
            self.q_s3.append(self.Q[s2[0],s3[1]].copy())
            delta = self.sarsa_episode(False)
            print("ep"+str(i)+' '+str(delta))
            i = i+1
        print('total episodes: ' + str(i))
        np.save('sarsa_q.npy',self.Q)
        print(delta)
        self.epsilon = 0
        self.sarsa_episode(False)
        
        #Z = np.max(self.Q.reshape(density*density,3), axis = 1)
        Z = -np.max(self.Q, axis = 2)
        obs = self.obs
        pos = np.linspace(obs.low[0],obs.high[0],self.density)
        v = np.linspace(obs.low[1]*10,obs.high[1]*10,self.density)
        pos, v = np.meshgrid(pos,v) 
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(pos, v, Z)
        plt.savefig('q_graph.png')

if __name__ == '__main__':
    epsilon = 100
    density = 30
    gamma = 1
    alpha = 0.1
    env = gym.make('MountainCar-v0')
    planner = Planner(env,epsilon,density,gamma,alpha)
    planner._td_policy_iter(True)


# epsilon = 1
# density = 50
# gamma = 0.9
# alpha = 0.3
# total episodes: 8129
# 0.000919705144430871
#Used to Plot, No need to care
# import matplotlib.pyplot as plt
# s1 = -1*np.asarray(planner.q_s1)
# s2 = -1*np.asarray(planner.q_s2)
# s3 = -1*np.asarray(planner.q_s3)
# print(s1.shape)
# t = np.arange(s1.shape[0])
# for i in range(3):
#     plt.plot(t,s1[:,i])
#     plt.title('Q('+str([0,0,i])+')')
#     plt.savefig(str(i)+'_sarsa_s1.png')
#     plt.show()
# for i in range(3):
#     plt.plot(t,s2[:,i])
#     plt.title('Q('+str([-1,0.05,i])+')')
#     plt.savefig(str(i)+'_sarsa_s2.png')
#     plt.show()
# for i in range(3):
#     plt.plot(t,s3[:,i])
#     plt.title('Q('+str([0.25,-0.05,i])+')')
#     plt.savefig(str(i)+'_sarsa_s3.png')
#     plt.show()
# plt.imshow(planner.policy)
# plt.colorbar()
# plt.title('sarsa-policy')
# plt.savefig('sarsa_policy.png')
# plt.show()

