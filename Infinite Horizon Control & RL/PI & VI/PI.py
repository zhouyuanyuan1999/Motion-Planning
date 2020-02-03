
# coding: utf-8

# In[23]:


"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal

class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''
    def __init__(self):
        
        #Hyperparameters
        self.dt_itr = 1
        self.a,self.b,self.sigma,self.k,self.r,self.gamma = 1,0.8,0.1,100,1,0.9
        self.dt = 0.05
        self.u_max = 1
        self.theta = np.pi
        n1,n2,n_u = 100,50,30
        self.v_max = 3
        self.n1,self.n2,self.n_u = n1,n2,n_u
        #Initialize state space, control space, V values
        self.s1 = np.linspace(0,2*self.theta,n1)
        self.s2 = np.linspace(-1*self.v_max,self.v_max,n2)
        self.u = np.linspace(-1*self.u_max,self.u_max,n_u)
        self.V = np.zeros((n1,n2))
        self.oldV = np.zeros((n1,n2))
        self.policy = np.zeros((n1,n2))
        #states
        x,y = np.meshgrid(self.s2,self.s1)
        x = x.flatten()
        y = y.flatten()
        x = x.reshape(x.shape[0],1)
        y = y.reshape(y.shape[0],1)
        self.states = np.hstack((y,x))
        self.states = self.states.reshape(n1,n2,2)
    
    def init_pf (self,n1,n2,n_u,dt):
        print('Start Calculate p_f for all x and u')
        print('Total i is '+str(n1))
        #initalize probability
        self.p_f = np.zeros((n1,n2,n_u,n1,n2))
        for i in range(n1):
            print('Calculating p_f '+str(i))
            for j in range(n2):
                for k in range(n_u):
                    angle = self.s1[i] + (self.s2[j]*dt)
                    v = self.s2[j] + (self.a*np.sin(self.s1[i]) - self.b*self.s2[j] + self.u[k])*dt
                    angle = angle%(2*np.pi)
                    if v > self.v_max:
                        v = self.v_max
                    elif v < -1*self.v_max:
                        v = -1*self.v_max
#                     if v < self.v_max and v > -1*self.v_max:
                    if True:
                        mu = np.array([angle,v])
                        cov = np.eye(2)*self.sigma
                        cov = cov.dot(cov.T)*dt
                        states1 = np.absolute(self.states - mu)
                        states = (np.array([2*np.pi,0]) - np.absolute(self.states - mu))
                        states = np.absolute(states)
                        states[:,:,0] = np.where(states1[:,:,0]>states[:,:,0],states[:,:,0],states1[:,:,0])
                        p = multivariate_normal.pdf(states, mean=[0,0], cov=cov)
                        #p = multivariate_normal.pdf(self.states, mean=mu, cov=cov)
                        p = p/np.sum(p)
                        self.p_f[i,j,k] = p
        print('p_f finished')
    
    def policy_itr (self,n1,n2,n_u,dt):
        #used to plot
        self.ss1 = np.array([np.int(n1/2),np.int(n2/2)])
        self.ss2 = np.array([np.int(n1/2),0])
        self.ss3 = np.array([np.int(n1/4),np.int(n2/4)])
        self.v_s1 = []
        self.v_s2 = []
        self.v_s3 = []
        
        self.init_pf(n1,n2,n_u,dt)
        #Policy Iteration Start
        print('Policy Iteration Start')
        print('[iteration #, error]')
        itr = 0
        error = 1
        while(itr<100 and error > 0.001):
            #used to plot
            self.v_s1.append(self.V[self.ss1[0],self.ss1[1]].copy())
            self.v_s2.append(self.V[self.ss2[0],self.ss2[1]].copy())
            self.v_s3.append(self.V[self.ss3[0],self.ss3[1]].copy())
            
            #get linear equations
            a = np.eye(n1*n2)
            b = np.zeros((n1*n2,))
            for i in range(n1):
                for j in range(n2):
                    k = np.int(self.policy[i,j])
                    l = (1 - np.exp(self.k*np.cos(self.s1[i]) - self.k) + self.r*self.u[k]*self.u[k]/2)*dt
                    b[i*n2 + j] = l
                    p = self.p_f[i,j,k].flatten()
                    p = -1*self.gamma*p
                    a[i*n2 + j] = a[i*n2 + j] + p
            self.V = np.linalg.solve(a,b).reshape(n1,n2)
            #update policy
            for i in range(n1):
                for j in range(n2):
                    cost = np.zeros((n_u,))
                    for k in range(n_u):
                        l = (1 - np.exp(self.k*np.cos(self.s1[i]) - self.k) + self.r*self.u[k]*self.u[k]/2)*dt
                        cost[k] = l + self.gamma*(np.sum(self.V*self.p_f[i,j,k]))
                    self.policy[i,j] = np.int(np.argmin(cost))
            itr += 1
            error = np.sum(np.absolute(self.oldV - self.V))
            print([itr, error])
            self.oldV = self.V.copy()
        print('Policy Iteration Finished')
        im = plt.imshow(self.V)
        plt.colorbar()
        plt.show()
        #plt.savefig('value_PI.png')
        for i in range(n1):
            for j in range(n2):
                self.policy[i,j] = self.u[np.int(self.policy[i,j])]
        img = plt.imshow(self.policy)
        plt.colorbar()
        plt.show()
        #plt.savefig('policy_PI.png')
                   
    def interpolation(self,dt):
        self.dt = 0.05
        self.t = np.arange(0.0, 5.0, self.dt)
        self.u_copy = self.u.copy()
        self.theta = []
        self.u = []
        x1_i,x2_i = 3.14,0
        for i in range(self.t.shape[0]):
            idx1 = np.int(np.argmin(np.absolute(x1_i - self.s1)))
            idx2 = np.int(np.argmin(np.absolute(x2_i - self.s2)))
            
            control = self.policy[idx1,idx2]
            mu = np.array([x2_i,self.a*np.sin(x1_i)-self.b*x2_i+control])*dt + np.array([x1_i,x2_i])
            cov = np.eye(2)*self.sigma
            cov = cov.dot(cov.T)*dt
            noise = np.random.multivariate_normal(mean=[0,0], cov=cov)
            s = mu + noise
            s[0] = s[0]%(2*np.pi)
            if s[1] > self.v_max:
                s[1] = self.v_max
            elif s[1] < -1*self.v_max:
                s[1] = -1*self.v_max

            x1_i = s[0].copy()
            x2_i = s[1].copy()
            self.theta.append(s[0])
            self.u.append(control)
       
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)

    '''
    Provide new rollout theta values to reanimate
    '''
    def new_data(self, theta):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = np.zeros(t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        #policy_iteration
        self.policy_itr(self.n1,self.n2,self.n_u,self.dt_itr)
        #iterpolation/generate policy
        self.interpolation(self.dt_itr)
        
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        plt.show()


if __name__ == '__main__':
    animation = EnvAnimate()
    animation.start()

#Used to Plot, No need to Care
# plt.figure(6)
# plt.imshow(animation.V)
# plt.colorbar()
# plt.title('VI-Value')
# plt.savefig('p4_pi_value.png')
# plt.show()

# plt.figure(7)
# plt.imshow(animation.policy)
# plt.colorbar()
# plt.title('VI-policy')
# plt.savefig('p4_pi_policy.png')
# plt.show()

