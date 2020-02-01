import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt

T =100 #number of moves
U = [0,1,2] #[rock, paper, scissor]
N_ = 11 #density, number of total discretized belief states = (N_ choose 2)

#Discretize belief state based on S_r,S_p,S_s
b_ = np.linspace(0, 1, N_)
B = np.unique(np.array([[r,p,1-r-p] for r in b_ for p in b_]),axis=0)
B = B[B.min(axis=1)>-0.000001]
print("Discretized belief state shape is "+str(B.shape))
#define a function that calculate Pr(Z_t = [r,p,s] | b_t)
def prZ_t (b_t):
    return np.array([(0.5*b_t[0] + 0.25*b_t[1] + 0.25*b_t[2]),(0.25*b_t[0] + 0.5*b_t[1] + 0.25*b_t[2]),(0.25*b_t[0] + 0.25*b_t[1] + 0.5*b_t[2])])
#define a function that map continous belief state to one of the discretized belief states
def discretize (b_t):
    return B[np.argmin(np.sum(np.absolute(b_t-B),axis=1))]
#define a function that find out discretized belief state b_(t+1) given b_t and Z_t and current time (number of games have played)
def b_t1 (b_t,Z_t,t):
    b_t1 = b_t*t
    b_t1[Z_t] = b_t1[Z_t] + 1
    return discretize(b_t1/(t+1))
#define possible scores
Score = []
for i in range(T+1):
    Score.append(np.array([s for s in range(-(i),i+1)]))

#define state X based on possible scores, discretized belief
#x = [b,score]
X = []
X.append(np.array([[1/3,1/3,1/3,0]]))
for i in (np.arange(T)+1):
    x=[]
    for score in Score[i]:
        for j in range(B.shape[0]):
            x.append(np.array([B[j][0],B[j][1],B[j][2],score]))
    x = np.asarray(x)
    #np.array( [[S[i][k][0],S[i][k][1],S[i][k][2],Score[i][j]] for k in range(S[i].reshape[0]) for j in range(Score[i].shape[0])] )
    X.append(x)
    
#define a function that find the matched state index at time t given discretized belief state and score
def find_state(b,s,t):
    x = [b[0],b[1],b[2],s]
    return np.argmin(np.sum(np.absolute(x-X[t]),axis=1))
#define a function that calculate the score based on your action
def s_t1 (s,Z_t,u_t):
    if(Z_t == 0):
        if(u_t == 0):
            return s
        if(u_t == 1):
            return s+1
        if(u_t == 2):
            return s-1
    if(Z_t == 1):
        if(u_t == 0):
            return s-1
        if(u_t == 1):
            return s
        if(u_t == 2):
            return s+1
    if(Z_t == 2):
        if(u_t == 0):
            return s+1
        if(u_t == 1):
            return s-1
        if(u_t == 2):
            return s
#define a function that find out V^(t+1) index 
def V_index (b,s,t,z,u):
    s_next = s_t1(s,z,u)
    b_next = b_t1(b,z,t)
    return find_state(b_next,s_next,t+1) 

if __name__=="__main__":
    #This part takes me around 30 minutes to run. If you do not want to run it, you can just use my saved results.
    yes = int(sys.argv[1])

    if yes == 0:
        #Start MDP
        V = [] #cost
        Pi = [] #optimal policy
        #add terminal cost to V
        V.append(np.array([q[3] for q in X[T]]))
        #compute all reward and optimal policy except the state state
        for i in (T-1-np.arange(T-1)):
            f = []
            for x in X[i]:
                b_t = x[:3]
                p = prZ_t(b_t)
                f_i = np.zeros((3,))
                for u in [0,1,2]:
                    f_i[u] = p[0]*V[T-1-i][V_index(b_t,x[3],i,0,u)] + p[1]*V[T-1-i][V_index(b_t,x[3],i,1,u)]+p[2]*V[T-1-i][V_index(b_t,x[3],i,2,u)]
                f.append(f_i)
            f=np.asarray(f)
            V.append(np.max(f,axis=1))
            Pi.append(np.argmax(f,axis=1))
        #for cost and optimal policy at state 0
        f_0 = np.zeros((3,))
        f_0[0] = 1/3*V[T-1][find_state([1,0,0],0,1)] + 1/3*V[T-1][find_state([0,1,0],-1,1)]+1/3*V[T-1][find_state([0,0,1],1,1)]
        f_0[1] = 1/3*V[T-1][find_state([0,1,0],0,1)] + 1/3*V[T-1][find_state([0,0,1],-1,1)]+1/3*V[T-1][find_state([1,0,0],1,1)]
        f_0[2] = 1/3*V[T-1][find_state([0,0,1],0,1)] + 1/3*V[T-1][find_state([1,0,0],-1,1)]+1/3*V[T-1][find_state([0,1,0],1,1)]
        f_0 = [f_0]
        Pi.append(np.argmax(f_0,axis=1))
        V.append(np.max(f_0,axis=1))
    elif yes == 1:
        V = np.load('Cost.npy')
        Pi = np.load('Optimal Policy.npy')
    else:
        print("Wrong Argument, See README")
        sys.exit()
    
    #Test
    G = 50 # number of trails

    Score_det = 0
    Score_st = 0
    Score_opt = 0

    #define a function that determines win or lose or draw
    def score_n (Z_t,u_t):
        if(Z_t == 0):
            if(u_t == 0):
                return 0
            if(u_t == 1):
                return 1
            if(u_t == 2):
                return -1
        if(Z_t == 1):
            if(u_t == 0):
                return -1
            if(u_t == 1):
                return 0
            if(u_t == 2):
                return 1
        if(Z_t == 2):
            if(u_t == 0):
                return 1
            if(u_t == 1):
                return -1
            if(u_t == 2):
                return 0
    #used to store every trail score
    s_det = np.zeros((G,))
    s_st = np.zeros((G,))
    s_opt = np.zeros((G,))

    #deterministic play
    U_det = np.remainder(np.arange(T), 3)

    for g in range(G):
        #random generate opponent move with bias towards Paper
        Opp = np.random.randint(4, size=T)
        for i in range(T):
            if Opp[i] == 0:
                Opp[i] = 1
            elif Opp[i] == 1:
                Opp[i] = 1
            elif Opp[i] == 2:
                Opp[i] = 0
            elif Opp[i] == 3:
                Opp[i] = 2

        #determistic play
        for i in range(T):
            Score_det = Score_det + score_n(Opp[i],U_det[i])
        s_det[g] = Score_det
        Score_det = 0

        #stochastic play
        U_st = np.random.randint(3, size=T)
        for i in range(T):
            Score_st = Score_st + score_n(Opp[i],U_st[i])
        s_st[g] = Score_st
        Score_st = 0

        #optimal play
        actions = np.zeros((3,))
        U_opt = [Pi[T-1][0]]
        Score_opt = Score_opt + score_n(Opp[0],U_opt[0])
        for i in (np.arange(T-1)+1):
            #find current state
            actions[Opp[i-1]] = actions[Opp[i-1]] + 1
            b = actions/i
            index = find_state(b,Score_opt,i)
            U_opt.append(Pi[len(Pi)-(i+1)][index])
            Score_opt = Score_opt + score_n(Opp[i],U_opt[i])
        s_opt[g] = Score_opt
        Score_opt = 0
        
    #plot the means
    x = np.arange(G)+1
    y_det = np.zeros((G,)) 
    for i in range(G):
        y_det[i] = np.sum(s_det[:(i+1)])/(i+1)
    y_st = np.zeros((G,)) 
    for i in range(G):
        y_st[i] = np.sum(s_st[:(i+1)])/(i+1)
    y_opt = np.zeros((G,)) 
    for i in range(G):
        y_opt[i] = np.sum(s_opt[:(i+1)])/(i+1)

    plt.plot(x,y_det)
    plt.plot(x,y_st)
    plt.plot(x,y_opt)

    plt.legend(['Determinstic', 'Stochastic', 'Optimal Policy'], loc='upper left')

    plt.xlabel('Number of Games')
    plt.ylabel('Means of Scores')
    plt.show()

    #plot standard deviation
    x = np.arange(G)+1
    y_det = np.zeros((G,)) 
    for i in range(G):
        y_det[i] = np.std(s_det[:(i+1)])
    y_st = np.zeros((G,)) 
    for i in range(G):
        y_st[i] = np.std(s_st[:(i+1)])
    y_opt = np.zeros((G,)) 
    for i in range(G):
        y_opt[i] = np.std(s_opt[:(i+1)])

    plt.plot(x,y_det)
    plt.plot(x,y_st)
    plt.plot(x,y_opt)

    plt.legend(['Determinstic', 'Stochastic', 'Optimal Policy'], loc='upper left')

    plt.xlabel('Number of Games')
    plt.ylabel('Standard deviations of Scores')
    plt.show()


