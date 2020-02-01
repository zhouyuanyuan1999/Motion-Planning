import numpy as np
import networkx as nx
import time

class RobotPlanner:
  __slots__ = ['boundary', 'blocks','explored','explored_h','G', 'path','move','smallest','firsttime','x_lastest','start_node','graph_finished','node_a','first_path','path_finished','stepsize']

  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks
    print(boundary)
    print(blocks)
    #used for LRTA*
    self.explored = []
    self.explored_h = []
    #used for the graph of RRT*
    self.G = nx.Graph()
    self.path = []
    self.move = 0
    self.firsttime = True
    self.x_lastest = []
    self.start_node = []
    self.graph_finished = False
    self.path_finished = False
    self.node_a = []
    self.first_path = True
    #used to check collision free
    self.smallest = np.min(np.absolute(blocks[:,:3] - blocks[:,3:6]))
    print(self.smallest)
  
  

  #this function checks if the node overstep the boundary or block
  # node should be (3,)
  def safe(self,node):
    node = node.reshape(1,3)
    if( (np.max(node <= self.boundary[0,:3]) == 1) or (np.max(node >= self.boundary[0,3:6]) == 1) ):
        return False
    for k in range(self.blocks.shape[0]):
        if( (np.sum(node >= self.blocks[k,:3]) == 3) and (np.sum(node <= self.blocks[k,3:6]) == 3) ):
            return False
    return True
  
  def collision_free (self, node1, node2):
      node1 = node1.reshape(3,)
      node2 = node2.reshape(3,)
      vector = (node2-node1)
      max_value = np.sum(vector**2)**0.5
      vector = vector/max_value
      
      lists = []
      itr = 0
      while(True):
           itr = itr+1
           d = itr*self.smallest
           if(d>=max_value):
                break;
           lists.append(node1 + (d*vector))
           lists.append(node2)
      for node in lists:
          for k in range(self.blocks.shape[0]):
              if( (np.sum(node >= self.blocks[k,:3]) == 3) and (np.sum(node <= self.blocks[k,3:6]) == 3) ):
                  return False
      return True

  #this function find all eight neighbors surround the node
  #smaller the d is, more dense the graph is. d represents how big one step is.
  def neighbors2(self,node, d = 0.5):
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))*d
    dR = np.delete(dR,13,axis=1)

    nei = np.array([node])
    nei = np.repeat(nei,26, axis=0) 
    nei = nei + dR.T
    

    return nei
  
  def neighbors(self, node, d = 0.5):
    dR = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    nei = np.array([node])
    nei = np.repeat(nei,6, axis=0)
    nei = nei + dR*d
    return nei

#   #return h value of the node
#   def h_node(self,node,goal):
#       if node.tolist() in self.explored:
#           return slef.explored_g[self.explored.index(node.tolist())]
#       return np.sum( (node-goal)**2 )**0.5

  #get all nei and cost
#   def nei_f (self, start, expanded):
#       neigh = self.neighbors(start, d=0.5)
#       neigh_index = []
#       for i in range(neigh.shape[0]):
#           if self.safe(neigh[i]) and self.collision_free(neigh[i],start) and (not (neigh[i].tolist() in expanded)):
#               neigh_index.append(i)
#       neigh = neigh[neigh_index,:]
#       C_sj = np.sum( (neigh - start.reshape(1,3))**2, axis=1 )**0.5
#       h_j = []
#       exp = np.asarray(self.explored)
#       for i in range(neigh.shape[0]):
#           if np.min(np.sum(np.absolute(neigh[i].reshape(1,3) - exp))):
#             h_j.append( self.explored_h[exp.index(neigh[i].tolist())] )
#           else:
#             h_j.append( np.sum( (neigh[i] - goal.reshape(1,3))**2)**0.5 )
#       f = C_sj + np.asarray(h_j)
#       return [neigh, f]

  def in_explored(self, node):
    if len(self.explored) > 0:
        node = node.reshape(1,3)
        d = np.linalg.norm((np.asarray(self.explored) - node), axis = 1)
        if np.min(d) < 0.00001:
            return np.argmin(d)
    return -1

  #For fast runtime, I decide to use N=1 for LRTA*
  def plan_fast(self,start,goal):
    #check if goal can be reached in one step
    if (np.sum((start.reshape(3,)-goal)**2) < 1 and self.collision_free(start,goal)):
        return goal
    
    #update and move
    neigh = self.neighbors(start, d=0.3)
    neigh_index = []
    for i in range(neigh.shape[0]):
        if self.safe(neigh[i]) and self.collision_free(neigh[i],start):
            neigh_index.append(i)
    neigh = neigh[neigh_index,:]
    
    h_j = []
    for i in range(neigh.shape[0]):
        idx = self.in_explored(neigh[i])
        if idx >= 0:
            h_j.append( self.explored_h[idx] )
        else:
            h_j.append( np.sum( (neigh[i] - goal.reshape(1,3))**2)**0.5 )

    f = np.asarray(h_j)
    idx = self.in_explored(start)
    if idx >= 0:
        if np.min(f) > self.explored_h[idx]:
            self.explored_h[idx] = np.min(f) + 0.5
    else:
        self.explored.append(start)
        if np.min(f) > np.sum((start - goal)**2)**0.5:
            self.explored_h.append(np.min(f)+0.5)
        else:
            self.explored_h.append(np.sum((start - goal)**2)**0.5)

    newpos = neigh[np.argmin(f)]
    return newpos

#For fast runtime, I decide to use N=1 for LRTA*
  def plan(self,start,goal):
    #check if goal can be reached in one step
    if (np.sum((start.reshape(3,)-goal)**2) < 1 and self.collision_free(start,goal)):
        return goal

    #update and move
    neigh = self.neighbors(start, d=0.3)
    neigh_index = []
    for i in range(neigh.shape[0]):
        if self.safe(neigh[i]):
            neigh_index.append(i)
    neigh = neigh[neigh_index,:]
    C_sj = np.sum( (neigh - start.reshape(1,3))**2, axis=1 )
    h_j = []
    exp = np.asarray(self.explored).tolist()
    for i in range(neigh.shape[0]):
        if neigh[i].tolist() in exp:
            h_j.append( self.explored_h[exp.index(neigh[i].tolist())] )
        else:
            h_j.append( np.sum( (neigh[i] - goal.reshape(1,3))**2) )

    f = C_sj + np.asarray(h_j)
    if start.tolist() in exp:
        self.explored_h[exp.index(start.tolist())] = np.min(f)
    else:
        self.explored_h.append(np.min(f))
        self.explored.append(start)

    newpos = neigh[np.argmin(f)]
    return newpos

#Functions for RRT
  def x_rand (self):
      x_coor = np.random.uniform(self.boundary[0,0],self.boundary[0,3])
      y_coor = np.random.uniform(self.boundary[0,1],self.boundary[0,4])
      z_coor = np.random.uniform(self.boundary[0,2],self.boundary[0,5])
#       while(self.safe(np.array([x_coor,y_coor,z_coor])) == False):
#         x_coor = np.random.uniform(self.boundary[0,0],self.boundary[0,3])
#         y_coor = np.random.uniform(self.boundary[0,1],self.boundary[0,4])
#         z_coor = np.random.uniform(self.boundary[0,2],self.boundary[0,5])
      return np.array( [[x_coor,y_coor,z_coor]] )

#This function returns index of the nearest node
  def nearest (self, node):
      node = node.reshape(1,3)
      ver = np.asarray(list(self.G.nodes))
      dist = np.sum((ver-node)**2, axis=1)
      return [ver[np.argmin(dist)], np.min(dist)]

#This function return the steered point with epsilon distance
  def steer(self, eps, near, node_random):
      near = near.reshape(3,)
      vector = node_random - near
      vector = (vector/np.sum(vector**2))*eps
      return (near+vector)

#Used to track time
  def tic(self):
    return time.time()

#Plan function for RRT based algorithm
  def plan_RRT(self,start,goal):
    if(self.graph_finished == False):
      if(self.firsttime):
        start_node = tuple(start.reshape(3,).tolist())
        self.G.add_node(start_node, cost = 0, parent = start_node)
        self.x_lastest = start
        self.firsttime = False
        self.start_node = start_node
        return start
      
      t0 = self.tic()
      print("graph")
      epsilon = 0.3
      x_lastest = self.x_lastest
      while not ( np.sum((x_lastest.reshape(3,)-goal)**2) < 0.98 and self.collision_free(x_lastest,goal) ):
        x_random  = self.x_rand()
        [x_near, dist] = self.nearest(x_random)
        if(dist > 0.001 and dist > epsilon):
#             if(dist <= epsilon):
#                 x_new = x_random
            if (dist > epsilon):
                x_new = self.steer(epsilon, x_near, x_random)
            if (self.safe(x_new) and self.collision_free(x_near, x_new)):
                x_lastest = x_new
                x_new_node = tuple(x_new.reshape(3,).tolist())
                x_near_node = tuple(x_near.reshape(3,).tolist())
                cost_new = self.G.node[x_near_node]['cost'] + dist
                self.G.add_node(x_new_node, cost = cost_new, parent = x_near_node)
                self.G.add_edge(x_new_node, x_near_node, weight = dist)
        if (self.tic() - t0) > 1.8:
            self.x_lastest = x_lastest
            return start
      print("finished")
      self.graph_finished = True
      self.x_lastest = x_lastest
      return start
    elif(self.graph_finished == True and self.path_finished == False):
      print("path")
      if self.first_path :
        self.path.append(goal)
        self.path.append(self.x_lastest.reshape(3,))
        self.node_a = tuple(self.x_lastest.reshape(3,).tolist())
        self.first_path = False
        return start
            
      t0 = self.tic()
      node_a = self.node_a
      while node_a != self.start_node:
        p = self.G.node[node_a]['parent']
        self.path.append(np.asarray(p))
        node_a = p
        if(self.tic() - t0) > 1.8:
          self.node_a = node_a
          return start
      self.path_finished = True
      self.path = self.path[::-1]
      return start
    else:
      print("finished")
      newpos = self.path[self.move]
      self.move = self.move+1
      return newpos
