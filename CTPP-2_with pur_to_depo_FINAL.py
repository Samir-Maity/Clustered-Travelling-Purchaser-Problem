##### GA_TSP with COMPARISON CROSSOVER & PROBABILISTIC Selection
import sys
import seed
import random
import math
import numpy as np
import tsplib95
import re
#from myapp import get_distance
from deepdiff import DeepDiff
import networkx
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import csv
import itertools
#import label,goto

#Importing required modules
 
#from sklearn.datasets import load_digits
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans


#node = 17

m = np.array([])
population = []  # list that holds paths
population1 = []  # list that holds loads
population2 = []  # list that holds costs

final_nodes = []
cl_nodes = []
#cluster_results = np.array([])
cluster_costs = np.array([])
population_size = 30  # max 120 combinations
ps = 0.34             # probability of selection
cross_prob = 0.5
mutate_prob = 0.01   # probability of mutation
#cx_prob = 0.34      # probability of crossover
n_generations = 500
# Maximum iteration
routes_length = [0] * population_size  # list to store the total length of each chromosome
newpopulation = [0]*population_size
newpopulation1 = [0]*population_size
newpopulation2 = [0]*population_size
fitness = [0]*population_size        # list to store the total fitness of each chromosome
best_path = math.inf

#random.seed(9)
seed = input('Please enter seed:  ')
random.seed(seed)

k_max_ite = 10
#distance_matrix = []

#problem = tsplib95.load('D:\\CHINA TPP\\TPP_python code\\dataset\\kroD100.tsp')

#problem = tsplib95.load('D:\\CHINA TPP\\TPP_python code\\dataset\\dantzig42.tsp')
problem = tsplib95.load('bays29.tsp')
#problem = tsplib95.load('D:\\CHINA TPP\\TPP_python code\\dataset\\eil51.tsp')
#print(list(problem.get_nodes()))
#########convert into a networkx.Graph##############
graph = problem.get_graph()

# convert into a numpy distance matrix
distances = (networkx.to_numpy_matrix(graph)) 
node = distances.shape[0]
#distances = [i * 8 for i in distances]
#print("distances = ",distances)



'''
n_o_p = int(input('Hom many product ? '))
pur_cst_of_p = [0]  * node    ## store the rate of each product at each market
b1 = []
for i in range(1,node):
    #r1 = random.sample(range(100, 105), 2)
    for j in range(n_o_p):
        #b1.append(round(random.uniform(100, 105),2))
        b1.append(random.randint(100, 150))
    pur_cst_of_p[i] = b1[:]
    
    b1[:] = []
'''
pur_cst_of_p = [0]  * node    ## store the rate of each product at each market
DF = [0]  * node  ###  store the purchased amount at each market

'''
for i in range(1,node):
    pur_cst_of_p[i] = random.randint(10, 20)
print(pur_cst_of_p)
with open("p_cost_file.txt", "w") as file:
    for s in pur_cst_of_p:
        file.write(str(s) + " ")
'''

#f = open('p_cost_file.txt','r')
f = open('p_cost_file.txt','r')
Lines = f.read()
pur_cst_of_p = Lines.split()
pur_cst_of_p[:] = list(map(int, pur_cst_of_p))

#print(pur_cst_of_p)

#b = input("")

##*************************** Store the DEMAND and AVAILABILITY *****************************************
avl = [0] * node   ### availability at each node
t_avl = 0    ### Total availability
#dmd = 2000

dmd = int(input('Please say the total demand: '))     ### Total demand
'''
while(t_avl < dmd):
    t_avl = 0
    for i in range(1,node):
        avl[i] = random.randint(120,150)
        t_avl = t_avl + avl[i]
    print("avl=",avl)
    print("Total = ",t_avl)
with open("avl_file.txt", "w") as file:
    for s in avl:
        file.write(str(s) + " ")
'''

f = open('avl_file.txt','r')
Lines = f.read()
avl = Lines.split()
avl[:] = list(map(int, avl))
print(avl)
##*************************************************************************
############## READ x,y co-ordinates of each points #####################
#node = 48
L,L2 = [0] * 1,[]
#f = open('D:\\CHINA TPP\\TPP_python code\\dataset\\kroD100.txt', 'r')
f = open('bays29.txt', 'r')
#f = open('D:\\CHINA TPP\\TPP_python code\\dataset\\dantzig42.txt', 'r')
#f = open('D:\\CHINA TPP\\TPP_python code\\dataset\\eil51.txt', 'r')
Lines = f.read()
L1 = Lines.split()
R = len(L1)
#print(L1,len(L1))
#print()
i1 = 0
x,y, = 0,0
xy = []

for i in range(node):
    L1.remove(L1[i1])
    i1 += 2
#print(L1,len(L1))
i = 0
while i < len(L1)-1 :
    x = float(L1[i])
    i += 1
    y = float(L1[i])
    i += 1
    xy.append((x,y))
'''
while 1:
    line = f.readline()
    if line.find("EOF") != -1: break
    (i,x,y) = line.split()
    x = float(x)
    y = float(y)
    xy_positions.append((x,y))
'''
print(xy)
############################################################3

#count = 0
#for line in Lines:
#    count += 1
#    print("Line{}:{}".format(count,line.strip()))


#ul.urlretrieve('http://www.math.uwaterloo.ca/tsp/world/' + filename, filename)

# Read file consisting of lines of the form "k: x y" where k is the
# point's index while x and y are the coordinates of the point. The
# distances are assumed to be Euclidean.

node = distances.shape[0]
cities = list(range(node))
best_index = [0] * len(cities)
cluster = [0] * node
#print(distances)
#for i in range(node) :
#  for j in range(node) :
#    if distances[i,j] == distances[j,i] :
#        distances[i,j] = 0
    
#print(distances)


'''

# importing the XML file
weights = tsputil.read_tsplib('burma14.xml')

# printing the weights matrix
tsputil.print_matrix(weights)

# creating a tsp problem from the imported weights matrix
tsp_instance = tsp(weights)

# printing the tsp problem details to console
print(tsp_instance)

'''



# calculates distance between 2 cities
def calc_distance(city1, city2):
    return distances[city1,city2]  # ord('A')=65


# creates a random route
def create_route(cl,A,D):
    #cl = cl
    #shuffled = random.sample(cl_nodes[cl], len(cl_nodes[cl]))
    d_cl = 0
    b1 = []
    
    #while(d_cl < D[cl]):
    while(d_cl < D[cl]):
        #a = temp_cl_nodes[i][j]
        a = random.randint(0,len(cl_nodes[cl])-1)
        a = cl_nodes[cl][a]
        if(a not in b1):
            #print("a = ",a)
            b1.append(a)
            d_cl = d_cl + avl[a]
        
    #cl_nodes[i] = b1

    return b1        #shuffled


# calculates length of an route
def calc_route_length(cl):
    #size = len(cl_nodes[cl])
    for i in range(population_size):
        size = len(population[i])
        route_l = 0
        pur_cost = 0
        d_cl = 0
        t = 0
        t2 = 0
        p = 0
        j1 = 0
        
        #b = input("")
        for j in range(size-1):
            #print("J = ",j)
            #x = input("")
            route_l = route_l + calc_distance(population[i][j], population[i][j+1])
            t = avl[population[i][j]]
            #DF[population[i][j]] = t
            d_cl = d_cl + t
            p = pur_cst_of_p[population[i][j]] * t
            pur_cost = pur_cost + p
        j = size-1

            #print("route_l, pur_cost, route_l + pur_cost = ",route_l,pur_cost,route_l + pur_cost)
        t1 = D[cl] - d_cl
        t = avl[population[i][j]]
        #if(t1 > 0):
        if(t1 >= t):
            t2 = t
        elif(t1 < t):
            #t3 = t - t1
            t2 = t1
        d_cl = d_cl + t2
        DF[population[i][j]] = t2
        p = pur_cst_of_p[population[i][j]] * t2
        pur_cost = pur_cost + p
        route_l = route_l + pur_cost
        D_full[cl] = d_cl
        #p_cost[cl] = pur_cost
            #x = input("")
            
            
        #route_l = route_l + calc_distance(population[i][size-1], 0) + calc_distance(0, population[i][0])
        #route_l = route_l + calc_distance(population[i][size-1], population[i][0])
        
        routes_length[i] = route_l
        route_l = 0
        #fitness[i] = 1 / routes_length[i]
       # print("fitness=", fitness[i])

# creates starting population
def create_population(cl,A,D):
    population[:] = []
    population1[:] = []
    b1 = []
    b2 = []
    for ik in range(population_size):
        b1[:] = []
        b2[:] = []
        d = 0
        b1 = create_route(cl,A,D)
        for i in range(len(b1)-1):
            b2.append(avl[b1[i]])
            d += avl[b1[i]]
        x = D[cl] - d
        b2.append(x)
        #print("b2 = ",sum(b2),D[cl])
        #x1 = input("")
        population.append(create_route(cl,A,D))
        population1.append(b2)



# swap with a probability 2 cities in a route
def swap_mutation(cl,ind):
    size = len(population[ind])
    picks = random.sample(range(0,size-1), 2)
    population[ind][picks[0]], population[ind][picks[1]] = population[ind][picks[1]], population[ind][picks[0]]
    population1[ind][picks[0]], population1[ind][picks[1]] = population1[ind][picks[1]], population1[ind][picks[0]]
#    temp = population[ind][picks[0]]
#    population[ind][picks[0]] = population[ind][picks[1]]
#    population[ind][picks[1]] = temp
  #  print("Mutated path: ", population[ind])



def comparison_crossover(cl,ind1, ind2,ind3):
    size = len(cl_nodes[cl])
    #print("D[cl] = ",D[cl])
    #ch1, ch2 = [0] * size, [0] * size
    ch1 = []
    ch2 = []
    temp1,temp2 = [0]* size,[0]* size
    
    temp1 = cl_nodes[cl][:]
    temp = cl_nodes[cl][:]
    '''
    b1 = []
    for i1 in range(size-1):
        b1.append(pur_cst_of_p[cl_nodes[cl][i1]])
    inx_b1 = b1.index(min(b1))
    #print("inx_b1 =",inx_b1)
    #xx = input("")
    c1 = cl_nodes[cl][inx_b1]
    b1[:] = []
    '''
    b = []
    b11 = []
    c1 = random.randint(0, size-1)
    
   
    c1 = temp1[c1]
    
    ch1.append(c1)
    b.append(avl[c1])
#    print("ch1[ 0 ]=", ch1[ 0 ])
    for k in range(size):
        if temp1[k]==c1:
            temp1[k]= -1
#    print("temp1[k]=", temp1[k])  

    ###### Generate Child1
     
    i2,i3,i4=0,0,0
    v1,v2,v3 = 0,0,0
    x1,x2,x3 = 0,0,0
    T1,T2,T3 = 0,0,0
    d_cl = avl[c1]
    j1 = 1
    while(d_cl < D[cl]):
    #for j1 in range(1, size) :
        #print("J11=",j1)
        t1,t2,t3 =0,0,0
        while i2 in range(len(ind1)) :
            if ind1[i2] in temp1:   
                  v1 = calc_distance(c1,ind1[i2])
                  t1 = D[cl] - d_cl
                  if t1 >= avl[ind1[i2]]:
                     t2 = avl[ind1[i2]] 
                  elif t1 < avl[ind1[i2]]:
                      t3 = avl[ind1[i2]] - t1
                      t2 = avl[ind1[i2]] - t3
                  T1 = t2
                  x1 = pur_cst_of_p[ind1[i2]] * T1
                  #x1 = pur_cst_of_p[ind1[i2]] * avl[ind1[i2]]
                  v1 = v1 + x1
                  break
            elif (ind1[i2] not in temp1) :
                i2+=1
                if i2 == len(ind1):
                     v1 = math.inf
                 
        while i3 in range(len(ind2)) : 
        #for i3 in range(size) :    
            if ind2[i3] in temp1:
                   v2 = calc_distance(c1,ind2[i3])
                   t1 = D[cl] - d_cl
                   if t1 >= avl[ind2[i3]]:
                      t2 = avl[ind2[i3]] 
                   else:
                      t3 = avl[ind2[i3]] - t1
                      t2 = avl[ind2[i3]] - t3
                   T2 = t2
                   x2 = pur_cst_of_p[ind2[i3]] * T2
                   #x2 = pur_cst_of_p[ind2[i3]] * avl[ind2[i3]]
                   v2 = v2 + x2
                   break
            elif (ind2[i3] not in temp1) :
                 i3+=1
                 if i3 == len(ind2):
                     v2 = math.inf
                     #break

        while i4 in range(len(ind3)) : 
        #for i3 in range(size) :    
            if ind3[i4] in temp1:
                   v3 = calc_distance(c1,ind3[i4])
                   t1 = D[cl] - d_cl
                   if t1 >= avl[ind3[i4]]:
                      t2 = avl[ind3[i4]] 
                   else:
                      t3 = avl[ind3[i4]] - t1
                      t2 = avl[ind3[i4]] - t3
                   T3 = t2
                   x3 = pur_cst_of_p[ind3[i4]] * T3
                   #x3 = pur_cst_of_p[ind3[i4]] * avl[ind3[i4]]
                   v3 = v3 + x3
                   break
            elif (ind3[i4] not in temp1) :
                 i4+=1
                 if i4 == len(ind3):
                     v3 = math.inf




        #print("i2=",i2)
        #print("i3=",i3)
        if  v1 < v2 and v1 < v3 :
            
            #print("i2 = ",i2,len(ind1))
            #x2 = input("")
            c1 = ind1[i2]
            ch1.append(c1)
            #b.append(T1)
            T = T1
        elif v2 < v1 and v2 < v3 :
            #ch1[j1] = ind2[i3]
            c1 = ind2[i3]
            ch1.append(c1)
            #b.append(T2)
            T = T2
        else:
            #ch1[j1] = ind3[i4]
            c1 = ind3[i4]
            ch1.append(c1)
            #b.append(T3)
            T = T3
        #c1=ch1[j1]
        d_cl = d_cl + T
        b.append(T)
        #print("d_cl = ",avl[c1],T,d_cl,D[cl])
        #x2 = input("")
        

        i2,i3,i4=0,0,0
        
        for k in range(size):
            if temp1[k]==c1:
                temp1[k]= -1

#    print("i2,i3=", i2,i3)     
############################################    
    #Generate Child2
        
    #for i1 in range(size):
    #    temp2[i1]=i1
    temp[:] = []
    temp2[:] = []
    temp2 = cl_nodes[cl][:]
    temp = cl_nodes[cl][:]
    c2 = random.randint(0, size-1)
    
    c2 = temp2[c2]
    
    
    ch2.append(c2)
    b11.append(avl[c2])
    
    for k in range(size):
        if temp2[k]==c2 :
            temp2[k]= -1
#    i4,i5=0,0
    #si=size-2
    #v1,v2,v3 = 0,0,0
    d_cl = 0
    d_cl = avl[c2]
    j1 = 0
    while(d_cl < D[cl]):
    #for j1 in range(size-1) :
        #print("J12=",j1)
        i2 = len(ind1)-1
        while i2 != -1 :
        #while i2 in range(len(ind1)-1) :
            if ind1[i2] in temp2 :
                if i2< size :
                  v1 = calc_distance(c2,ind1[i2])
                  t1 = D[cl] - d_cl
                  if t1 >= avl[ind1[i2]]:
                     t2 = avl[ind1[i2]] 
                  elif t1 < avl[ind1[i2]]:
                      t3 = avl[ind1[i2]] - t1
                      t2 = avl[ind1[i2]] - t3
                  T1 = t2
                  x1 = pur_cst_of_p[ind1[i2]] * T1
                  #x1 = pur_cst_of_p[ind1[i2]] * avl[ind1[i2]]
                  v1 = v1 + x1
                  
                  break
            elif ind1[i2] not in temp2 :
                  i2 -= 1
                  
        i3 = len(ind2)-1
        while i3 != -1 :
        #while i3 in range(len(ind2)-1) :    
            if ind2[i3] in temp2:
                   v2 = calc_distance(c2,ind2[i3])
                   t1 = D[cl] - d_cl
                   if t1 >= avl[ind2[i3]]:
                      t2 = avl[ind2[i3]] 
                   else:
                      t3 = avl[ind2[i3]] - t1
                      t2 = avl[ind2[i3]] - t3
                   T2 = t2
                   x2 = pur_cst_of_p[ind2[i3]] * T2
                   #x2 = pur_cst_of_p[ind2[i3]] * avl[ind2[i3]]
                   v2 = v2 + x2
                   
                   break
            elif ind2[i3] not in temp2 :
                  i3 -= 1
                  
        i4 = len(ind3)-1
        while i4 != -1 :
        #while i4 in range(len(ind3)-1) :    
            if ind3[i4] in temp2:
                   v3 = calc_distance(c2,ind3[i4])
                   t1 = D[cl] - d_cl
                   if t1 >= avl[ind3[i4]]:
                      t2 = avl[ind3[i4]] 
                   else:
                      t3 = avl[ind3[i4]] - t1
                      t2 = avl[ind3[i4]] - t3
                   T3 = t2
                   x3 = pur_cst_of_p[ind3[i4]] * T3
                   #x3 = pur_cst_of_p[ind3[i4]] * avl[ind3[i4]]
                   v3 = v3 + x3
                   
                   break
            elif ind3[i4] not in temp2 :
                 i4 -= 1



#        print("i2,i3=", i2,i3)             
        #print("VV2=",v2)
        if  v1 < v2 and v1 < v3:
            #ch2[si-j1] = ind1[i2]
            c2 = ind1[i2]
            ch2.append(c2)
            T = T1
        elif v2 < v1 and v2 < v3:
            #ch2[si-j1] = ind2[i3]
            c2 = ind2[i3]
            ch2.append(c2)
            T = T2
        else:
            #ch2[si-j1] = ind3[i4]
            c2 = ind3[i4]
            ch2.append(c2)
            T = T3
        #c2=ch2[si-j1]
        d_cl = d_cl + T
        b11.append(T)
        #j1 += 1

        i2,i3,i4=0,0,0
        #x1,x2,x3 = 0,0,0
        for k in range(size):
            if temp2[k]==c2 :
                temp2[k]= -1
    #print("ch1 = ",ch1)
    ch2.reverse()
    b11.reverse()
    
    #print("ch1 = ",ch1)
    #print("ch2 = ",ch2)
    #print("BB11 = ",b,b11)
    return ch1, ch2,b,b11

#####################################################################

def prob_selection():
    #size = len(cl_nodes[cl])
    s = math.inf
    m1 = 0
    
    #label:A
    g = random.randint(0,n_generations)
#    key = 0
    ind = 0
    a1=0
    for m in range(population_size):
     x = routes_length.index(min(routes_length))
     #if routes_length[x] < s : 
     m1 = x #find_fittest()
         #print("m1=",m1)
    
    t0=random.randint(6,101)

    r=random.uniform(0,1)
    p = g//(n_generations)
#    print("p=",p)
    k = ((100.0 * p))
#    k = round(k, 3)
    t = t0 * ((1 - r)**k)
    #print("t=",t)
#    t = round(t, 3)


    #for m in range(population_size):
    m = random.randint(0,population_size-1)
    r1 = random.uniform(0,1)
    if r1 <= ps :
        a1=m
            #ind=ind+1
        #break
    elif np.exp((routes_length[m1] - routes_length[m])//t)>r1 :         
         a1=m
            #ind = ind + 1
         #break
    else:
        a1=m1
            #ind = ind + 1
#        if ind == population_size :
#            ind = population_size - 1
        #break
    
    return a1





# find fittest path called every generation
def find_fittest():
    key = math.inf
    fittest = 0
    for i1 in range(population_size):
        if routes_length[i1] < key:
            key = routes_length[i1]
            fittest = i1
    return fittest



def calc_dist(p1,p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) +
                     math.pow((p2[1] - p1[1]), 2)) 
                    



#K-Means Algorithm
def kmeans(k):
    temp1,temp2 = [0]* node,[0]* node
    ncenter = []
    center_c = np.array([])
    center_c = np.ones((k,2))
    ncenter = []
    fcenter_c = []
    
    min_dist = 9999
    
    final_nodes = [[]for i in range(k)]
          # store randomly chosen centroid
    center = list(range(k))
    print("center = ",center)
      
    center = random.sample(range(1, node-1), k)
    print("center_nodes = ",center)
    x,y = 0,0
    for i in range(k):
        (x,y) = xy[center[i]][0],xy[center[i]][1]
        ncenter.append((x,y))
           
    for j in range(k_max_ite) :
       print("ITERATION = ",j)

       print("k=",k)
       c_nodes = [[]for i in range(k)]
       prev_nodes = [[]for i in range(k)]
      
       c_xy = []
       #center_p = []
       


       print("centroid_cords = ",ncenter)

#-------------------------------------------------------------------------
       c_dist = np.array([])
       c_dist = np.ones(k)
       #c_n = np.array([])
       #c_n = np.zeros((k,node))
       c_ns = [[0]*1]*k
       c_n = []
       c1,c2,c3,c4 = [],[],[],[] 
       mc1,mc2,mc3,mc4 = [],[],[],[] 
       for p in range(1,node):
           for c in range(0,k):
               p1 = ncenter[c]
               p2 = xy[p]
               dist = calc_dist(p1,p2)
               c_dist[c] = dist
           #print(c_dist)
           pos = np.argmin(c_dist)
           #print(pos,p)
           c_n.append(pos)
       t = []
       t1 = []
       for i in range(1,node):
           t.append(i)
       for c in range(k):
           j = 0
           t1[:] = []
           for i in t:
                if(c_n[j] == c):
                    #c_ns[c].append(i)
                    t1.append(i)
                    j += 1
                else:
                    j += 1
           c_ns[c] = t1[:]  

       #print("Nodes are: ")
       #print(c_n)
       #print(c_ns)  
       ######  store previous centroids  ################
       prev_center = ncenter.copy()
       ######  Calculate new centroids  ##############
       #c_ns_sum = [[0]*1]*k
       lx = [[0]*1]*k
       ly = [[0]*1]*k
       
       for c in range(k):
           sx,sy = 0,0
           for i in range(len(c_ns[c])):
               p = c_ns[c][i]
               x,y = xy[p][0],xy[p][1]
               #print(p,x,y)
               sx += x
               sy += y
           sx = sx/ len(c_ns[c])
           sy = sy/ len(c_ns[c])
           
           print(sx,sy)
           #center_p.append((sx,sy))
           center_p[c] = (sx,sy)

       print("Current center =",center_p)

       ###################################################
       c_nodes = c_ns.copy()
       #print("differences b/w lists:")
       #print(DeepDiff(prev_center, center_p))
       

       #for c in range(k) :
       if DeepDiff(prev_center, center_p) : 
            #print("YES")
            prev_center[:] = []
            #ncenter[:] = []
            prev_center = center_p.copy()
            ncenter = center_p.copy()
            final_nodes = c_nodes
            #print('prev_nodes=')
            #print(prev_nodes)
       else :
            final_nodes = c_nodes
            break
 
    print("Final = \n",final_nodes)
    print("centers are:")
    print(center_p )
    #return final_nodes

###################################################
    #final_nodes[0] = [2,4,5,8,11,25,28]
    #final_nodes[1] = [16,25,20]
    #final_nodes[2] = [1,3,6,7,9,10,12,13,14,15,17,18,19,21,22,23,24,26,27]
    #center_p[0] = [360,1980]
    #center_p[1] = [750.0, 2030.0]
    #center_p[2] = [830.0, 1770.0]
#-----------------------------------------------------------------------
    '''
    import csv
    temp3 = range(0,node)
    m = np.ones((node,k+1))
    m1 = np.ones((node,k+1))
    m[:,k] = temp3
    m1[:,k] = temp3
    with open('data', 'w') as f:
    # create the csv writer
        writer = csv.writer(f)

        for i in range(1,node):
            for j in range(0,k):
                m[i][j] = calc_distance(i,center[j])
            writer.writerow(m[i,:])
        print(m)

        for i in range(1,node):
            for j in range(0,k):
                if(i in final_nodes[j]):
                    m1[i][j] = calc_distance(i,center[j])
                #else:
                    #m1[i][j] = 0
    '''
#-----------------------------------------------------------------------

    lx1,ly1 = [],[]
    for i in range(k):
        for j in range(len(final_nodes[i])):
            p = final_nodes[i][j]
            lx1.append(xy[p][0])
            ly1.append(xy[p][1])
    text = range(node)
    #print("text = ")
    #print(text)
    text1 = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]
    #plt.figure(figsize=(10,8))
    # Creating equally spaced 100 data in range 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, 100)
    # Setting radius
    radius = 2

    # Generating x and y data
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    #ThetaArray = np.linspace(0, 2*np.pi, 36, endpoint=True)
    #x = 0.8*np.cos(ThetaArray)
    #y = 0.8*np.sin(ThetaArray)

#-------plot it. note how the line color changes...
    plt.plot(x,y)
    #plt.show()
    

    for i in range(k):
        if(i == 0):
            
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
            
        if(i == 1):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 2", color= "red", marker= "*", s=30)
        if(i == 2):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 3", color= "blue", marker= "*", s=30,alpha = 1.0)
        if(i == 3):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 4", color= "yellow", marker= "*", s=30,alpha = 1.0)
        if(i == 4):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 5", color= "orange", marker= "*", s=30,alpha = 1.0)
        if(i == 5):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 6", color= "pink", marker= "*", s=30,alpha = 1.0)
        if(i == 6):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 7", color= "purple", marker= "*", s=30,alpha = 1.0)
        if(i == 7):
            for i1 in range(len(final_nodes[i])):
                x,y = xy[final_nodes[i][i1]][0],xy[final_nodes[i][i1]][1]
                plt.scatter(x, y, label= "cluster 7", color= "brown", marker= "*", s=30,alpha = 1.0)
    for i in range(k):
        for i1 in range(len(center_p)):
            x,y = center_p[i1][0],center_p[i1][1]
            plt.scatter(x, y, label= "center", color= "black", marker= "o", s=10)
    for i in range(k):
        x,y = center_p[i][0],center_p[i][1]
        plt.annotate(text1[i], (x, y + 0.2))
    for i in range(0,len(text)):
        x,y = xy[i][0],xy[i][1]
        plt.annotate(text[i], (x, y + 0.2))  #### put name for each point

    #plt.figure(figsize=(12,8))
    # x-axis label
    
    #plt.xlim([0, 190])
    #plt.ylim([0, 180])
    
#    plt.xlim([0, 2200])
#    plt.ylim([0, 2600])
    
    #plt.xlim([0, 4200])
    #plt.ylim([0, 4300])
    
    plt.xlabel('x - axis')
    # frequency label
    plt.ylabel('y - axis')
    # plot title
    #plt.title('Before clusters of bays29 problems')
    # showing legend
    #plt.legend()
  
    # function to show the plot
    
    plt.show()







###**************************************************************************************




    
    return final_nodes
###########################################################################################
###########################################################################################

k1 = input('How many cluster? \t')
k = int(k1)
center_p = [0]*k
##############*************************

############################****************************

cl_nodes1 = kmeans(k)
cl_nodes = cl_nodes1.copy()
cluster_results = []     ### store final nodes of each cluster
cluster_results_load = []     ### store final nodes of each cluster and corresponding purchased product amount
cluster_results_rev = [0]*k     ### store reverse of final nodes of each cluster
cluster_costs = np.ones(k)        ### store relevant best path cost of each cluster path

########### cluster wise total availability  ########################

tr_cost = [0] * k      ###### set transportation cost per unit rate (from center to depot)
for i in range(k):
    tr_cost[i] = round(random.uniform(1.4,1.5),2)
#print("TR = ",tr_cost)
#r = input("")

A = [0]*k
A1 = [0]*k
D = [0]*k
D_full = [0]*k
p_cost = [0]*k
for i in range(k):
    a = len(cl_nodes[i])
    tl = 0
    for j in range(a):
        a1 = cl_nodes[i][j] 
        tl = tl + avl[a1]
    A[i] = tl
print("Actual Avaailability per cluster : ",A)
#b = input("")

j = 0
for i in range(k):
    A1[i] = random.uniform(0,1)
    j = j + A1[i]
#print("Demand percentage per cluster : ",A1,j)
for i in range(k):
    D[i] = round((A1[i]/j)*dmd)   #### D[i]  stores the actual total demands of each cluster
print("Actual Demand per cluster : ",D)

##****************************************************************************
#######################  Adjust the demand as per cluster wise avalability  ##############################
t_dmd = [0]*k
t_avl = [0]*k
#t_dmd = [i[0] for i in sorted(enumerate(D), key=lambda x:x[1])]   ## sort the List D
t_dmd = sorted(D)
t_avl = [i[0] for i in sorted(enumerate(A), key=lambda x:x[1])]   ## sort the List A
print("Actual Demand after sorting per cluster : ",t_dmd)
print("Actual Avaailability after sorting per cluster : ",t_avl)
for i in range(k):
    D[t_avl[i]] = t_dmd[i]
print(D)

#DF = avl[:]
#print("DF = ",DF)
#b = input("")

##***************************************************************************

########***************************************########################
for cl in range(k):
    
    b_path = 4444444444
    #if(int(len(cl_nodes[cl]) < 3 or len(cl_nodes[cl]) == 0)):
        #cl += 1
    
    print('###############')
    
    #print(cl_nodes[cl])
    # initialize algorithm
    create_population(cl,A,D)
    #print("Population initialization:", "\n", population)
    calc_route_length(cl)
    #print("Population's paths length:", "\n", routes_length)
    
        
    for y in range(n_generations):
        #DF = avl[:]
        newpopulation[:] = []
        newpopulation1[:] = []
        for i in range(population_size):
            a1 = prob_selection()
            newpopulation.append(population[a1])
            newpopulation1.append(population1[a1])
            
        
        #for i in range(population_size):
            #population[i] = newpopulation[i][:]
        population = newpopulation[:]
        population1 = newpopulation1[:]
        calc_route_length(cl)
        
        #print("NEW population after selection = \n",population)
        #print("Population's paths length:", "\n", routes_length)
        #xx = input("")
        
        parent = [0] * 3
        pa = int((cross_prob * population_size)/2)
        for i in range(pa):
            parent = random.sample(range(0, population_size), 3)
 
            # update population
            count = 0
            p2 = [0]*2
 
            p2 = random.sample(range(0, population_size), 2)  ## select randomly any two position from total population
            #print(p2)        
            population[parent[0]], population[parent[1]],population1[parent[0]],population1[parent[1]] = comparison_crossover(cl,population[parent[0]], population[parent[1]],population[parent[2]])
            #population1[parent[0]], population1[parent[1]] = b,b11
            #print("NOW: ",population)
            #xx = input("")
            
        # calculate lengths for updated generation
            calc_route_length(cl)
        #print("UPDATED population after crossover = \n",population)
        #print("Population's paths length:", "\n", routes_length)
        

        # pick the paths for mutation based on a probability
        for i in range(1,population_size):
            rand = random.uniform(0,1)
            if rand < mutate_prob:
                swap_mutation(cl,i)

        # calculate lengths after mutation
        calc_route_length(cl)

        # find best path overall
        #best_path = math.inf
        x = routes_length.index(min(routes_length))
        if routes_length[x] < b_path:
            index = find_fittest()
            #print("index=",index)
            b_path = routes_length[x]
            best_index = population[x]

    
        

        print("Best route of generation", y+1, ": ", best_index,  "Route length: ",
                b_path, "\n")
        #xx = input("")
#        print("Population of generation", j+1, ": \n", population)
#        print("Routes lengths:", routes_length, "\n")
    print("Best path is:", best_index, "with length", b_path, "\n\n")
    cluster_results.append(best_index)
    cluster_results_load.append(population1[index])
    cluster_costs[cl] = b_path
    
#cluster_results = [[6,15,22,24,26],[1,2,4,20,25,27,28],[3,9,13,14,16,17,18,21]]    
#print(cluster_costs)
#print(cluster_results)  ##store the optimum path with in each cluster
   
    
#population[:] = []

Z,Z1 = [],[]
for i in range(k):
    Z = Z + cluster_results[i]
    Z1 = Z1 + cluster_results_load[i]
    
        



########################################################################################################
##################### plot each optimum cluster  #########################################


for i in range(k):
    if(i == 0):
            
        for i1 in range(len(cl_nodes[i])):
            x,y = xy[cl_nodes[i][i1]][0],xy[cl_nodes[i][i1]][1]
            plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
            
    if(i == 1):
        for i1 in range(len(cl_nodes[i])):
            x,y = xy[cl_nodes[i][i1]][0],xy[cl_nodes[i][i1]][1]
            plt.scatter(x, y, label= "cluster 2", color= "red", marker= "*", s=30)
    if(i == 2):
        for i1 in range(len(cl_nodes[i])):
            x,y = xy[cl_nodes[i][i1]][0],xy[cl_nodes[i][i1]][1]
            plt.scatter(x, y, label= "cluster 3", color= "blue", marker= "*", s=30,alpha = 1.0)
    if(i == 3):
        for i1 in range(len(cl_nodes[i])):
            x,y = xy[cl_nodes[i][i1]][0],xy[cl_nodes[i][i1]][1]
            plt.scatter(x, y, label= "cluster 4", color= "yellow", marker= "*", s=30,alpha = 1.0)





lx1,ly1 = [],[]
for i in range(k):
    for j in range(len(cluster_results[i])):
        p = cluster_results[i][j]
        lx1.append(xy[p][0])
        ly1.append(xy[p][1])
text = range(node)
text1 = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]
x = []
y = []
x1 = []
y1 = []
for i in range(k):
    if(i == 0):
        x.append(center_p[i][0])    
        y.append(center_p[i][1])
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
            #plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
            #plt.plot(x, y)
        x.append(center_p[i][0])    
        y.append(center_p[i][1])

        plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
        plt.plot(x,y,linestyle='solid',color='blue')
        x[:] = []
        y[:] = []
            
    if(i == 1):
        x.append(center_p[i][0])    
        y.append(center_p[i][1])
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            #plt.scatter(x, y, label= "cluster 2", color= "red", marker= "*", s=30)
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        x.append(center_p[i][0])    
        y.append(center_p[i][1])
        plt.scatter(x, y, label= "cluster 2", color= "red", marker= "*", s=30)
        plt.plot(x,y,linestyle='solid',color='red')
        x[:] = []
        y[:] = []
    if(i == 2):
        x.append(center_p[i][0])    
        y.append(center_p[i][1])
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            #plt.scatter(x, y, label= "cluster 3", color= "blue", marker= "*", s=30,alpha = 1.0)
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        x.append(center_p[i][0])    
        y.append(center_p[i][1])
        plt.scatter(x, y, label= "cluster 3", color= "blue", marker= "*", s=30,alpha = 1.0)
        plt.plot(x,y,linestyle='solid',color='green')
        x[:] = []
        y[:] = []
    if(i == 3):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            #plt.scatter(x, y, label= "cluster 4", color= "yellow", marker= "*", s=30,alpha = 1.0)
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 4", color= "purple", marker= "*", s=30,alpha = 1.0)
        plt.plot(x,y,linestyle='solid',color='yellow')
        x[:] = []
        y[:] = []
    if(i == 4):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            #plt.scatter(x, y, label= "cluster 5", color= "orange", marker= "*", s=30,alpha = 1.0)
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 5", color= "brown", marker= "*", s=30,alpha = 1.0)
        plt.plot(x,y,linestyle='solid',color='blue')
        x[:] = []
        y[:] = []
    if(i == 5):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            #plt.scatter(x, y, label= "cluster 6", color= "pink", marker= "*", s=30,alpha = 1.0)
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 6", color= "black", marker= "*", s=30,alpha = 1.0)
        plt.plot(x,y,linestyle='solid',color='pink')
        x[:] = []
        y[:] = []
    if(i == 6):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 7", color= "purple", marker= "*", s=30,alpha = 1.0)
        plt.plot(x, y)
        x[:] = []
        y[:] = []
    if(i == 7):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 8", color= "brown", marker= "*", s=30,alpha = 1.0)
        plt.plot(x, y)
        x[:] = []
        y[:] = []
    if(i == 8):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 9", color= "red", marker= "o", s=30,alpha = 1.0)
        plt.plot(x, y)
        x[:] = []
        y[:] = []
    if(i == 9):
        for i1 in range(len(cluster_results[i])):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
            x.append(xy[cluster_results[i][i1]][0])
            y.append(xy[cluster_results[i][i1]][1])
        plt.scatter(x, y, label= "cluster 10", color= "red", marker= "*", s=20,alpha = 1.0)
        plt.plot(x, y)

for i in range(k):
    for i1 in range(len(center_p)):
        x,y = center_p[i1][0],center_p[i1][1]
        plt.scatter(x, y, label= "center", color= "black", marker= "o", s=10)
for i in range(k):
    x,y = center_p[i][0],center_p[i][1]
    plt.annotate(text1[i], (x, y + 0.2))
for i in range(0,len(text)):
    x,y = xy[i][0],xy[i][1]
    plt.annotate(text[i], (x, y + 0.2))  #### put name for each point

x,y = xy[0][0],xy[0][1]
plt.scatter(x, y, color= "green", marker= "s", s=30)
plt.annotate("Depot", (x, y + 0.2))

x = []
y = []

for i in range(k):
    x.append(center_p[i][0])
    y.append(center_p[i][1])
    x.append(xy[0][0])
    y.append(xy[0][1])
    plt.plot(x,y,linestyle='solid',color='black')

    #plt.figure(figsize=(12,8))
    # x-axis label
#plt.xlim([0, 190])
#plt.ylim([0, 180])

##plt.xlim([0, 2200])
##plt.ylim([0, 2600])

#plt.xlim([0, 2080])
#plt.ylim([0, 2580])

#plt.xlim([0, 4300])
#plt.ylim([0, 4500])

#plt.xlabel('x - axis')
    # frequency label
#plt.ylabel('y - axis')
    # plot title
    #plt.title('Before clusters of bays29 problems')
    # showing legend
#plt.legend()
  
    # function to show the plot
ax=plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#plt.axis('off')
plt.savefig("test.png", bbox_inches='tight',pad_inches=0)

plt.show()



########################################################################################################

def calc_finalpath(final_path):
    route_l = 0
    for j1 in range(len(final_path)-1):
        p1 = final_path[j1]
        p2 = final_path[j1+1]
        route_l = route_l + calc_distance(p1, p2)
    #route_l = route_l + calc_distance(0, final_path[0]) + calc_distance(0, final_path[-1])
    return route_l


def plot_finalpath(final_path):
    
    lx1,ly1 = [],[]
    for i in range(k):
        for j in range(len(final_path)):
            p = final_path[j]
            lx1.append(xy[p][0])
            ly1.append(xy[p][1])
    text = range(node)

    x = []
    y = []

    for i1 in range(len(final_path)):
                #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
        x.append(xy[final_path[i1]][0])
        y.append(xy[final_path[i1]][1])
                #plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
                #plt.plot(x, y)
    plt.scatter(x, y, label= "re-link", color= "green", marker= "*", s=30)
    plt.plot(x,y,linestyle='solid',color='blue')

    for i in range(0,len(text)):
        x,y = xy[i][0],xy[i][1]
        plt.annotate(text[i], (x, y + 0.2))  #### put name for each point

        #plt.figure(figsize=(12,8))
        # x-axis label
    #plt.xlim([0, 190])
    #plt.ylim([0, 180])
        
#    plt.xlim([0, 2200])
#    plt.ylim([0, 2600])
        
    #plt.xlim([0, 4300])
    #plt.ylim([0, 4500])
        
    plt.xlabel('x - axis')
        # frequency label
    plt.ylabel('y - axis')
        # plot title
        #plt.title('Before clusters of bays29 problems')
        # showing legend
    #plt.legend()
  
    # function to show the plot
   
    plt.show()

#######################################  cluster re-linking  #####################################################
cluster_results = [[6,15,22,24,26],[1,2,4,20,25,27,28],[3,9,13,14,16,17,18,21]]    

cl_arr = []
b = []
final_path = []
dd = [0] * (k)
dd[0] = math.inf
final_cost = math.inf
for i in range(k):
    cl_arr.append(i)
cl_arr[0] = math.inf

    #print("Centroids are= ",center_p)
    
for i in range(1,k):
    p1 = center_p[0]
    print("i = ",i)
    p2 = center_p[i]
    dd[i] = calc_dist(p1,p2)
print("DD = ",dd)
cl_no = dd.index(min(dd))
print("minpos_cluster = ",cl_no)

cl_arr[cl_no] = math.inf
a1 = cluster_results[cl_no].copy()
for i2 in range(len(a1)):
    p1 = center_p[0]
    p2 = xy[a1[i2]]
    a = calc_dist(p1,p2)
    b.append(a)
print(b)
minpos = b.index(min(b))
print(minpos)
n1 =  a1[minpos]
b[:] = []
a2 = cluster_results[0].copy()
for i2 in range(len(a2)):
    p1 = center_p[cl_no]
    p2 = xy[a2[i2]]
    a = calc_dist(p1,p2)
    b.append(a)
print(b)
minpos = b.index(min(b))
print(minpos)
n2 =  a2[minpos]    
print(n1,n2)
indx1 = a1.index(n1)
indx2 = a2.index(n2)
print("INDEX1,INDEX2 : ",indx1,indx2)

temp1 = [0] * len(a1)
temp2 = [0] * len(a2)
d1,d2 = 0,0
if(indx2 == 0):
    a2.reverse()
elif(indx2 != len(a2)-1):
    d1 = calc_distance(n2,a2[indx2+1])
    d2 = calc_distance(n2,a2[indx2-1])

#print("d1,d2 = ",d1,d2)
if(d1 > d2):
    i2 = 0
    i1 = indx2
    while(i1 != -(len(a2)-indx2)):
        #print("YES")
        temp2[i2] = a2[i1]
        i2 += 1
        i1 -= 1
    temp2.reverse()
    a2[:] = []
    a2 = temp2.copy()
    temp2[:] = []
    print("a2 = ",a2)


if(d1 < d2):
    i2 = 0
    i1 = indx2
    while(i1 != (len(a2))):
        #print("YES")
        temp2[i2] = a2[i1]
        i2 += 1
        i1 += 1
    i1 = 0
    while(i1 < indx2):
        temp2[i2] = a2[i1]
        i2 += 1
        i1 += 1
    temp2.reverse()
    a2[:] = []
    a2 = temp2.copy()
    temp2[:] = []
    print("a2 = ",a2)

    
###############################################################################################
if(indx1 == 0):
    a1.reverse()
elif(indx1 > 0 and indx1 < len(a1)-1):
    d1 = calc_distance(n1,a1[indx1+1])
    d2 = calc_distance(n1,a1[indx1-1])

print("d1,d2 = ",d1,d2)
if(d1 > d2):
    i2 = 0
    i1 = indx1
    while(i1 != -(len(a1)-indx1)):
        #print("YES")
        temp1[i2] = a1[i1]
        i2 += 1
        i1 -= 1
    a1[:] = []
    a1 = temp1.copy()
    temp1[:] = []
    print("a1 = ",a1)


if(d1 < d2):
    i2 = 0
    i1 = indx1
    while(i1 != (len(a1))):
        #print("YES")
        temp1[i2] = a1[i1]
        i2 += 1
        i1 += 1
    i1 = 0
    while(i1 < indx1):
        temp1[i2] = a1[i1]
        i2 += 1
        i1 += 1
    a1[:] = []
    a1 = temp1.copy()
    temp1[:] = []
    print("a1 = ",a1)




##################################################################################################
final_path = final_path + a2 + a1
print(final_path)
input()
plot_finalpath(final_path)
count = cl_arr.count(math.inf)   ###*************** count the clusters remains
print("Remain_clusters are: ",cl_arr)
if((k-count) == 1):
    for i in range(k):
        if(cl_arr[i] != math.inf):
            ind_x = i
            ind_value = cl_arr[i]
    print("REMAIN Cluster is: ",ind_value)
    
##### After merge of two cluster, find the nearest node from remain cluster center
    #v1,v2 = 0,0
    p1 = xy[final_path[0]]
    p2 = center_p[ind_x]
    d1 = calc_dist(p1,p2)
    p1 = xy[final_path[-1]]
    p2 = center_p[ind_x]
    d2 = calc_dist(p1,p2)
    print("v1,v2 = ",d1,d2)
    if(d1<d2):
        v = final_path[0]
    else:
        v = final_path[-1]
    
    a2[:] = []
    b[:] = []
    a2 = cluster_results[ind_value].copy()
    for i2 in range(len(a2)):
        p1 = xy[v]
        p2 = xy[a2[i2]]
        a = calc_dist(p1,p2)
        b.append(a)
    print("b = ",b)
    minpos = b.index(min(b))
    print(minpos)
    n2 =  a2[minpos]
    print("node = ",n2)
##
    indx2 = a2.index(n2)
    print("INDEX2 : ",indx2)

    temp2 = [0] * len(a2)
    if(indx2 == -1):
        a2.reverse()
    elif(indx2 > 0 and indx2 < len(a2)-1):
        d1 = calc_distance(n2,a2[indx2+1])
        d2 = calc_distance(n2,a2[indx2-1])

    print("d1,d2 = ",d1,d2)
    if(d1 > d2):
        i2 = 0
        i1 = indx2
        while(i1 != -(len(a2)-indx2)):
            #print("YES")
            temp2[i2] = a2[i1]
            i2 += 1
            i1 -= 1
        #temp2.reverse()
        a2[:] = []
        a2 = temp2.copy()
        temp2[:] = []
        print("a2 = ",a2)


    elif(d1 < d2):
        i2 = 0
        i1 = indx2
        while(i1 != (len(a2))):
            #print("YES")
            temp2[i2] = a2[i1]
            i2 += 1
            i1 += 1
        i1 = 0
        while(i1 < indx2):
            temp2[i2] = a2[i1]
            i2 += 1
            i1 += 1
        temp2.reverse()
        a2[:] = []
        a2 = temp2.copy()
        temp2[:] = []
        print("a2 = ",a2)
    indx = final_path.index(v)
    print("position of v = ",indx)
    if(indx == 0):
        final_path.reverse()
    final_path = final_path + a2
    print(final_path)
    plot_finalpath(final_path)

##############  if no. of cluster is more than three
ind_x = []
cl_x1 = []    ## store v1 to all center distance
cl_x2 = []    ## store v2 to all center distance

if((k-count) >= 2):
    while(cl_arr[1:] != cl_arr[:-1]) :
        for i in range(k):
            if(cl_arr[i] != math.inf):
                ind_x.append(i)
            
        print("REMAIN Clusters are: ",ind_x)
        for i in range(len(ind_x)):
            v1 = final_path[0]
            v2 = final_path[-1]
            p1 = xy[v1]
            p2 = center_p[ind_x[i]]
            d1 = calc_dist(p1,p2)
            cl_x1.append(d1)
            p1 = xy[v2]
            p2 = center_p[ind_x[i]]
            d2 = calc_dist(p1,p2)
            cl_x2.append(d2)
        print(cl_x1)
        print(cl_x2)
        mind_clus1 = cl_x1.index(min(cl_x1))
        mind_clus1 = ind_x[mind_clus1]
        mind1 = min(cl_x1)
        mind_clus2 = cl_x2.index(min(cl_x2))
        mind_clus2 = ind_x[mind_clus2]
        mind2 = min(cl_x2)
        print("mind1,mind2 = ",mind1,mind2)
    
#************************
        if(mind1 < mind2):
            mind_clus = mind_clus1
            v = v1
        else:
            mind_clus = mind_clus2
            v = v2
        
        cl_arr[mind_clus] = math.inf     ######  Selected cluster
        print("Selected Cluster = ",mind_clus)
        print("V = ",v)
#**********************************************
        a2[:] = []
        b[:] = []
        a2 = cluster_results[mind_clus].copy()
        for i2 in range(len(a2)):
            p1 = xy[v]
            p2 = xy[a2[i2]]
            a = calc_dist(p1,p2)
            b.append(a)
        print("b = ",b)
        minpos = b.index(min(b))
        print(minpos)
        n2 =  a2[minpos]
        print("node = ",n2)
##
        indx2 = a2.index(n2)
        print("INDEX2 : ",indx2)

        temp2 = [0] * len(a2)
        if(indx2 == -1):
            a2.reverse()
        elif(indx2 > 0 and indx2 < len(a2)-1):
            d1 = calc_distance(n2,a2[indx2+1])
            d2 = calc_distance(n2,a2[indx2-1])

        print("d1,d2 = ",d1,d2)
        if(d1 > d2):
            i2 = 0
            i1 = indx2
            while(i1 != -(len(a2)-indx2)):
                #print("YES")
                temp2[i2] = a2[i1]
                i2 += 1
                i1 -= 1
            #temp2.reverse()
            a2[:] = []
            a2 = temp2.copy()
            temp2[:] = []
            print("a2 = ",a2)


        elif(d1 < d2):
            i2 = 0
            i1 = indx2
            while(i1 != (len(a2))):
                #print("YES")
                temp2[i2] = a2[i1]
                i2 += 1
                i1 += 1
            i1 = 0
            while(i1 < indx2):
                temp2[i2] = a2[i1]
                i2 += 1
                i1 += 1
            temp2.reverse()
            a2[:] = []
            a2 = temp2.copy()
            temp2[:] = []
            print("a2 = ",a2)
        indx = final_path.index(v)
        print("position of v = ",indx)
        if(indx == 0):
            final_path.reverse()
        final_path = final_path + a2
        print(final_path)
        plot_finalpath(final_path)
        ind_x[:] = []
        cl_x1[:] = []
        cl_x2[:] = []
#********************************************





 
 

route_l = calc_finalpath(final_path)
print(final_path)
print(route_l)
if(route_l < final_cost):
    final_route = final_path.copy()
    final_path[:] = []
    final_cost = route_l
else:
    final_path[:] = []
    
    

print("Z = ",Z)
print("Z1 = ",Z1)
DF[:] = []
for i in range(len(final_route)):
    ind = Z.index(final_route[i])
    DF.append(Z1[ind])

t = [0]*node
DM = [0]*k
CP = [0]*k
dm = []
cp = []
p1 = 0
for i in range(node):
    t[i] = i
##################################### cluster wise nodes ands corresponding purchased products and cost ###################
for j1 in range(3):
    p1 = 0
    dm[:] = []
    cp[:] = []
    for j in range(len(cluster_results[j1])):
        d1 = t.index(cluster_results[j1][j])
        d2 = final_route.index(cluster_results[j1][j])
        a1 = pur_cst_of_p[d1] * DF[d2]
        p1 = p1 + a1
        dm.append(DF[d2])
        cp.append(pur_cst_of_p[d1])
    DM.append(dm)
    CP.append(cp)
    print("cl,Pur_cst,markt,dmnd,per_cst = \n",j1,p1,cluster_results[j1],dm,cp)
    p_cost[j1] = p1
    
###############################################





trcost = 0
'''
for i in range(k):
    if(D[i] <= 300):
        d = 100
    else:
        d = 150
    p1 = center_p[i]
    p2 = xy[0]
    trcost = trcost + (tr_cost[i] * calc_dist(p1,p2)) + d
    print("calc_dist(p1,p2) = ",calc_dist(p1,p2))
'''
#cluster_results
j = 0
b1 = 0
trcost = 0
d = 0
#print("len(final_route) = ",len(final_route))
for i in range(len(final_route)-1):
    d = d + DF[i]
    #print("final_route[j] = ",final_route[j],DF[j])
    trcost = trcost + (calc_distance(final_route[j],final_route[j+1]) * 1)*(d / 300)
    j += 1
    
#j += 1
print("NOW j = ",j)
d = d + DF[j]
p1 = xy[final_route[j]]
p2 = xy[0]
trcost = trcost + (calc_dist(p1,p2) * 1)*(d / 300)
#print("d = ", d)

#x= input("")
avl1 = [0] * len(final_route)
dvl = [0] * len(final_route)
p_c = [0] * len(final_route)

    
for i in range(len(final_route)):
    avl1[i] = avl[final_route[i]]
    ind = t.index(final_route[i])
    p_c[i] = pur_cst_of_p[ind] * DF[i]


#print("Cost = ",final_cost + sum(p_cost) + calc_distance(0,final_route[0]) + calc_distance(final_route[-1],0) )        
print("Final route is:")#, final_route)##, "and cost is :",final_cost + calc_distance(0, final_route[0]) + calc_distance(0, final_route[-1]))   
print(final_route)
print("Total cost = ",(final_cost * 1.5) + sum(p_cost)  + trcost)
print("Routing cost = ",final_cost * 1.5)
print("Purchasing cost",sum(p_c),sum(p_cost))
print("Transportation cost fron each center to depot:",trcost)


print("Availability market wise: ")
print(avl1)


print("Purchased product market wise: ")
print(DF)


print("Product cost market wise: ")
print(p_c)


#print("Final Deamnd fulfill :",d)
print("Demand fulfilled per cluster wise and cost =  ",D_full,p_cost)
print("Demand  per cluster wise =  ",D)


#######################################  plot all clusters after re-linking ######################################

lx1,ly1 = [],[]
for i in range(k):
    for j in range(len(final_route)):
        p = final_route[j]
        lx1.append(xy[p][0])
        ly1.append(xy[p][1])
text = range(node)
text1 = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]

x = []
y = []
clr = ['red', 'blue','green','yellow','pink','purple','orange']
for i1 in range(len(final_route)):
            #x,y = xy[cluster_results[i][i1]][0],xy[cluster_results[i][i1]][1]
    x.append(xy[final_route[i1]][0])
    y.append(xy[final_route[i1]][1])
            #plt.scatter(x, y, label= "cluster 1", color= "green", marker= "*", s=30)
            #plt.plot(x, y)
plt.scatter(x, y, label= "re-link", color= "green", marker= "*", s=30)
plt.plot(x,y,linestyle='solid',color='blue')

#for i in range(k):
#    for j in range(len(cluster_results[i])):
#        x.append(xy[cluster_results[i][j]][0])
#        y.append(xy[cluster_results[i][j]][1])
#        n = clr[i]
#        plt.plot(x,y,linestyle='solid',color = n)
    #x[:] = []
    #y[:] = []

for i in range(k):
    for i1 in range(len(center_p)):
        x,y = center_p[i1][0],center_p[i1][1]
        plt.scatter(x, y, label= "center", color= "red", marker= "o", s=20)
for i in range(k):
    x,y = center_p[i][0],center_p[i][1]
    plt.annotate(text1[i], (x, y + 0.2))
    
x,y = xy[0][0],xy[0][1]
plt.scatter(x, y, color= "green", marker= "s", s=30)
plt.annotate("Depot", (x, y + 0.2))


x = []
y = []
#x.append(xy[final_route[0]][0])
#y.append(xy[final_route[0]][1])
#x.append(xy[0][0])
#y.append(xy[0][1])
#x.append(xy[final_route[-1]][0])
#y.append(xy[final_route[-1]][1])
for i in range(k):
    x.append(center_p[i][0])
    y.append(center_p[i][1])
    x.append(xy[0][0])
    y.append(xy[0][1])
    plt.plot(x,y,linestyle='solid',color='yellow')


for i in range(0,len(text)):
    x,y = xy[i][0],xy[i][1]
    plt.annotate(text[i], (x, y + 0.2))  #### put name for each point

    #plt.figure(figsize=(12,8))
    # x-axis label
#plt.xlim([0, 190])
#plt.ylim([0, 180])
    
##plt.xlim([0, 2080])
##plt.ylim([0, 2580])
    
#plt.xlim([0, 4300])
#plt.ylim([0, 4500])

plt.xlabel('x - axis')
    # frequency label
plt.ylabel('y - axis')
    # plot title
    #plt.title('Before clusters of bays29 problems')
    # showing legend
#plt.legend()
  
    # function to show the plot
   
plt.show()

