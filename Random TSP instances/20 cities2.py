# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:15:46 2020

@author: Malek
"""
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import entropy
import pandas as pd
import random
import tsplib95

fitness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
entrop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for k in range(30):
    dist = np.zeros((20,20))
    for i in range(19):
        for j in range(i+1, 20):
           dist[i][j] = random.uniform(1,5)  
    dist_list = [(0,0,0) for x in range(190)]  
    l = 0
    #for l in range(1000):
    while l < 190:   
        for i in range(19):
            for j in range(i+1, 20):
                dist_list[l] = (i,j, dist[i][j])
                print(dist_list[l])
                l = l+1
    print(dist_list)
    # Initialize fitness function object using dist_list
    fitness_dists = mlrose.TravellingSales(distances = dist_list)
    
    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length = 20, fitness_fn = fitness_dists, maximize=False)

    best_state, best_fitness, fitness[k] = mlrose.random_walk(problem_fit)            
    print(best_state)
    
    print(best_fitness)
    
    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
    print("The fitness curve is:", fitness[k])
    entrop[k] = entropy(fitness[k], base=6)    
    print("The entropy for th k excutions is:", entrop[k])


print(entrop)
df = pd.DataFrame(entrop)
df.to_excel("TSP_random_20.xlsx", index=True, startrow=0)

#pA = test / test.sum()
#Shannon2 = -np.sum(pA*np.log6(pA))

#test = pd.Series(sm.tsa.acf(test, nlags=100))
#df = pd.DataFrame(test)
#test2 = pd.Series(sm.tsa.acf(test2, nlags=100))
#df2 = pd.DataFrame(test2)
#plot_acf(test, lags=100)

#df.to_excel("TSP_LH31.xlsx", index=True, startrow=0)
