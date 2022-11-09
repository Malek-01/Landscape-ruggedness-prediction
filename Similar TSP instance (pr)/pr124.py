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
import networkx

fitness = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
entrop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
problem = tsplib95.load('C:/Users/\Malek/Desktop/Evostar 2023/Experiment/Third experiments/jorlib-master/jorlib-master/jorlib-core/src/test/resources/tspLib/tsp/pr124.tsp')
graph = problem.get_graph()
distance_matrix = networkx.to_numpy_matrix(graph)
distance_matrix = np.squeeze(np.asarray(distance_matrix))

for k in range(30):
    dist = np.zeros((124,124))
    for i in range(123):
        for j in range(i+1, 124):
            dist[i][j] = distance_matrix[i][j] 
            dist_list = [(0,0,0) for x in range(7626)]  
    l = 0
    #for l in range(1000):
    while l < 7626:   
        for i in range(123):
            for j in range(i+1, 124):
                dist_list[l] = (i,j, dist[i][j])
                print(dist_list[l])
                l = l+1
    print(dist_list)
    # Initialize fitness function object using dist_list
    fitness_dists = mlrose.TravellingSales(distances = dist_list)
    
    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length = 124, fitness_fn = fitness_dists, maximize=False)

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
df.to_excel("TSP_pr_124.xlsx", index=True, startrow=0)