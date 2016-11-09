# -*- coding: utf-8 -*-
"""
Created

@author: frank, mmitri

"""

import numpy as np
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
import string
import os
from subprocess import Popen, PIPE, STDOUT
import sys
import math
from random import randint
import itertools
import random
import copy
import operator

'''
#==============================================================================
#            GRAPH PERTURBATION MODELS

#  models:
# From : Abhijin Adiga, Anil Kumar S. Vullikanti, 2013, "How Robust Is the Core of a Network?"
  - Random perturbation (ERP) Erdos-Rényi random graph
  - Degree assortative perturbation (CLP) Chung-Lu random graph model
  - Link prediction based model (LPP), we use a different version than Clauset et al.
  see : Lichtenwalter et al. , 2010, "New Perspectives and Methods in Link Prediction"

# From : Newman et al., 2008, "Robustness of Community Structure in Networks"
They restrict the perturbed networks to having the SAME numbers of vertices and edges as the original unperturbed network (same degrees sequence also),
only the positions of the edges will be perturbed.
  - Configuration model
#============================================================================== '''

"""
MODEL 1 : Configuration model (which is also the null model normally used in the definition  of the modularity)
"""
def configuration_model_perturbation(G, alpha, dict_nodes_degrees_CONFIG, all_edges_degrees_product, list_of_existing_edges):
    H = G # ATTENTION !!!!! H = G.copy() gives strange results (i.e. for prob=0, NMI of spectral clust = 0.08... !!! same for fastgreedy)
    H_copy = G.copy() # Very important to do a COPY !!!
    n_nodes = H.number_of_nodes()  
    n_edges = H.number_of_edges()  
    # The probability of any particular edge falling between vertices i and j is e_ij/m 
    # The expected number of edges between vertices in the configuration model is : e_ij=KiKj/2m 
    # We go through each edge in the original network in turn and with proba epsilon we remove it and replace it with a new edge between a pair of vertices (i,j)
    # chosen randomly with proba e_ij/m
    prob = alpha/float(100)
    nb_edges_to_rewire = int(prob*n_edges)
    print '\n Probability of removing an edge:', prob

    # Iterator over all possible edges
    #all_edges = itertools.combinations(range(n_nodes),2)
    ''' !!!!!! itertools.combinations (along with many of the other itertools methods) return a generator expression. Generators can only be read exactly 1 time.'''
    ''' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    all_edges_list = list(itertools.combinations(range(n_nodes),2))
    

    '''  all_edges_degrees_product is equivalent to the squared distances in kmeans++ algo closest_dist_sq '''
    
    current_pot = all_edges_degrees_product.sum()
    cumulative_pot = all_edges_degrees_product.cumsum()

    # Iterator over all EXISTING edges
    nb_rewired = 0
    already_chosen_edges = []
       
    while nb_rewired < nb_edges_to_rewire:
        
        # Choose randomly an existing edge
        alea = np.random.randint(0, n_edges-1)
        
        if alea not in already_chosen_edges:
            
            already_chosen_edges.append(alea)
            e = list_of_existing_edges[alea]
            
            # because if degree=1 and we delete edge, node will be removed from graph, bug in nmi computation
            H_copy.remove_edge(str(e[0]),str(e[1]))
            #if (dict_nodes_degrees_CONFIG[e[0]]>1 and dict_nodes_degrees_CONFIG[e[1]]>1):
            if nx.is_connected(H_copy):
                
                H.remove_edge(str(e[0]),str(e[1]))
                # Count the number of rewired edges
                nb_rewired = nb_rewired + 1
                
                # Replace deleted edge with new one between a pair of vertices (i,j) chosen randomly with proba K(i)K(j)
                ''' Inspired by the k-means++ algo '''
                rand_val = np.random.random_sample(1) * current_pot
                
                # Attention, rand_val is an array
                candidate_id = np.searchsorted(cumulative_pot, rand_val[0])
                
                candidate_edge = all_edges_list[candidate_id]
                
                # !!! Add only non existing edges to keep the same number of edges
                if not H.has_edge(*(str(candidate_edge[0]), str(candidate_edge[1]))): # G.has_edge(*e)  #  e is a 2-tuple (u,v)
                    H.add_edge(str(candidate_edge[0]), str(candidate_edge[1]))
                else:
                    already_exist=True
                    # Take the next candidate until edge doesn't exist
                    while already_exist:
                        candidate_id = candidate_id + 1
                        candidate_edge = all_edges_list[candidate_id]
                        
                        if not H.has_edge(*(str(candidate_edge[0]), str(candidate_edge[1]))): # G.has_edge(*e)  #  e is a 2-tuple (u,v)
                            H.add_edge(str(candidate_edge[0]), str(candidate_edge[1]))
                            # Exit while
                            already_exist = False
    
            # we want H_copy to stay identical to H
            elif not nx.is_connected(H_copy):
                H_copy.add_edge(str(e[0]),str(e[1]))
            
    print 'Nb rewired :::::: ?',(nb_rewired/float(n_edges))
    return H
        
"""
MODEL 2 : Noise based on uniform perturbation (Erdos-Renyi G(n,e/n) model)
Returns the perturbed graph
"""
def uniform_perturbation(G, epsilon, addORdel):
    H = G # ATTENTION !!!!! H = G.copy() gives strange results (i.e. for prob=0, NMI of spectral clust = 0.08... !!! same for fastgreedy)
    H_copy = G.copy() # Very important to do a COPY !!!
    n_nodes = H.number_of_nodes()  
    prob = epsilon/float(n_nodes) # probability of edge addition
    print '\n Probability of addition/deletion :', prob
    
    if addORdel=='Add':
        # Iterator over all possible edges
        all_edges = itertools.combinations(range(n_nodes),2)
        # For all possible edges: add e with prob. prob
        for e in all_edges:
            if random.random() < prob:
                #if H.has_node(str(e[0])) and H.has_node(str(e[1])):
                H.add_edge(str(e[0]), str(e[1]))   
                
        return H
    
    elif addORdel=='Delete':
        # Iterate over all possible EXISTING edges
        edges_to_delete = []

        for e in H.edges_iter():
            if random.random() < prob:
                edges_to_delete.append(e)

        # because edges_iter() is returning an error when deleting edges
        # H.remove_edges_from(edges_to_delete)
        for e in edges_to_delete:
            # because if degree=1 and we delete edge, node will be removed from graph, bug in nmi computation
            H_copy.remove_edge(str(e[0]),str(e[1]))
            #if (H.degree(e[0])>1 and H.degree(e[1])>1):
            if nx.is_connected(H_copy):
                H.remove_edge(str(e[0]),str(e[1]))
            # we want H_copy to stay identical to H
            elif not nx.is_connected(H_copy):
                H_copy.add_edge(str(e[0]),str(e[1]))
                
        return H


"""
MODEL 3 : Noise based on preferential perturbation to high degree nodes (Chung-Lu model)
Returns the perturbed graph
"""
def preferential_perturbation(G, epsilon, addORdel, dict_nodes_degrees_CLP):
    H = G # ATTENTION !!!!! H = G.copy() gives strange results (i.e. for prob=0, NMI of spectral clust = 0.08... !!! same for fastgreedy)
    H_copy = G.copy() # Very important to do a COPY !!!
    n_nodes = H.number_of_nodes() 
    prob = epsilon/float(n_nodes) # probability of edge addition
    avg_degree= np.average(list(H.degree().values())) # avg degree of graph

    if addORdel=='Add':
        # Iterator over all possible edges
        all_edges = itertools.combinations(range(n_nodes),2)
        # For all possible edges: add e=(u,v) with prob.: ((du * dv) / d^2_avd) * prob
        for e in all_edges:
            prob_edge_add = ((dict_nodes_degrees_CLP[str(e[0])] * dict_nodes_degrees_CLP[str(e[1])]) / pow(avg_degree, 2)) * prob
            if random.random() < prob_edge_add:
                #if H.has_node(str(e[0])) and H.has_node(str(e[1])):
                H.add_edge(str(e[0]), str(e[1]))
        return H

    elif addORdel=='Delete':
        # Iterator over all EXISTING edges
        edges_to_delete = []
        
        for e in H.edges_iter():
            prob_edge_del = ((dict_nodes_degrees_CLP[e[0]] * dict_nodes_degrees_CLP[e[1]]) / pow(avg_degree, 2)) * prob
            if random.random() < prob_edge_del:
                edges_to_delete.append(e)
                
        for e in edges_to_delete:
            # because if degree=1 and we delete edge, node will be removed from graph, bug in nmi computation
            H_copy.remove_edge(e[0],e[1])
            #if (dict_nodes_degrees_CLP[e[0]]>1 and dict_nodes_degrees_CLP[e[1]]>1):
            if nx.is_connected(H_copy):
                H.remove_edge(e[0],e[1])
            # we want H_copy to stay identical to H
            elif not nx.is_connected(H_copy):
                H_copy.add_edge(e[0],e[1])  
                
        return H
        

"""
MODEL 4 : Noise based on link prediction using the Katz centrality measure.
list_predicted_edges_prob: sorted list in descending order (src dst) based on
the predicted probability (OUTPUT OF the software LPmade)
ATTENTION : RUN LPmade before to get list_predicted_edges_prob

Returns the perturbed graph
"""
def link_prediction_perturbation(G, epsilon, list_predicted_edges_prob):
    H = G # ATTENTION !!!!! H = G.copy() gives strange results (i.e. for prob=0, NMI of spectral clust = 0.08... !!! same for fastgreedy)
    n_nodes = H.number_of_nodes()
    # Number of edges to be added
    n_edges_to_add = int(math.ceil((n_nodes * epsilon) / 2))
    # Add edges from the list of predicted edges    
    H.add_edges_from(list_predicted_edges_prob[:n_edges_to_add])
    print "Number of nodes in H: ", H.number_of_nodes()
    return H


"""
Returns the Jaccard index of two sets (lists) A, B
"""
def jaccard_index(A, B):
    n = len(set(A).intersection(set(B)))    
    return n / float(len(set(A)) + len(set(B)) - n)



"""
Code for evaluating core and truss robustness under uniform perturbation
Execute noise model for 100 times - get mean value
"""
def evaluate_uniform_perturbation(G, epsilon_values): 
    
    # Core and truss decomposition of Initial graph G  
    max_truss_nodes_Init = truss_decomposition(G)
    max_core_nodes_Init = kcore_decomposition(G)
    
    n_added_edges = [] # Number of edges in the graph per epsilon (added by noise)
    truss_jaccard_idx = [] # Jaccard indices per epsilon for max truss
    core_jaccard_idx = [] # Jaccard indices per epsilon for max core
    
    for e in epsilon_values:
        print "nodes in G: ", G.number_of_nodes()
        print "edges in G: ", G.number_of_edges()
        H = uniform_perturbation(G, e)
        n_added_edges.append(H.number_of_edges() - G.number_of_edges())
        max_truss_nodes_H = truss_decomposition(H)
        max_core_nodes_H = kcore_decomposition(H) 
        truss_jaccard_idx.append(jaccard_index(max_truss_nodes_Init, max_truss_nodes_H))              
        core_jaccard_idx.append(jaccard_index(max_core_nodes_Init, max_core_nodes_H))
        
    return n_added_edges,truss_jaccard_idx,core_jaccard_idx


"""
Code for evaluating core and truss robustness under preferential (deg assortative) 
perturbation. Execute noise model for 100 times - get mean value
"""
def evaluate_preferential_perturbation(G, epsilon_values):  
    
    # Core and truss decomposition of Initial graph G  
    max_truss_nodes_Init = truss_decomposition(G)
    max_core_nodes_Init = kcore_decomposition(G)
    
    n_added_edges = [] # Number of edges in the graph per epsilon (added by noise)
    truss_jaccard_idx = [] # Jaccard indices per epsilon for max truss
    core_jaccard_idx = [] # Jaccard indices per epsilon for max core
    
    for e in epsilon_values:
        print "nodes in G: ", G.number_of_nodes()
        print "edges in G: ", G.number_of_edges()
        H = preferential_perturbation(G, e)
        n_added_edges.append(H.number_of_edges() - G.number_of_edges())
        max_truss_nodes_H = truss_decomposition(H)
        max_core_nodes_H = kcore_decomposition(H) 
        truss_jaccard_idx.append(jaccard_index(max_truss_nodes_Init, max_truss_nodes_H))              
        core_jaccard_idx.append(jaccard_index(max_core_nodes_Init, max_core_nodes_H))
        
    return n_added_edges,truss_jaccard_idx,core_jaccard_idx
    
    
"""
Code for evaluating core and truss robustness under edge perturbation given 
by a link prediction algorithm (Katz)

Input: G
      link_pred_file.pred (as produced by LPMade) sorted based on the 
      probability (run bash commands before)
"""
def evaluate_link_prediction_perturbation(G, link_pred_file):
    # Read .pred file and get the sorted list of predicted edges
    predicted_edges = []
    fp = open(link_pred_file,"r")    
    for line in fp.readlines():
        inpList=string.split(line)
        src = str(inpList[0])
        dst = str(inpList[1])
        pred_probability = inpList[2]
        # Create a tuple per edge
        edge_tuple = ();
        edge_tuple = (src, dst);
        predicted_edges.append(edge_tuple) # Add edge to the list        
    fp.close()
    
    # Core and truss decomposition of Initial graph G  
    max_truss_nodes_Init = truss_decomposition(G)
    max_core_nodes_Init = kcore_decomposition(G)
    
    epsilon_values = [40] # Epsilon (noise) values
    n_added_edges = [] # Number of edges in the graph per epsilon (added by noise)
    truss_jaccard_idx = [] # Jaccard indices per epsilon for max truss
    core_jaccard_idx = [] # Jaccard indices per epsilon for max core
    
    # Repeat for several levels of noise
    for e in epsilon_values:
        print "nodes in G: ", G.number_of_nodes()
        print "edges in G: ", G.number_of_edges()
        H = link_prediction_perturbation(G, e, predicted_edges)
        n_added_edges.append(H.number_of_edges() - G.number_of_edges())
        max_truss_nodes_H = truss_decomposition(H)
        max_core_nodes_H = kcore_decomposition(H) 
        truss_jaccard_idx.append(jaccard_index(max_truss_nodes_Init, max_truss_nodes_H))              
        core_jaccard_idx.append(jaccard_index(max_core_nodes_Init, max_core_nodes_H))    
    return epsilon_values,n_added_edges,truss_jaccard_idx,core_jaccard_idx

"""
Main function
"""
if __name__ == "__main__":

    print "Hello World!"
    filename = sys.argv[1]
    filename = filename.split(".")[0]
    
    G = read_graph(filename+'.txt')
    print "nodes in G: ", G.number_of_nodes()
    print "edges in G: ", G.number_of_edges()
    
#   # Just for testing the size of the core vs. truss
#    core = kcore_decomposition(G)
#    truss = truss_decomposition(G)
#    
#    res = [val for val in core if val in truss]
#    
#    print "Common nodes ", res
#    print "Number of common", len(res)
        
    
    # Uniform perturbation
    epsilon_values = [0.4,0.8,1.2,1.6,2,2.5,3, 3.5,4] # Epsilon (noise) values
    n_added_edges = [0]*len(epsilon_values)
    truss_jaccard_idx = [0]*len(epsilon_values)
    core_jaccard_idx = [0]*len(epsilon_values)
    for i in range(10):
        print "========= Iteration No " + str(i+1) + " ============="
        e,t,c = evaluate_uniform_perturbation(G, epsilon_values)
        # Get sum over all iterations - used in average
        n_added_edges = map(operator.add, n_added_edges, e)
        truss_jaccard_idx = map(operator.add, truss_jaccard_idx, t)
        core_jaccard_idx = map(operator.add, core_jaccard_idx, c)
    
    # Get average
    for i in range(len(epsilon_values)):
        n_added_edges[i] = n_added_edges[i] / 10.0
        truss_jaccard_idx[i] = truss_jaccard_idx[i] / 10.0
        core_jaccard_idx[i] = core_jaccard_idx[i] / 10.0
    
    # Plot
    plt.figure(1)
    plt.plot(epsilon_values, core_jaccard_idx, 'b^--', linewidth=2, label='Core')
    plt.plot(epsilon_values, truss_jaccard_idx, 'rs--', linewidth=2, label='Truss')
    plt.xlabel('Epsilon values')
    plt.ylabel('Jaccard index')
    plt.legend(loc='lower left', numpoints = 1 )
    plt.spines['right'].set_visible(False)
    plt.spines['top'].set_visible(False)
    plt.xaxis.set_ticks_position('bottom')
    plt.yaxis.set_ticks_position('left')
    plt.draw()
    plt.savefig(filename+'_truss_core_random.pdf')
    print "Random - truss: ", truss_jaccard_idx
    print "Random - core: ", core_jaccard_idx
    print "Random - Edges added: ", n_added_edges
    del n_added_edges,truss_jaccard_idx,core_jaccard_idx
    
    
    # Preferential perturbation
    epsilon_values = [0.4,0.8,1.2,1.6,2,2.5,3, 3.5,4] # Epsilon (noise) values
    n_added_edges = [0]*len(epsilon_values)
    truss_jaccard_idx = [0]*len(epsilon_values)
    core_jaccard_idx = [0]*len(epsilon_values)
    for i in range(10):
        print "========= Iteration No " + str(i+1) + " ============="
        e,t,c = evaluate_preferential_perturbation(G, epsilon_values)
        # Get sum over all iterations - used in average
        n_added_edges = map(operator.add, n_added_edges, e)
        truss_jaccard_idx = map(operator.add, truss_jaccard_idx, t)
        core_jaccard_idx = map(operator.add, core_jaccard_idx, c)

    # Get average
    for i in range(len(epsilon_values)):
        n_added_edges[i] = n_added_edges[i] / 10.0
        truss_jaccard_idx[i] = truss_jaccard_idx[i] / 10.0
        core_jaccard_idx[i] = core_jaccard_idx[i] / 10.0

    # Plot
    plt.figure(2)
    plt.plot(epsilon_values, core_jaccard_idx, 'b^--', linewidth=2, label='Core')
    plt.plot(epsilon_values, truss_jaccard_idx, 'rs--', linewidth=2, label='Truss')
    plt.xlabel('Epsilon values')
    plt.ylabel('Jaccard index')
    plt.legend(loc='lower left', numpoints = 1 )
    plt.spines['right'].set_visible(False)
    plt.spines['top'].set_visible(False)
    plt.xaxis.set_ticks_position('bottom')
    plt.yaxis.set_ticks_position('left')
    plt.draw()
    plt.savefig(filename+'_truss_core_preferential.pdf')
    print "Preferential - truss: ", truss_jaccard_idx
    print "Preferential - core: ", core_jaccard_idx
    print "Preferential - Edges added: ", n_added_edges
    del epsilon_values,n_added_edges,truss_jaccard_idx,core_jaccard_idx
    
    
    # Perturbation based on link prediction
#    fname = 'Data_Link_Prediction_LPMade/' + filename + '_sorted_descending.pred'
#    epsilon_values,n_added_edges,truss_jaccard_idx,core_jaccard_idx = \
#                                evaluate_link_prediction_perturbation(G, fname)
#    # Plot
#    plt.figure(3)
#    plt.plot(epsilon_values, core_jaccard_idx, 'b^--', linewidth=2, label='Core')
#    plt.plot(epsilon_values, truss_jaccard_idx, 'rs--', linewidth=2, label='Truss')
#    plt.xlabel('Epsilon values')
#    plt.ylabel('Jaccard index')
#    plt.legend(loc='lower left', numpoints = 1 )
#    plt.spines['right'].set_visible(False)
#    plt.spines['top'].set_visible(False)
#    plt.xaxis.set_ticks_position('bottom')
#    plt.yaxis.set_ticks_position('left')
#    plt.draw()
#    plt.savefig(filename+'_truss_core_link_prediction.pdf')
#    print "truss ", truss_jaccard_idx
#    print "core ", core_jaccard_idx
#    print "Edges added: ", n_added_edges
#    del epsilon_values,n_added_edges,truss_jaccard_idx,core_jaccard_idx    
    # Compare results

    
