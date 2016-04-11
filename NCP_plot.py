# -*- coding: utf-8 -*-
"""

@author: mmitri
"""

import numpy as np
import matplotlib.pyplot as plt
from igraph import *
from community_quality_NetworkX import *

def NCP_plot(graph, clusters , quality_measure , algos):
    ''' Compute the NCP plot. In order to more finely resolve community structure in large networks, we introduce the network community
    profile plot (NCP plot). Intuitively, the NCP plot measures the quality of the best possible
    community in a large network, as a function of the community size. Formally, we may define it as the
    conductance value of the best conductance set of cardinality k in the entire network, as a function of k.
    
    See Lekovec et al. 2008 : "Community Structure in Large Networks Natural Cluster Sizes and the Absence of Large Well-Defined Clusters"
    
    graph : the original graph
    clusters : a VertexClustering object if using iGraph algo or a dictionnary if other
    
    '''
    # Print number of clusters
    nb_clusters = len(clusters)
    print 'Number of clusters', len(clusters)
    
    # Initialization
    k_values = []
    NCP_values = {}
    NCP_values_min = []
    k_distribution = []

    if algos=='iGraph':
        # The VertexClustering object returned by cl.as_clustering() has a crossing() method - 
        # this gives you a Boolean vector which contains True for edges that are between clusters 
        # and False for edges that are within a cluster. You can easily extract the indices of the crossing edges like this:
        
        crossing_edges = [index for index, is_crossing in enumerate(clusters.crossing()) if is_crossing]
        
        left_end_point = []
        right_end_point = []
    
        # Iterate over all crossing edges. G.es is the edges sequence of G
        for edge in graph.es[crossing_edges]:
            left_end_point.append(str(edge.tuple[0]))
            right_end_point.append(str(edge.tuple[1]))
        
    
    # Get the number of vertices and edges in each subgraph and compute NCP
    for i in range(nb_clusters):
        
        if algos=='iGraph':
            k = clusters.subgraph(i).vcount() # k is cardinality of the subgraph, i.e. |S|, the number of vertices, the community size
            k_values.append(k)
            edges_in_subgraph = clusters.subgraph(i).ecount()
            
        elif algos=='non_iGraph':
            k = len(clusters[i]) # k is cardinality of the subgraph, i.e. |S|, the number of vertices, the community size
            k_values.append(k)
            
        
        if algos=='iGraph':
            if quality_measure=='InternalDensity':
            # InternalDensity : is the internal edge density of the node set S
            # InternalDensity= m(s)/(n(s)*(n(s)-1)/2)
                
                quality_value = edges_in_subgraph/float((k*(k-1)/2))
                
            
            if quality_measure=='Conductance':
                # Measures the fraction of total edge volume that points outside the cluster
                # phi(S)=s/v , v is the sum of degrees of nodes in S , s is the number of edges on the boundaries of S
                    
                # Iterate over all nodes in the subgraph and compute s and v
                s=0
                v=0
                
                for node in VertexSeq(clusters.subgraph(i)):
                    # noeud['name'] is actually the node's id (it's a trick because iGraph relabel nodes from 0 in each subgraph)
                    s = s + left_end_point.count(node['name']) + right_end_point.count(node['name'])
                    
                    v = v + graph.degree(node['name'])
                
                # Compute conductance
                quality_value = s/float(v)

        elif algos=='non_iGraph':
            
            if quality_measure=='Conductance':
                # Measures the fraction of total edge volume that points outside the cluster
                # phi(S)=s/v , v is the sum of degrees of nodes in S , s is the number of edges on the boundaries of S
                
                # Compute conductance
                c1_set = set(clusters[i]) # cluster i in list format with nodes id
                all_nodes = set(nx.nodes(graph))
                c2_set = all_nodes.difference(c1_set) # the rest of the graph !!
                
                c1 = list(c1_set)
                c2 = list(c2_set)
                
                quality_value = conductance(graph,c1,c2,weight=None)
            
            
        ''' Compute the NCP values, i.e. set(quality_value) per k. It's a dictionnary, key=k and value=set(of quality measure values) '''
       
        NCP_values[k] = NCP_values.get(k, [])
        NCP_values[k].append(quality_value)

    # Sort the set of k's
    k_values = sorted(np.unique(k_values) , reverse=False)
    
    # Compute min(quality_values) for each k
    for omega in k_values:
        
        min_quality = np.min(NCP_values[omega])
        
        ''' !!!!!!! We can use NCP_values_min directly to plot because k_values are already sorted !!!! '''
        NCP_values_min.append(min_quality)
        
        # To plot the distibution of cluster sizes
        k_distribution.append(len(NCP_values[omega]))
    
    # Get the frequency
    k_distribution[:] = [x/float(nb_clusters) for x in k_distribution]
    
#    # Plot
#    fig1 = plt.figure()
#
#    plt.loglog(k_values , NCP_values_min, linewidth=1 , marker='o')
#    plt.title('NCP plot of Leskovec 2008')
#    plt.xlabel('k (number of nodes in the cluster)')
#    plt.ylabel('Phi('+quality_measure+')')
#    plt.draw()
    
    
    return k_values , NCP_values_min , list(np.cumsum(k_distribution))
        

        
