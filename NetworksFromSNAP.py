# -*- coding: utf-8 -*-
"""


@author: mmitri
"""
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx import *
from Add_Noise import *
from scipy.sparse import coo_matrix
from numpy import hstack, ones
from array import array
import sys, os

def swap(a,b):
    if a > b:
        return b,a
    return a,b

def graph_to_sparse_matrix(graph):
    xs, ys = map(array, zip(*graph.get_edgelist()))
    if not graph.is_directed():
        xs, ys = hstack((xs, ys)).T, hstack((ys, xs)).T
    else:
        xs, ys = xs.T, ys.T
    return coo_matrix((ones(xs.shape), (xs, ys)))
    
def dict_nodes_degrees(graph):
    # Dictionnary that will map node_id --> degree (to speed up CLP noise model becaue he has to compute degrees for all combinations at every call !!!!!!)
    dict_nodes_degrees = dict()
    for node in graph.nodes_iter():
        dict_nodes_degrees[node] = graph.degree(node)
    
    return dict_nodes_degrees 
    

def convert_1(filename , new_graph_destination , perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t' , nodetype=str):
    """reads two-column edgelist, returns mapping satisfying those requirements :
    Specifically, the network topology must conform to the following requirements:
• .pairs is a list of tab-delimited pairs of numeric indices, e.g., "54\t91\n"
• the network described is a SINGLE COMPONENT
• there are NO SELF-LOOPS or MULTI-EDGES in the file
• the MINIMUM NODE ID = 0 in the input file
• the MAXIMUM NODE ID can be anything, but the program will use less memory if nodes are labeled sequentially

    """
    edges = set()
    all_nodes = []
    sequ_labeled_all_nodes = {}
    G = nx.Graph() # Base class for undirected graphs
    
    ''' Reading the text file '''

    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        
        if ("#" in L[0]):  # to skip comments
            pass
        else:     
            ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
            
            G.add_edge(ni,nj)
            #if ni != nj: # skip any self-loops...
                # Construct the graph with networkX
                #G.add_edge(ni,nj)
                # or with python set() : # edges is a set() so no repetitions (by definition of set() ) an due to swap - counts edges (u,v) et (v,u) ONLY ONCE !
                # edges.add( swap(ni,nj) ) 
            
    # Remove self loops          
    G.remove_edges_from(G.selfloop_edges())  
        
    ''' Extracting the greatest connected component (GCC) '''
    
    GCC = max(nx.connected_component_subgraphs(G), key=len)
    
#    if (perturb_model == None):
#        n = GCC.number_of_nodes()
#        m = GCC.number_of_edges()
#    
#        print '\n Number of nodes in GCC (undirected):', n
#        print '\n Number of edges in GCC (undirected):', m
    

    ''' Converts node ids in GCC to the range [0 - |V|-1] '''
    # Node ids may changed and not be in 0 - N-1 range since GCC was extracted.
    
    GCC_relabeled = nx.Graph()  # The new graph (GCC+sequentially labeled nodes)
    nid = 0  # Create a dictionnary with (key=index,value=node) to have a sequentially labeled nodes starting from index = 0 !!!
    node_ID = dict()

    for e in GCC.edges_iter():
        ni = e[0]
        nj = e[1]    
        if (not node_ID.has_key(ni)):  # If the node does not exist in the dict
            node_ID[ni] = nid
            nid = nid + 1
        if (not node_ID.has_key(nj)):  # If the node does not exist in the dict
            node_ID[nj] = nid
            nid = nid + 1
        
        # Add edge    
        GCC_relabeled.add_edge(str(node_ID[ni]), str(node_ID[nj]))
    
    del GCC
    G = nx.Graph()
        
    ''' Doing graph perturbation '''
    
    if (perturb_model != None):
        
        if perturb_model=='ERP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            G = uniform_perturbation(GCC_relabeled, epsilon, addORdel)
            
            
        elif perturb_model=='CLP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # To speed up the noise model
            if hasattr(convert_1, 'dict_nodes_degrees_CLP'):
                pass
            else:
                dict_nodes_degrees_CLP = dict_nodes_degrees(GCC_relabeled)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.dict_nodes_degrees_CLP = dict_nodes_degrees_CLP
            
            G = preferential_perturbation(GCC_relabeled, epsilon, addORdel, dict_nodes_degrees_CLP = convert_1.dict_nodes_degrees_CLP)
            
            
        elif perturb_model=='LPP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # Read .pred file and get the sorted list of predicted edges
            #link_pred_file.pred (as produced by LPMade) sorted based on the probability (run bash commands before)
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
            
            G = link_prediction_perturbation(GCC_relabeled, epsilon, predicted_edges)
            
            
        elif perturb_model=='CONFIG':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # To speed up the noise model
            if (hasattr(convert_1, 'dict_nodes_degrees_CONFIG')):
                pass
            else:
                ''''''
                dict_nodes_degrees_CONFIG = dict_nodes_degrees(GCC_relabeled)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.dict_nodes_degrees_CONFIG = dict_nodes_degrees_CONFIG
                
            if hasattr(convert_1, 'all_edges_degrees_product'):
                pass
            else:
                ''''''
                all_edges = itertools.combinations(range(GCC_relabeled.number_of_nodes()),2)
                all_edges_degrees_product = []
                for e in all_edges:
                    # Replace deleted edge with new one between a pair of vertices (i,j) chosen randomly with proba K(i)K(j)
                    proba_config = (dict_nodes_degrees_CONFIG[str(e[0])]*dict_nodes_degrees_CONFIG[str(e[1])])
                    all_edges_degrees_product.append(proba_config)
                
                all_edges_degrees_product = np.asarray(all_edges_degrees_product)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.all_edges_degrees_product = all_edges_degrees_product

            if hasattr(convert_1, 'list_of_existing_edges'):
                pass
            else:
                ''''''
                list_of_existing_edges = []
                
                for e in GCC_relabeled.edges_iter():
                    list_of_existing_edges.append(e)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.list_of_existing_edges = list_of_existing_edges
            
            G = configuration_model_perturbation(GCC_relabeled, epsilon, dict_nodes_degrees_CONFIG = convert_1.dict_nodes_degrees_CONFIG, all_edges_degrees_product = convert_1.all_edges_degrees_product , list_of_existing_edges = convert_1.list_of_existing_edges)
        
        
    else:
        G = GCC_relabeled

    
    """ Writing the graph in Fast Greedy Modularity Maximization format """
    
    f = open(new_graph_destination,'w')  # 'w' for only writing (an existing file with the same name will be erased)
    
    for edge in G.edges_iter():
        f.write(edge[0])
        f.write(delimiter)
        f.write(edge[1])
        f.write('\n') # new edge
        # e.g., "54\t91\n"
    f.close()
    
    # Return the graph also
    return G

def convert_to_igraph_format(source_path, convert_path , algo_name , net_name ,  perturb_model=None , epsilon=None , addORdel=None , link_pred_file=None):
    
    """ Chose the algo"""
    if algo_name == 'iGraph_algos':
        
        """ Chose the network"""
        if  net_name == 'Email-Enron':
            net_path = source_path+'Email-Enron.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'CA-GrQc':
            net_path = source_path+'CA-GrQc.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'CA-AstroPhys':
            net_path = source_path+'CA-AstroPh.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'Wiki-Vote':
            net_path = source_path+'Wiki-Vote.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)

        if  net_name == 'AS-Caida':
            net_path = source_path+'as-caida20040105.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'CA-HepTh':
            net_path = source_path+'CA-HepTh.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'P2P-GNutella':
            net_path = source_path+'p2p-Gnutella08.txt'
            graph = convert_1(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        return graph
        
        
    """ Chose the algo"""
    if algo_name == 'METIS':
        
        """ Chose the network"""
        if  net_name == 'Email-Enron':
            net_path = source_path+'Email-Enron.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'CA-GrQc':
            net_path = source_path+'CA-GrQc.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'CA-AstroPhys':
            net_path = source_path+'CA-AstroPh.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
            
        if  net_name == 'Wiki-Vote':
            net_path = source_path+'Wiki-Vote.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)

        if  net_name == 'AS-Caida':
            net_path = source_path+'as-caida20040105.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)

        if  net_name == 'CA-HepTh':
            net_path = source_path+'CA-HepTh.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)

        if  net_name == 'P2P-GNutella':
            net_path = source_path+'p2p-Gnutella08.txt'
            nb_nodes = convert_to_METIS_graph_format(net_path,convert_path, perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t',nodetype=str)
        
    
        return nb_nodes
    
    
'''============================================================================
                                     METIS
============================================================================'''

def convert_to_METIS_graph_format(filename,metis_graph_destination , perturb_model , epsilon , addORdel , link_pred_file , delimiter='\t' , nodetype=str):
    """reads two-column edgelist, returns mapping node -> set of neighbors for METIS algorithm
    """
    adj = defaultdict(set) # node to set of neighbors
    edges = set()
    all_nodes = []
    sequ_labeled_all_nodes = {}
    G = nx.Graph() # Base class for undirected graphs
    
    ''' Reading the text file '''

    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        
        if ("#" in L[0]):  # to skip comments
            pass
        else:     
            ni,nj = nodetype(L[0]),nodetype(L[1]) # other columns ignored
            
            G.add_edge(ni,nj)
            #if ni != nj: # skip any self-loops...
                # Construct the graph with networkX
                #G.add_edge(ni,nj)
                # or with python set() : # edges is a set() so no repetitions (by definition of set() ) an due to swap - counts edges (u,v) et (v,u) ONLY ONCE !
                # edges.add( swap(ni,nj) ) 
            
    # Remove self loops          
    G.remove_edges_from(G.selfloop_edges())  
        
    ''' Extracting the greatest connected component (GCC) '''
    
    GCC = max(nx.connected_component_subgraphs(G), key=len)
    
#    if (perturb_model == None):
#        n = GCC.number_of_nodes()
#        m = GCC.number_of_edges()
#    
#        print '\n Number of nodes in GCC (undirected):', n
#        print '\n Number of edges in GCC (undirected):', m
    

    ''' Converts node ids in GCC to the range [0 - |V|-1] '''
    # Node ids may changed and not be in 0 - N-1 range since GCC was extracted.
    
    GCC_relabeled = nx.Graph()  # The new graph (GCC+sequentially labeled nodes)
    nid = 0  # Create a dictionnary with (key=index,value=node) to have a sequentially labeled nodes starting from index = 0 !!!
    node_ID = dict()

    for e in GCC.edges_iter():
        ni = e[0]
        nj = e[1]    
        if (not node_ID.has_key(ni)):  # If the node does not exist in the dict
            node_ID[ni] = nid
            nid = nid + 1
        if (not node_ID.has_key(nj)):  # If the node does not exist in the dict
            node_ID[nj] = nid
            nid = nid + 1
        
        # Add edge    
        GCC_relabeled.add_edge(str(node_ID[ni]), str(node_ID[nj]))
    
    del GCC
    G = nx.Graph()
        
    ''' Doing graph perturbation '''
    
    if (perturb_model != None):
        
        if perturb_model=='ERP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            G = uniform_perturbation(GCC_relabeled, epsilon, addORdel)
            
            
        elif perturb_model=='CLP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # To speed up the noise model
            if hasattr(convert_1, 'dict_nodes_degrees_CLP'):
                pass
            else:
                dict_nodes_degrees_CLP = dict_nodes_degrees(GCC_relabeled)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.dict_nodes_degrees_CLP = dict_nodes_degrees_CLP
            
            G = preferential_perturbation(GCC_relabeled, epsilon, addORdel, dict_nodes_degrees_CLP = convert_1.dict_nodes_degrees_CLP)
            
            
        elif perturb_model=='LPP':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # Read .pred file and get the sorted list of predicted edges
            #link_pred_file.pred (as produced by LPMade) sorted based on the probability (run bash commands before)
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
            
            G = link_prediction_perturbation(GCC_relabeled, epsilon, predicted_edges)
            
            
        elif perturb_model=='CONFIG':
            print '\n Perturbing the graph using ' + perturb_model + ' perturbation model'
            
            # To speed up the noise model
            if (hasattr(convert_1, 'dict_nodes_degrees_CONFIG')):
                pass
            else:
                ''''''
                dict_nodes_degrees_CONFIG = dict_nodes_degrees(GCC_relabeled)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.dict_nodes_degrees_CONFIG = dict_nodes_degrees_CONFIG
                
            if hasattr(convert_1, 'all_edges_degrees_product'):
                pass
            else:
                ''''''
                all_edges = itertools.combinations(range(GCC_relabeled.number_of_nodes()),2)
                all_edges_degrees_product = []
                for e in all_edges:
                    proba_config = (dict_nodes_degrees_CONFIG[str(e[0])]*dict_nodes_degrees_CONFIG[str(e[1])])
                    all_edges_degrees_product.append(proba_config)
                
                all_edges_degrees_product = np.asarray(all_edges_degrees_product)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.all_edges_degrees_product = all_edges_degrees_product
            
            if hasattr(convert_1, 'list_of_existing_edges'):
                pass
            else:
                ''''''
                list_of_existing_edges = []
                
                for e in GCC_relabeled.edges_iter():
                    list_of_existing_edges.append(e)
                # Functions are objects in Python and can have arbitrary attributes assigned to them.
                convert_1.list_of_existing_edges = list_of_existing_edges
            
            G = configuration_model_perturbation(GCC_relabeled, epsilon, dict_nodes_degrees_CONFIG = convert_1.dict_nodes_degrees_CONFIG, all_edges_degrees_product = convert_1.all_edges_degrees_product , list_of_existing_edges = convert_1.list_of_existing_edges)
       
    else:
        G = GCC_relabeled


    ''' Create dictionnary '''
        
    for e in G.edges_iter():
        ni = e[0]
        nj = e[1]
        # edges is a set() so no repetitions (by definition of set() ) an due to swap - counts edges (u,v) et (v,u) ONLY ONCE !
        # edges.add( swap(ni,nj) ) 
        
        # Adjacencies in METIS format 
        adj[ni].add(nj)
        adj[nj].add(ni) # since undirected
        
        all_nodes.append(int(ni))
        all_nodes.append(int(nj))
        
    
    # Create a dictionnary with (key=node,value=index) to have a sequentially labeled nodes starting from index=1 !!!
    ''' VERY IMPORTANT TO SORT for METIS FORMAT '''
    all_nodes = sorted(np.unique(all_nodes) , reverse=False) # we have the list of ordered nodes
    for index,node in enumerate(all_nodes):
        ''' ************************************************************************************ '''
        ''' ******* ATTENTION : first node's index = 1 for METIS (0 for all other algos) ******* '''
        sequ_labeled_all_nodes[str(node)] = index+1 
    

    """ Writing the graph in METIS format """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    f = open(metis_graph_destination,'w')
    
    # header line
    f.write(' ')
    f.write(str(n))
    f.write(' ')
    f.write(str(m))
    f.write('\n')
    
    # remaining n lines contain information for each vertex of the graph
    
    adj = dict(adj)
    
    ''' Converts node ids to the range [1 - |V|] for METIS '''
    for node in all_nodes: # ATTENTION : nodes ids are in integer format in all_nodes
        for neighbor in adj[str(node)]: # extracting neighbors from the set corresponding to key "node"
            f.write(' ')
            f.write(str(sequ_labeled_all_nodes[neighbor]))
            #f.write(str(neighbor))
            f.write(' ')
        f.write('\n') # new node
        
    f.close()
    
    return n


def METIS_Clustering(source_path, convert_path , network_name , npart ,  perturb_model=None , epsilon=None , addORdel=None , link_pred_file=None):
    ''' All the METIS clustering pipeline 
    npart = Number of cluster (parameter of METIS algo) 
    '''
    
    '''=============================================================================
    # WRITE GRAPHS IN METIS FORMAT
    #=============================================================================='''
    
    """ Convert graphs to METIS format """
    
    nb_nodes = convert_to_igraph_format(source_path, convert_path, algo_name='METIS' , net_name=network_name , perturb_model=perturb_model , epsilon=epsilon , addORdel=addORdel , link_pred_file=link_pred_file)
    
    '''=============================================================================
    # CLUSTERING WITH METIS
    #=============================================================================='''
    
    cmd = 'gpmetis '+convert_path+' '+str(npart)
    os.system(cmd) # returns the exit status
    
    '''=============================================================================
    # RESULTS
    #=============================================================================='''
    
    results = convert_path + '.part.' + str(npart)
    
    clusters = np.empty((nb_nodes))
    
    node_id = 0
    
    for line in open(results, 'U'):
        L = line.strip().split('\n')
        # In the results file of METIS, each line correspond to the node id and the value to the cluster the node belongs to
        clust_id = int(L[0])

        clusters[node_id] = clust_id
        # First node has index=1
        #clusters[clust_id] = clusters.get(clust_id, [])
        #clusters[clust_id].append(node_id)
        # nodes are sequentially ordered in the file (line num !!)
        node_id += 1
    
    return clusters
    


            
    
