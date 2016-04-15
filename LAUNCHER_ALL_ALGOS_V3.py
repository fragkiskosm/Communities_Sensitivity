# -*- coding: utf-8 -*-
"""


@author: mmitri
"""

from igraph import *
import numpy as np
import networkx as nx
from NetworksFromSNAP import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import spectral_clustering
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from igraph import arpack_options
from NCP_plot import *
from normalized_laplacian_spectrum import *
from variation_of_information_score import *
from Plot_and_save_NCP_charts import *
from community_quality_NetworkX import *
import matplotlib as mpl

'''============================================================================
#    PARAMETERS
#=============================================================================='''

''' Chose algos'''

ifBlondelMultilevel = True

ifFastGreedyMaxModularity = True

ifINFOMAP = True

ifLeadingEigenvector = True
arpack_options.maxiter=1000000000

ifWalktrap = True

ifLabelPropagation = True

ifSpectralClustering = True

ifMETIS = True


algos_labels = []
algos_line_style = []
nb_active_algos = 8

''' Chose the network '''
network_name = ['Email-Enron' , 'CA-GrQc' , 'CA-AstroPhys' , 'Wiki-Vote' , 'AS-Caida', 'CA-HepTh', 'P2P-GNutella']
network_name = network_name[5]

''' Chose the number of run '''
nb_run = 2


''' Chose the noise model '''
noise_model = ['ERP' , 'CLP' , 'CONFIG']
noise_model = noise_model[0]

add_or_delete = 'Add' # Add or 'Delete'

if (noise_model != 'CONFIG'):
    if add_or_delete=='Delete':
        epsilon_valuesI = [ 0 , 0.01 , 0.2 , 0.4 , 0.6 , 0.8 , 1 , 1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5 , 6]
        epsilon_values = [i * 475 for i in epsilon_valuesI]
    elif add_or_delete=='Add':
        epsilon_valuesI = [ 0 , 0.01 , 0.2 , 0.4 , 0.6 , 0.8 , 1 , 1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5 , 6]
        epsilon_values = [i * 0.32 for i in epsilon_valuesI]
else:
    # Alpha values between 0 and 1 
    epsilon_values = np.linspace(0, 30, num=16)


''' Paths '''
source_path = '/home/mmitri/Documents/Stage/Data/Source/'

convert_path = '/home/mmitri/Documents/Stage/Data/Converted/' + network_name + '_iGraphFormat.txt'
convert_path_metis = '/home/mmitri/Documents/Stage/Data/Converted/' + network_name + '_MetisFormat.txt'

convert_path_P = '/home/mmitri/Documents/Stage/Data/Converted/' + network_name + '_iGraphFormat_Perturbed.txt'
convert_path_metis_P = '/home/mmitri/Documents/Stage/Data/Converted/' + network_name + '_MetisFormat_Perturbed.txt'

path_results = '/home/mmitri/Documents/Stage/Codes/Results/'


''' Evaluating the results '''
# Create empty 3D arrays fot saving the value of metrics quantifying clusters differences :
# arg1 = number of pages (i.e. depth, third dimension), i.e. number of algos, une page for each algo
# arg2 = rows, i.e. number of simulations
# arg3 = columns, i.e. number of values of epsilon
nmi_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
vi_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
ari_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
modularity_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
nb_clusters_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))

n_added_deleted_edges = np.empty((nb_run, len(epsilon_values)))

''' Enable saving results '''
ifSave = True

''' NCP plots parameters '''
scoring_function = 'Conductance' #'Conductance'

save__NCP_values_min_Louvain , save__NCP_values_min_FastGreedyMM , save__NCP_values_min_Infomap , save__NCP_values_min_LeadingEigen , save__NCP_values_min_Walktrap , save__NCP_values_min_LabelPropag , save__NCP_values_min_Spectral , save__NCP_values_min_Metis = {} , {} , {} , {} , {} , {} , {} , {}
save__NCP_k_values_Louvain , save__NCP_k_values_FastGreedyMM , save__NCP_k_values_Infomap , save__NCP_k_values_LeadingEigen , save__NCP_k_values_Walktrap , save__NCP_k_values_LabelPropag , save__NCP_k_values_Spectral , save__NCP_k_values_Metis = {} , {} , {} , {} , {} , {} , {} , {}
save__NCP_k_distribution_Louvain , save__NCP_k_distribution_FastGreedyMM , save__NCP_k_distribution_Infomap , save__NCP_k_distribution_LeadingEigen , save__NCP_k_distribution_Walktrap , save__NCP_k_distribution_LabelPropag , save__NCP_k_distribution_Spectral , save__NCP_k_distribution_Metis = {} , {} , {} , {} , {} , {} , {} , {}
    
Fiedler_value_per_noise = np.empty((nb_run, len(epsilon_values)))

'''============================================================================
#          BEGIN SIMULATIONS
#==========================================================================='''


''' ********  U ==> Unperturbed graph ******** '''
print '\n ******** Original graph ********' 

G_networkX = convert_to_igraph_format(source_path, convert_path, algo_name='iGraph_algos' , net_name=network_name)

G = Graph.Read_Edgelist(f=convert_path, directed=False)

# We set vertices name=index to be able to retrieve them when we extract sub-graphs associated with each cluster
# Because nodese ids are relabeled starting from 0 when subgraph extracted
for idx, v in enumerate(VertexSeq(G)):
    v['name'] = str(idx) # now each vertex has an attribute 'name'
    
print 'Summary original graph:'
summary(G)



''' ************************************************* DO CLUSTERING ****************************************'''
        
if ifBlondelMultilevel:
    
    ''' Clustering the original graph'''
    # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
    clusters_Louvain = G.community_multilevel()
    
    # Label for plot and other savings
    algos_labels.append('Louvain')
    algos_line_style.append('o')
    
    
if ifFastGreedyMaxModularity:
    
    ''' Clustering the original graph'''
    # Returns: an appropriate VertexDendrogram object.
    communities_FastGreedyMM = G.community_fastgreedy()
    
    """ Cuts the dendrogram at the given level and returns a corresponding VertexClustering object.
    
    as_clustering()
    
    Parameters:
    n - the desired number of clusters. Merges are replayed from the beginning until the membership vector has exactly n distinct elements 
    or until there are no more recorded merges, whichever happens first. If None, the optimal count hint given by the clustering algorithm will 
    be used If the optimal count was not given either, it will be calculated by selecting the level where the modularity is maximal. """
    
    # clusters is a VertexClustering object.
    clusters_FastGreedyMM = communities_FastGreedyMM.as_clustering()
    
    # Label for plot
    algos_labels.append('FastGreedyMM')
    algos_line_style.append('^')


if ifINFOMAP:

    ''' Clustering the original graph'''
    # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
    clusters_Infomap = G.community_infomap()
    
    #print '\ Number of clusters for original graph: ', len(clusters)
    #print '\n Modularity value after clustering (original G) is :', G.modularity(clusters)
    
    # Label for plot
    algos_labels.append('Infomap')
    algos_line_style.append('s')
    


if ifLeadingEigenvector:    
    
    ''' Clustering the original graph'''
    # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
    clusters_LeadingEigen = G.community_leading_eigenvector(arpack_options=arpack_options)
    
    #print '\ Number of clusters for original graph: ', len(clusters)
    #print '\n Modularity value after clustering (original G) is :', G.modularity(clusters)
    
    # Label for plot
    algos_labels.append('LeadingEigen')
    algos_line_style.append('p')
    

if ifWalktrap:
    
    ''' Clustering the original graph'''
    # Returns: an appropriate VertexDendrogram object.
    communities_Walktrap = G.community_walktrap()
    
    """ Cuts the dendrogram at the given level and returns a corresponding VertexClustering object.
    
    as_clustering()
    
    Parameters:
    n - the desired number of clusters. Merges are replayed from the beginning until the membership vector has exactly n distinct elements 
    or until there are no more recorded merges, whichever happens first. If None, the optimal count hint given by the clustering algorithm will 
    be used If the optimal count was not given either, it will be calculated by selecting the level where the modularity is maximal. """
    
    # clusters is a VertexClustering object.
    clusters_Walktrap = communities_Walktrap.as_clustering()
    
    # Label for plot
    algos_labels.append('Walktrap')
    algos_line_style.append('h')



if ifLabelPropagation:
    
    ''' Clustering the original graph'''
    # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
    clusters_LabelPropag = G.community_label_propagation()
    
    #print '\ Number of clusters for original graph: ', len(clusters)
    #print '\n Modularity value after clustering (original G) is :', G.modularity(clusters)
    
    # Label for plot
    algos_labels.append('LabelPropag')
    algos_line_style.append('>')
    


if ifSpectralClustering:

    ''' Clustering the original graph'''
    # If affinity is the adjacency matrix of a graph, this method can be used to find normalized graph cuts.
    #A = np.array(list(G.get_adjacency()))
    #A = graph_to_sparse_matrix(G)
    A = nx.adjacency_matrix(G_networkX)
    #adj_matrix = np.array(A.todense())
    # Different label assignment strategies can be used, corresponding to the assign_labels parameter of SpectralClustering. 
    # The "kmeans" strategy can match finer details of the data, but it can be more unstable. In particular, unless you control 
    # the random_state, it may not be reproducible from run-to-run, as it depends on a random initialization. On the other hand, 
    # the "discretize" strategy is 100% reproducible, but it tends to create parcels of fairly even and geometrical shape.
    
    nb_clust_spectral = len(clusters_Louvain)
    
    clusters_Spectral = spectral_clustering(affinity = A, n_clusters = nb_clust_spectral , assign_labels="kmeans", random_state = 1)
    
    # Label for plot
    algos_labels.append('Spectral')
    algos_line_style.append('x')
            
            
if ifMETIS:
    ''' Clustering the original graph'''
    
    nb_clust_Metis = len(clusters_Louvain)
    
    clusters_METIS = METIS_Clustering(source_path, convert_path_metis , network_name , npart=nb_clust_Metis)
    
    # Label for plot
    algos_labels.append('Metis')
    algos_line_style.append('D')

        
''' ********  P ===> Preparing perturbed graphs ******** '''
print '\n ******** Perturbed graphs ********' 
    
for i in range(nb_run):
    print "========= Simulation No " + str(i+1) + " ============="
    
    ''' NCP plots parameters '''

    NCP_values_min_Louvain , NCP_values_min_FastGreedyMM , NCP_values_min_Infomap , NCP_values_min_LeadingEigen , NCP_values_min_Walktrap , NCP_values_min_LabelPropag , NCP_values_min_Spectral , NCP_values_min_Metis = [] , [] , [] , [] , [] , [] , [] , []
    NCP_k_values_Louvain , NCP_k_values_FastGreedyMM , NCP_k_values_Infomap , NCP_k_values_LeadingEigen , NCP_k_values_Walktrap , NCP_k_values_LabelPropag , NCP_k_values_Spectral , NCP_k_values_Metis = [] , [] , [] , [] , [] , [] , [] , []
    NCP_k_distribution_Louvain , NCP_k_distribution_FastGreedyMM , NCP_k_distribution_Infomap , NCP_k_distribution_LeadingEigen , NCP_k_distribution_Walktrap , NCP_k_distribution_LabelPropag , NCP_k_distribution_Spectral , NCP_k_distribution_Metis = [] , [] , [] , [] , [] , [] , [] , []

    
    for j,eps in enumerate(epsilon_values):
        
        G_P_networkX = convert_to_igraph_format(source_path, convert_path_P, algo_name='iGraph_algos' , net_name=network_name , perturb_model=noise_model , epsilon = eps , addORdel=add_or_delete , link_pred_file=None)
        
        G_P = Graph.Read_Edgelist(f=convert_path_P, directed=False)
        
        # We set vertices name=index to be able to retrieve them when we extract sub-graphs associated with each cluster
        # Because nodese ids are relabeled starting from 0 when subgraph extracted
        for idx, v in enumerate(VertexSeq(G_P)):
            v['name'] = str(idx) # now each vertex has an attribute 'name'
    
        print '\n Summary perturbed graph:'
        summary(G_P)
        
        # Save the % of added/deleted edges
        n_added_deleted_edges[i,j] = (np.absolute((G_P.ecount() - G.ecount()))/float(G.ecount()))*100
        
        # Eigendecomposition of the graph Laplacian
        L_spectrum = normalized_laplacian_spectrum(G_P_networkX)
        # Spectral lower bound value (see Leskovec 2008)
        Fiedler_value_per_noise[i,j] = L_spectrum[1]/float(2)
        
        # Initialize k, the depth of the 3D matrix of metrics values
        k=0
        
        ''' ************************************************* DO CLUSTERING ****************************************'''
        
        
        if ifBlondelMultilevel:
            '''============================================================================
            #    BLONDEL MULTILEVEL
            #=============================================================================='''
            
            '''Community structure based on the multilevel algorithm of Blondel et al.
            
            This is a bottom-up algorithm: initially every vertex belongs to a separate community, and vertices are moved between communities 
            iteratively in a way that maximizes the vertices' local contribution to the overall modularity score. When a consensus is reached 
            (i.e. no single move would increase the modularity score), every community in the original graph is shrank to a single vertex 
            (while keeping the total weight of the adjacent edges) and the process continues on the next level. The algorithm stops when it is not 
            possible to increase the modularity any more after shrinking the communities to vertices.
            
            This algorithm is said to run almost in linear time on sparse graphs.
            
            Ref :: VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of community hierarchies in large networks, J Stat Mech P10008 (2008), '''
            
            
            ''' Clustering the perturbed graph'''
            # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
            clusters_P_Louvain = G_P.community_multilevel()
            
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_Louvain , scoring_function , algos='iGraph')
            
        
            ''' ********  Evaluating clusters similarity ******** '''
            
            print '\n ******** Results ********'+' Simu. no. = '+str(i+1)+' / '+'Noise level = '+str(j+1)
            
            nmi_igraph = compare_communities(clusters_Louvain, clusters_P_Louvain, method='nmi', remove_none=False)
            print '\n Normalized Mutual Information Louvain algo is :', nmi_igraph
            vi_igraph = compare_communities(clusters_Louvain, clusters_P_Louvain, method='vi', remove_none=False)
            ari_igraph = compare_communities(clusters_Louvain, clusters_P_Louvain, method='adjusted_rand', remove_none=False)
            
            # Saving the results
            nmi_values[k,i,j] = nmi_igraph
            vi_values[k,i,j] = vi_igraph
            ari_values[k,i,j] = ari_igraph
            modularity_values[k,i,j] = G_P.modularity(clusters_P_Louvain)
            nb_clusters_values[k,i,j] = len(clusters_P_Louvain)
            
            k=k+1
            
            NCP_values_min_Louvain.append(NCP_values_min)
            NCP_k_values_Louvain.append(k_values)
            NCP_k_distribution_Louvain.append(k_distribution)
            
            
            
        if ifFastGreedyMaxModularity:
            '''============================================================================
            #    FAST GREEDY MODULARITY MAXIMIZATION
            #=============================================================================='''
            
            """ Community structure based on the greedy optimization of modularity.
            
            This algorithm merges individual nodes into communities in a way that greedily maximizes the modularity score of the graph. 
            It can be proven that if no merge can increase the current modularity score, the algorithm can be stopped since no further increase can be achieved. 
            
            Ref :: A Clauset, MEJ Newman and C Moore: Finding community structure in very large networks. Phys Rev E 70, 066111 (2004). """
            
            """
            FROM --> http://www.cs.unm.edu/~aaron/research/fastmodularity.htm
            
            Input file requirements
            The program was written to handle networks in the form of a flat text file containing edge adjacencies. I call this file format a .pairs file. 
            Specifically, the network topology must conform to the following requirements:
            • .pairs is a list of tab-delimited pairs of numeric indices, e.g., "54\t91\n"
            • the network described is a SINGLE COMPONENT
            • there are NO SELF-LOOPS or MULTI-EDGES in the file
            • the MINIMUM NODE ID = 0 in the input file
            • the MAXIMUM NODE ID can be anything, but the program will use less memory if nodes are labeled sequentially
            
            This file format may seem a bit peculiar, but it was sufficient for the demonstration that the algorithm performs as promised. 
            You are free to alter the file import function readInputFile() to fit your needs.
            
            An example input file, for Zachary's karate club network is here.
            
            Common error messages
            The most common error message returned by the program is of the form "WARNING: invalid join (X Y) found at top of heap". 
            If you receive this error message, it is almost surely because your input .pairs file does not meet the above requirements. 
            In particular, it likely contains self-loops, multi-edges or is not a single connected component. 
            If you receive this message, try fixing these issues with your network and trying again.
            """
            
            
            ''' Clustering the perturbed graph'''
            # Returns: an appropriate VertexDendrogram object.
            communities_P_FastGreedyMM = G_P.community_fastgreedy()
            
            """ Cuts the dendrogram at the given level and returns a corresponding VertexClustering object.
            
            as_clustering()
            
            Parameters:
            n - the desired number of clusters. Merges are replayed from the beginning until the membership vector has exactly n distinct elements 
            or until there are no more recorded merges, whichever happens first. If None, the optimal count hint given by the clustering algorithm will 
            be used If the optimal count was not given either, it will be calculated by selecting the level where the modularity is maximal. """
            
            # clusters is a VertexClustering object.
            clusters_P_FastGreedyMM = communities_P_FastGreedyMM.as_clustering()

            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_FastGreedyMM , scoring_function , algos='iGraph')
        
        
            ''' ********  Evaluating clusters similarity ******** '''
            
            nmi_igraph = compare_communities(clusters_FastGreedyMM, clusters_P_FastGreedyMM, method='nmi', remove_none=False)
            print '\n Normalized Mutual Information Fast Greedy Modularity Max is :', nmi_igraph
            vi_igraph = compare_communities(clusters_FastGreedyMM, clusters_P_FastGreedyMM, method='vi', remove_none=False)
            ari_igraph = compare_communities(clusters_FastGreedyMM, clusters_P_FastGreedyMM, method='adjusted_rand', remove_none=False)
            
            # Saving the results
            nmi_values[k,i,j] = nmi_igraph
            vi_values[k,i,j] = vi_igraph
            ari_values[k,i,j] = ari_igraph
            modularity_values[k,i,j] = G_P.modularity(clusters_P_FastGreedyMM)
            nb_clusters_values[k,i,j] = len(clusters_P_FastGreedyMM)
            
            k=k+1

            NCP_values_min_FastGreedyMM.append(NCP_values_min)
            NCP_k_values_FastGreedyMM.append(k_values)
            NCP_k_distribution_FastGreedyMM.append(k_distribution)
            
            del clusters_P_FastGreedyMM
            


        if ifINFOMAP:
            '''============================================================================
            #    INFOMAP
            #=============================================================================='''
            
            """ Finds the community structure of the network according to the Infomap method of Martin Rosvall and Carl T. Bergstrom.
            
            Ref :: M. Rosvall and C. T. Bergstrom: Maps of information flow reveal community structure in complex networks, PNAS 105, 1118 (2008)  
            
            http://www.tp.umu.se/~rosvall/code.html    """


            ''' Clustering the perturbed graph'''
            # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
            robust_nmi = []
            robust_vi = []
            robust_ari = []
            robust_modularity = []
            robust_nb_clust = []
            
            for rob in range(10):
                
                clusters_P_Infomap = G_P.community_infomap()
                
                nmi_igraph = compare_communities(clusters_Infomap, clusters_P_Infomap, method='nmi', remove_none=False)
                vi_igraph = compare_communities(clusters_Infomap, clusters_P_Infomap, method='vi', remove_none=False)
                ari_igraph = compare_communities(clusters_Infomap, clusters_P_Infomap, method='adjusted_rand', remove_none=False)
                
                robust_nmi.append(nmi_igraph)
                robust_vi.append(vi_igraph)
                robust_ari.append(ari_igraph)
                robust_modularity.append(G_P.modularity(clusters_P_Infomap))
                robust_nb_clust.append(len(clusters_P_Infomap))
            
            #print '\ Number of clusters for perturbed graph: ', len(clusters_P)
            #print '\n Modularity value after clustering (perturbed G) is :', G_P.modularity(clusters_P)
            
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_Infomap , scoring_function , algos='iGraph')
            
            
            ''' ********  Evaluating clusters similarity ******** '''
            
            print '\n Normalized Mutual Information INFOMAP is :', nmi_igraph
            
            # Saving the results
            nmi_values[k,i,j] = np.mean(robust_nmi)
            vi_values[k,i,j] = np.mean(robust_vi)
            ari_values[k,i,j] = np.mean(robust_ari)
            modularity_values[k,i,j] = np.mean(robust_modularity)
            nb_clusters_values[k,i,j] = round(np.mean(robust_nb_clust),2)
            
            k=k+1 

            NCP_values_min_Infomap.append(NCP_values_min)
            NCP_k_values_Infomap.append(k_values)
            NCP_k_distribution_Infomap.append(k_distribution)
            
            del clusters_P_Infomap
            


        if ifLeadingEigenvector:
            '''============================================================================
            #    LEADING EIGENVECTOR
            #=============================================================================='''
            
            """ Newman's leading eigenvector method for detecting community structure. This is the proper implementation of the recursive, divisive algorithm: 
            each split is done by maximizing the modularity regarding the original network.
            
            Ref :: MEJ Newman: Finding community structure in networks using the eigenvectors of matrices Phys. Rev. E 74, 036104 (2006)  """


            ''' Clustering the perturbed graph'''
            # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
            clusters_P_LeadingEigen = G_P.community_leading_eigenvector(arpack_options=arpack_options)
            
            #print '\ Number of clusters for perturbed graph: ', len(clusters_P)
            #print '\n Modularity value after clustering (perturbed G) is :', G_P.modularity(clusters_P)
            
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_LeadingEigen , scoring_function , algos='iGraph')
            
            
            ''' ********  Evaluating clusters similarity ******** '''
            
            nmi_igraph = compare_communities(clusters_LeadingEigen, clusters_P_LeadingEigen, method='nmi', remove_none=False)
            print '\n Normalized Mutual Information Leading Eigen V is :', nmi_igraph
            vi_igraph = compare_communities(clusters_LeadingEigen, clusters_P_LeadingEigen, method='vi', remove_none=False)
            ari_igraph = compare_communities(clusters_LeadingEigen, clusters_P_LeadingEigen, method='adjusted_rand', remove_none=False)
            
            # Saving the results
            nmi_values[k,i,j] = nmi_igraph
            vi_values[k,i,j] = vi_igraph
            ari_values[k,i,j] = ari_igraph
            modularity_values[k,i,j] = G_P.modularity(clusters_P_LeadingEigen)
            nb_clusters_values[k,i,j] = len(clusters_P_LeadingEigen)
            
            k=k+1

            NCP_values_min_LeadingEigen.append(NCP_values_min)
            NCP_k_values_LeadingEigen.append(k_values)
            NCP_k_distribution_LeadingEigen.append(k_distribution)
            
            del clusters_P_LeadingEigen



        if ifWalktrap:
            '''============================================================================
            #    WALKTRAP
            #=============================================================================='''
        
            '''Community detection algorithm of Latapy & Pons, based on random walks.
            
            The basic idea of the algorithm is that short random walks tend to stay in the same community. The result of the clustering will be represented as a dendrogram.
            
            Ref :: Pascal Pons, Matthieu Latapy: Computing communities in large networks using random walks, http://arxiv.org/abs/physics/0512106. 
            
            https://www-complexnetworks.lip6.fr/~latapy/PP/walktrap.html   ''' 
        
            
            ''' Clustering the perturbed graph'''
            # Returns: an appropriate VertexDendrogram object.
            communities_P_Walktrap = G_P.community_walktrap()
            
            """ Cuts the dendrogram at the given level and returns a corresponding VertexClustering object.
            
            as_clustering()
            
            Parameters:
            n - the desired number of clusters. Merges are replayed from the beginning until the membership vector has exactly n distinct elements 
            or until there are no more recorded merges, whichever happens first. If None, the optimal count hint given by the clustering algorithm will 
            be used If the optimal count was not given either, it will be calculated by selecting the level where the modularity is maximal. """
            
            # clusters is a VertexClustering object.
            clusters_P_Walktrap = communities_P_Walktrap.as_clustering()
            
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_Walktrap , scoring_function , algos='iGraph')
            
            
            ''' ********  Evaluating clusters similarity ******** '''
            
            nmi_igraph = compare_communities(clusters_Walktrap, clusters_P_Walktrap, method='nmi', remove_none=False)
            print '\n Normalized Mutual Information Walktrap is :', nmi_igraph
            vi_igraph = compare_communities(clusters_Walktrap, clusters_P_Walktrap, method='vi', remove_none=False)
            ari_igraph = compare_communities(clusters_Walktrap, clusters_P_Walktrap, method='adjusted_rand', remove_none=False)
            
            # Saving the results
            nmi_values[k,i,j] = nmi_igraph
            vi_values[k,i,j] = vi_igraph
            ari_values[k,i,j] = ari_igraph
            modularity_values[k,i,j] = G_P.modularity(clusters_P_Walktrap)
            nb_clusters_values[k,i,j] = len(clusters_P_Walktrap)
            
            k=k+1

            NCP_values_min_Walktrap.append(NCP_values_min)
            NCP_k_values_Walktrap.append(k_values)
            NCP_k_distribution_Walktrap.append(k_distribution)
            
            clusters_P_Walktrap



        if ifLabelPropagation:
            '''============================================================================
            #    LABEL PROPAGATION
            #=============================================================================='''
            ''' Finds the community structure of the graph according to the label propagation method of Raghavan et al. Initially, each vertex is assigned a different label. 
            After that, each vertex chooses the dominant label in its neighbourhood in each iteration. Ties are broken randomly and the order in which the vertices are updated
            is randomized before every iteration. The algorithm ends when vertices reach a consensus. Note that since ties are broken randomly, there is no guarantee that the 
            algorithm returns the same community structure after each run. In fact, they frequently differ. See the paper of Raghavan et al on how to come up with an aggregated 
            community structure.
            
            Ref  :: Raghavan, U.N. and Albert, R. and Kumara, S. Near linear time algorithm to detect community structures in large-scale networks. 
            Phys Rev E 76:036106, 2007. http://arxiv.org/abs/0709.2938. '''


            ''' Clustering the perturbed graph'''
            # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
            robust_nmi = []
            robust_vi = []
            robust_ari = []
            robust_modularity = []
            robust_nb_clust = []
            
            for rob in range(10):
                clusters_P_LabelPropag = G_P.community_label_propagation()
                
                nmi_igraph = compare_communities(clusters_LabelPropag, clusters_P_LabelPropag, method='nmi', remove_none=False)
                vi_igraph = compare_communities(clusters_LabelPropag, clusters_P_LabelPropag, method='vi', remove_none=False)
                ari_igraph = compare_communities(clusters_LabelPropag, clusters_P_LabelPropag, method='adjusted_rand', remove_none=False)
                
                robust_nmi.append(nmi_igraph)
                robust_vi.append(vi_igraph)
                robust_ari.append(ari_igraph)
                robust_modularity.append(G_P.modularity(clusters_P_LabelPropag))
                robust_nb_clust.append(len(clusters_P_LabelPropag))
            
            #print '\ Number of clusters for perturbed graph: ', len(clusters_P)
            #print '\n Modularity value after clustering (perturbed G) is :', G_P.modularity(clusters_P)

            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_LabelPropag , scoring_function , algos='iGraph')
            
        
            ''' ********  Evaluating clusters similarity ******** '''
        
            print '\n Normalized Mutual Information Label Propag is :', nmi_igraph
            
            
            # Saving the results
            nmi_values[k,i,j] = np.mean(robust_nmi)
            vi_values[k,i,j] = np.mean(robust_vi)
            ari_values[k,i,j] = np.mean(robust_ari)
            modularity_values[k,i,j] = np.mean(robust_modularity)
            nb_clusters_values[k,i,j] = round(np.mean(robust_nb_clust),2)
            
            k=k+1
            
            NCP_values_min_LabelPropag.append(NCP_values_min)
            NCP_k_values_LabelPropag.append(k_values)
            NCP_k_distribution_LabelPropag.append(k_distribution)
            
            del clusters_P_LabelPropag
            


        if ifSpectralClustering:
            '''============================================================================
            #    SPECTRAL CLUSTERING
            #=============================================================================='''
            
            '''SpectralClustering does a low-dimension embedding of the affinity matrix between samples, followed by a KMeans 
            in the low dimensional space. It is especially efficient if the affinity matrix is sparse and the pyamg module is installed. 
            SpectralClustering requires the number of clusters to be specified. It works well for a small number of clusters but is not advised when using many clusters.
            For two clusters, it solves a convex relaxation of the normalised cuts problem on the similarity graph: cutting the graph in two so that the 
            weight of the edges cut is small compared to the weights of the edges inside each cluster. This criteria is especially interesting when working 
            on images: graph vertices are pixels, and edges of the similarity graph are a function of the gradient of the image
            
            Ref :: Normalized cuts and image segmentation, 2000 Jianbo Shi, Jitendra Malik '''
            

            ''' Clustering the perturbed graph'''
            # If affinity is the adjacency matrix of a graph, this method can be used to find normalized graph cuts.
            #A_P = np.array(list(G_P.get_adjacency()))
            #A_P = graph_to_sparse_matrix(G_P)
            A_P = nx.adjacency_matrix(G_P_networkX)
            #adj_matrix_P = np.array(A_P.todense())
            # Different label assignment strategies can be used, corresponding to the assign_labels parameter of SpectralClustering. 
            # The "kmeans" strategy can match finer details of the data, but it can be more unstable. In particular, unless you control 
            # the random_state, it may not be reproducible from run-to-run, as it depends on a random initialization. On the other hand, 
            # the "discretize" strategy is 100% reproducible, but it tends to create parcels of fairly even and geometrical shape.
            
            nb_clust_spectral = len(clusters_P_Louvain)
            
            clusters_P_Spectral = spectral_clustering(affinity = A_P, n_clusters = nb_clust_spectral , assign_labels="kmeans", random_state = 1)
            
            # Transform assignment in a dict format 
            clusters_dict_P_Spectral = {}

            for (node_id , clust_id) in enumerate(clusters_P_Spectral): 
            # each line in the list correspond to the node id and the value to the cluster id the node belongs to
                clusters_dict_P_Spectral[clust_id] = clusters_dict_P_Spectral.get(clust_id, [])
                clusters_dict_P_Spectral[clust_id].append(str(node_id))
    
    
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P_networkX, clusters_dict_P_Spectral , scoring_function , algos='non_iGraph')
            
            
            ''' ********  Evaluating clusters similarity ******** '''
            
            nmi_scikit = normalized_mutual_info_score(labels_true = clusters_Spectral, labels_pred = clusters_P_Spectral)
            print '\n Normalized Mutual Information Spectral clustering algo is :', nmi_scikit
            vi_scikit = variation_of_information_score(labels_true = clusters_Spectral, labels_pred = clusters_P_Spectral)
            ari_scikit = adjusted_rand_score(labels_true = clusters_Spectral, labels_pred = clusters_P_Spectral)
            
            # Saving the results
            nmi_values[k,i,j] = nmi_scikit
            vi_values[k,i,j] = vi_scikit
            ari_values[k,i,j] = ari_scikit
            
            # list of sets (Non-overlaping sets of nodes)
            communities = []
            for key, value in clusters_dict_P_Spectral.iteritems():
                communities.append(set(value))
            
            modularity_values[k,i,j] = modularity(G_P_networkX, communities)
            
            nb_clusters_values[k,i,j] = nb_clust_spectral
            
            k=k+1
            
            NCP_values_min_Spectral.append(NCP_values_min)
            NCP_k_values_Spectral.append(k_values)
            NCP_k_distribution_Spectral.append(k_distribution)
            
            del clusters_P_Spectral
            

        if ifMETIS:
            ''' Clustering the perturbed graph ''' 
            
            nb_clust_Metis = len(clusters_P_Louvain)
            
            clusters_P_METIS = METIS_Clustering(source_path, convert_path_metis_P , network_name , npart=nb_clust_Metis , perturb_model=noise_model , epsilon = eps , addORdel=add_or_delete , link_pred_file=None)


            # Transform assignment in a dict format 
            clusters_dict_P_METIS = {}

            for (node_id , clust_id) in enumerate(clusters_P_METIS): 
            # each line in the list correspond to the node id and the value to the cluster id the node belongs to
                clusters_dict_P_METIS[int(clust_id)] = clusters_dict_P_METIS.get(int(clust_id), [])
                ''' ATTENTION : since we useG_P_networkX graph, first node id = 0 !!! '''
                clusters_dict_P_METIS[int(clust_id)].append(str(node_id))
    
    
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P_networkX, clusters_dict_P_METIS , scoring_function , algos='non_iGraph')
            
            
            ''' ********  Evaluating clusters similarity ******** '''
            
            nmi_scikit = normalized_mutual_info_score(labels_true = clusters_METIS, labels_pred = clusters_P_METIS )
            print '\n Normalized Mutual Information METIS algo is :', nmi_scikit
            vi_scikit = variation_of_information_score(labels_true = clusters_METIS, labels_pred = clusters_P_METIS )
            ari_scikit = adjusted_rand_score(labels_true = clusters_METIS, labels_pred = clusters_P_METIS )
            
            # Saving the results
            nmi_values[k,i,j] = nmi_scikit
            vi_values[k,i,j] = vi_scikit
            ari_values[k,i,j] = ari_scikit

            communities = []
            for key, value in clusters_dict_P_METIS.iteritems():
                communities.append(set(value))

            modularity_values[k,i,j] = modularity(G_P_networkX, communities)  # list of sets (Non-overlaping sets of nodes)
            
            nb_clusters_values[k,i,j] = nb_clust_Metis
            
            
            k=k+1
            
            NCP_values_min_Metis.append(NCP_values_min)
            NCP_k_values_Metis.append(k_values)
            NCP_k_distribution_Metis.append(k_distribution)
            
            del clusters_P_METIS


    # Save the NCP plot values for several runs
    save__NCP_values_min_Louvain[i] , save__NCP_values_min_FastGreedyMM[i] , save__NCP_values_min_Infomap[i] , save__NCP_values_min_LeadingEigen[i] , save__NCP_values_min_Walktrap[i] , save__NCP_values_min_LabelPropag[i] , save__NCP_values_min_Spectral[i] , save__NCP_values_min_Metis[i] = NCP_values_min_Louvain , NCP_values_min_FastGreedyMM , NCP_values_min_Infomap , NCP_values_min_LeadingEigen , NCP_values_min_Walktrap , NCP_values_min_LabelPropag , NCP_values_min_Spectral , NCP_values_min_Metis        
    save__NCP_k_values_Louvain[i] , save__NCP_k_values_FastGreedyMM[i] , save__NCP_k_values_Infomap[i] , save__NCP_k_values_LeadingEigen[i] , save__NCP_k_values_Walktrap[i] , save__NCP_k_values_LabelPropag[i] , save__NCP_k_values_Spectral[i] , save__NCP_k_values_Metis[i] = NCP_k_values_Louvain , NCP_k_values_FastGreedyMM , NCP_k_values_Infomap , NCP_k_values_LeadingEigen , NCP_k_values_Walktrap , NCP_k_values_LabelPropag , NCP_k_values_Spectral , NCP_k_values_Metis
    save__NCP_k_distribution_Louvain[i] , save__NCP_k_distribution_FastGreedyMM[i] , save__NCP_k_distribution_Infomap[i] , save__NCP_k_distribution_LeadingEigen[i] , save__NCP_k_distribution_Walktrap[i] , save__NCP_k_distribution_LabelPropag[i] , save__NCP_k_distribution_Spectral[i] , save__NCP_k_distribution_Metis[i] = NCP_k_distribution_Louvain , NCP_k_distribution_FastGreedyMM , NCP_k_distribution_Infomap , NCP_k_distribution_LeadingEigen , NCP_k_distribution_Walktrap , NCP_k_distribution_LabelPropag , NCP_k_distribution_Spectral , NCP_k_distribution_Metis
    
'''============================================================================
#                          PLOTS and SAVE
#==========================================================================='''

#mpl.rcParams['font.family'] = 'serif'

''' PLOT NMI '''
fig1 = plt.figure()

for z in range(len(nmi_values)): # len(3D) gives the depth so per algo values
    if (noise_model != 'CONFIG'):
        plt.errorbar(np.mean(n_added_deleted_edges, axis=0), np.mean(nmi_values[z,:,:], axis=0), xerr=None, yerr=np.std(nmi_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        plt.errorbar(epsilon_values, np.mean(nmi_values[z,:,:], axis=0), xerr=None, yerr=np.std(nmi_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(noise_model+' + '+add_or_delete, fontsize=10)
    else:
        plt.title('ConfMP', fontsize=10)
    plt.ylabel('Normalized Mutual Information', fontsize=14)
    plt.ylim(0,1)
    plt.xlim(-2,30)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.draw()
    
if ifSave:
    fig1.savefig(path_results+'ALLalgos_NMI_'+network_name+'_'+noise_model+'_'+add_or_delete+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
    
    # Save average n_added_deleted_edges
    np.save(path_results+'AVG_n_added_deleted_edges_'+network_name+'_'+noise_model+'_'+add_or_delete , np.mean(n_added_deleted_edges, axis=0))
    # Save NMI values
    np.save(path_results+'NMIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete , nmi_values)
    
    ## LOAD 
    ## Save average n_added_deleted_edges
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/AVG_n_added_deleted_edges_'+network_name+'_'+noise_model+'_'+add_or_delete)
    #
    ## Save NMI values
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/NMIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete)


''' PLOT VI '''

fig2 = plt.figure()

for z in range(len(vi_values)): # len(3D) gives the depth so per algo values
    if (noise_model != 'CONFIG'):
        plt.errorbar(np.mean(n_added_deleted_edges, axis=0), np.mean(vi_values[z,:,:], axis=0), xerr=None, yerr=np.std(vi_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        plt.errorbar(epsilon_values, np.mean(vi_values[z,:,:], axis=0), xerr=None, yerr=np.std(vi_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(noise_model+' + '+add_or_delete, fontsize=10)
    else:
        plt.title('ConfMP', fontsize=10)
    plt.ylabel('Variation of Information', fontsize=14)
    plt.xlim(-2,30)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.draw()
    
if ifSave:
    fig2.savefig(path_results+'ALLalgos_VI'+network_name+'_'+noise_model+'_'+add_or_delete+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
    
    # Save VI values
    np.save(path_results+'VIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete ,vi_values)
    
    ## LOAD 
    ## Save average n_added_deleted_edges
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/AVG_n_added_deleted_edges_'+network_name+'_'+noise_model+'_'+add_or_delete)
    #
    ## Save NMI values
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/NMIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete)

''' PLOT ARI '''
fig1 = plt.figure()

for z in range(len(ari_values)): # len(3D) gives the depth so per algo values
    if (noise_model != 'CONFIG'):
        plt.errorbar(np.mean(n_added_deleted_edges, axis=0), np.mean(ari_values[z,:,:], axis=0), xerr=None, yerr=np.std(ari_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        plt.errorbar(epsilon_values, np.mean(ari_values[z,:,:], axis=0), xerr=None, yerr=np.std(ari_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(noise_model+' + '+add_or_delete, fontsize=10)
    else:
        plt.title('ConfMP', fontsize=10)
    plt.ylabel('Adjusted Rand Index', fontsize=14)
    plt.ylim(0,1)
    plt.xlim(-2,30)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.draw()
    
if ifSave:
    fig1.savefig(path_results+'ALLalgos_ARI_'+network_name+'_'+noise_model+'_'+add_or_delete+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
        # Save NMI values
    np.save(path_results+'ARIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete , ari_values)
    
    

''' PLOT Modularity '''

fig3 = plt.figure()

for z in range(len(modularity_values)): # len(3D) gives the depth so per algo values
    if (noise_model != 'CONFIG'):
        plt.errorbar(np.mean(n_added_deleted_edges, axis=0), np.mean(modularity_values[z,:,:], axis=0), xerr=None, yerr=np.std(modularity_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        plt.errorbar(epsilon_values, np.mean(modularity_values[z,:,:], axis=0), xerr=None, yerr=np.std(modularity_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(noise_model+' + '+add_or_delete, fontsize=10)
    else:
        plt.title('ConfMP', fontsize=10)
    plt.ylabel('Modularity', fontsize=14)
    plt.xlim(-2,30)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.draw()
    
if ifSave:
    fig3.savefig(path_results+'ALLalgos_Modularity'+network_name+'_'+noise_model+'_'+add_or_delete+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
    
    # Save VI values
    np.save(path_results+'Modularityvalues_'+network_name+'_'+noise_model+'_'+add_or_delete ,modularity_values)
    
    ## LOAD 
    ## Save average n_added_deleted_edges
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/AVG_n_added_deleted_edges_'+network_name+'_'+noise_model+'_'+add_or_delete)
    #
    ## Save NMI values
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/NMIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete)



''' PLOT Nb clusters '''

fig4 = plt.figure()

for z in range(len(nb_clusters_values)): # len(3D) gives the depth so per algo values
    if (noise_model != 'CONFIG'):
        plt.errorbar(np.mean(n_added_deleted_edges, axis=0), np.mean(nb_clusters_values[z,:,:], axis=0), xerr=None, yerr=np.std(nb_clusters_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        plt.errorbar(epsilon_values, np.mean(nb_clusters_values[z,:,:], axis=0), xerr=None, yerr=np.std(nb_clusters_values[z,:,:], axis=0), linewidth=1 , marker=algos_line_style[z], label=algos_labels[z])
        plt.xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(noise_model+' + '+add_or_delete, fontsize=10)
    else:
        plt.title('ConfMP', fontsize=10)
    plt.ylabel('Number of clusters', fontsize=14)
    plt.xlim(-2,30)
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.draw()
    
if ifSave:
    fig4.savefig(path_results+'ALLalgos_NbClust'+network_name+'_'+noise_model+'_'+add_or_delete+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
    
    # Save VI values
    np.save(path_results+'nb_clusters_values_'+network_name+'_'+noise_model+'_'+add_or_delete ,nb_clusters_values)
    
    ## LOAD 
    ## Save average n_added_deleted_edges
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/AVG_n_added_deleted_edges_'+network_name+'_'+noise_model+'_'+add_or_delete)
    #
    ## Save NMI values
    #np.load('/home/mmitri/Documents/Stage/Codes/Results/NMIvalues_'+network_name+'_'+noise_model+'_'+add_or_delete)


''' Plot NCPs for Louvain '''
if ifBlondelMultilevel:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_Louvain , save__NCP_k_values_Louvain , save__NCP_k_distribution_Louvain , Fiedler_value_per_noise , algo_name='Louvain')

''' Plot NCPs for FastGreedyMM '''
if ifFastGreedyMaxModularity:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_FastGreedyMM , save__NCP_k_values_FastGreedyMM , save__NCP_k_distribution_FastGreedyMM , Fiedler_value_per_noise , algo_name='FastGreedyMM')

''' Plot NCPs for Infomap '''
if ifINFOMAP:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_Infomap , save__NCP_k_values_Infomap , save__NCP_k_distribution_Infomap , Fiedler_value_per_noise , algo_name='Infomap')

''' Plot NCPs for LeadingEigen '''
if ifLeadingEigenvector:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_LeadingEigen , save__NCP_k_values_LeadingEigen , save__NCP_k_distribution_LeadingEigen , Fiedler_value_per_noise , algo_name='LeadingEigen')

''' Plot NCPs for Walktrap '''
if ifWalktrap:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_Walktrap , save__NCP_k_values_Walktrap , save__NCP_k_distribution_Walktrap , Fiedler_value_per_noise , algo_name='Walktrap')

''' Plot NCPs for LabelPropag '''
if ifLabelPropagation:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_LabelPropag , save__NCP_k_values_LabelPropag , save__NCP_k_distribution_LabelPropag , Fiedler_value_per_noise , algo_name='LabelPropag')                       


''' Plot NCPs for Spectral Clustering '''
if ifSpectralClustering:
    
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_Spectral , save__NCP_k_values_Spectral , save__NCP_k_distribution_Spectral , Fiedler_value_per_noise , algo_name='Spectral clustering')                       

''' Plot NCPs for Metis '''
if ifMETIS:
    Plot_and_save_NCP_charts(path_results, network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , 
                         save__NCP_values_min_Metis , save__NCP_k_values_Metis , save__NCP_k_distribution_Metis , Fiedler_value_per_noise , algo_name='Metis') 
                         
    
