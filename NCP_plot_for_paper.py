# -*- coding: utf-8 -*-
"""
@author: mmitri
"""

from igraph import *
import numpy as np
import networkx as nx
from NetworksFromSNAP import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import spectral_clustering
from sklearn.metrics.cluster import normalized_mutual_info_score
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

ifBlondelMultilevel = False

ifFastGreedyMaxModularity = False

ifINFOMAP = True

ifLeadingEigenvector = False
arpack_options.maxiter=10000000

ifWalktrap = False

ifLabelPropagation = False

ifSpectralClustering = False
nb_clust_spectral = 40

ifMETIS = False
nb_clust_Metis = 40


ifSpinglass = False
ifMCL = False


algos_labels = []
algos_line_style = []
nb_active_algos = 1

''' Chose the network '''
network_name = ['Email-Enron' , 'GRQCcollab' , 'AstroPhysicsCollab' , 'Wiki-Vote' , 'AS-Caida', 'CA-HepTh']
network_name = network_name[4]

''' Chose the number of run '''
nb_run = 1


''' Chose the noise model '''
noise_model = ['ERP' , 'CLP' , 'LPP' , 'CONFIG']
noise_model = noise_model[0]

add_or_delete = 'Add' # Add or 'Delete'

if (noise_model != 'CONFIG'):
    if add_or_delete=='Delete':
        epsilon_valuesI = [ 0 , 0.01 , 0.2 , 0.4 , 0.6 , 0.8 , 1 , 1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5 , 6]
        epsilon_values = [i * 225.9 for i in epsilon_valuesI]
    elif add_or_delete=='Add':
        epsilon_valuesI = [ 0 ]
        epsilon_values = [i * 0.323 for i in epsilon_valuesI]
else:
    # Alpha values between 0 and 1 
    epsilon_values = np.linspace(0, 0.3, num=16)


''' Evaluating the results '''
# Create empty 3D arrays fot saving the value of metrics quantifying clusters differences :
# arg1 = number of pages (i.e. depth, third dimension), i.e. number of algos, une page for each algo
# arg2 = rows, i.e. number of simulations
# arg3 = columns, i.e. number of values of epsilon
nmi_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
vi_values = np.empty((nb_active_algos , nb_run , len(epsilon_values)))
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

''' ********  P ===> Preparing perturbed graphs ******** '''
print '\n ******** Perturbed graphs ********' 
    
for i in range(nb_run):
    print "========= Simulation No " + str(i+1) + " ============="
    
    ''' NCP plots parameters '''

    NCP_values_min_Louvain , NCP_values_min_FastGreedyMM , NCP_values_min_Infomap , NCP_values_min_LeadingEigen , NCP_values_min_Walktrap , NCP_values_min_LabelPropag , NCP_values_min_Spectral , NCP_values_min_Metis = [] , [] , [] , [] , [] , [] , [] , []
    NCP_k_values_Louvain , NCP_k_values_FastGreedyMM , NCP_k_values_Infomap , NCP_k_values_LeadingEigen , NCP_k_values_Walktrap , NCP_k_values_LabelPropag , NCP_k_values_Spectral , NCP_k_values_Metis = [] , [] , [] , [] , [] , [] , [] , []
    NCP_k_distribution_Louvain , NCP_k_distribution_FastGreedyMM , NCP_k_distribution_Infomap , NCP_k_distribution_LeadingEigen , NCP_k_distribution_Walktrap , NCP_k_distribution_LabelPropag , NCP_k_distribution_Spectral , NCP_k_distribution_Metis = [] , [] , [] , [] , [] , [] , [] , []

    
    for j,eps in enumerate(epsilon_values):
    
        convert_path_P = '/home/mmitri/Documents/Stage/Data/iGraphFormatPerturbed/' + network_name + '_iGraphFormat_Perturbed.txt'
        
        G_P_networkX = convert_to_igraph_format(convert_path_P, algo_name='iGraph_algos' , net_name=network_name , perturb_model=noise_model , epsilon = eps , addORdel=add_or_delete , link_pred_file=None)
        
        G_P = Graph.Read_Edgelist(f=convert_path_P, directed=False)
        
        # We set vertices name=index to be able to retrieve them when we extract sub-graphs associated with each cluster
        # Because nodese ids are relabeled starting from 0 when subgraph extracted
        for idx, v in enumerate(VertexSeq(G_P)):
            v['name'] = str(idx) # now each vertex has an attribute 'name'
    
        print '\n Summary perturbed graph:'
        summary(G_P)


        
        # Initialize k, the depth of the 3D matrix of metrics values
        k=0
        
        ''' ************************************************* DO CLUSTERING ****************************************'''
        
        
     
        if ifINFOMAP:
            '''============================================================================
            #    INFOMAP
            #=============================================================================='''
            
            """ Finds the community structure of the network according to the Infomap method of Martin Rosvall and Carl T. Bergstrom.
            
            Ref :: M. Rosvall and C. T. Bergstrom: Maps of information flow reveal community structure in complex networks, PNAS 105, 1118 (2008)  
            
            http://www.tp.umu.se/~rosvall/code.html    """


            ''' Clustering the perturbed graph'''
            # a list of VertexClustering objects, one corresponding to each level (if return_levels is True), or a VertexClustering corresponding to the best modularity.
            clusters_P_Infomap = G_P.community_infomap()
            
            #print '\ Number of clusters for perturbed graph: ', len(clusters_P)
            #print '\n Modularity value after clustering (perturbed G) is :', G_P.modularity(clusters_P)
            
            ''' NCP plots '''
            k_values , NCP_values_min , k_distribution = NCP_plot(G_P, clusters_P_Infomap , scoring_function , algos='iGraph')
            
            #mpl.rcParams['font.family'] = 'serif'
            
            fig1 = plt.figure()
            #plt.rc('font', family='Helvetica')

            plt.loglog(k_values , NCP_values_min, linewidth=1, marker='o', markersize=4)
            plt.title('NCP plot', fontsize=10)#(r'\textbf{NCP plot}')
            plt.xlabel('k (number of nodes in the cluster)', fontsize=14)#(r'\textbf{k (number of nodes in the cluster)}')
            plt.xlim(1,300)
            plt.ylabel('Conductance', fontsize=14)#(r'$\Phi$')
            plt.draw()
            fig1.savefig('/home/mmitri/Documents/Stage/Codes/Research/NCPpaper/NCP_infomap'+network_name+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
            
