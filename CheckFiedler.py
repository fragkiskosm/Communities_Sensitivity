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
from sklearn.metrics.cluster import normalized_mutual_info_score
from igraph import arpack_options
from NCP_plot import *
from normalized_laplacian_spectrum import *
from variation_of_information_score import *
from Plot_and_save_NCP_charts import *
from community_quality_NetworkX import *

'''============================================================================
#    PARAMETERS
#=============================================================================='''

''' Chose algos'''

ifBlondelMultilevel = False

ifFastGreedyMaxModularity = False

ifINFOMAP = False

ifLeadingEigenvector = False
arpack_options.maxiter=10000000

ifWalktrap = False

ifLabelPropagation = False

ifSpectralClustering = True
nb_clust_spectral = 40

ifMETIS = True
nb_clust_Metis = 40


ifSpinglass = False
ifMCL = False


algos_labels = []
algos_line_style = []
nb_active_algos = 2

''' Chose the network '''
network_name = ['Email-Enron' , 'GRQCcollab' , 'AstroPhysicsCollab' , 'Wiki-Vote' , 'as-Caida']
network_name = network_name[1]

''' Chose the number of run '''
nb_run = 1


''' Chose the noise model '''
noise_model = ['ERP' , 'CLP' , 'LPP' , 'CONFIG']
noise_model = noise_model[0]

add_or_delete = 'Delete' # Add or 'Delete'

if (noise_model != 'CONFIG'):
    if add_or_delete=='Delete':
        epsilon_valuesI = [ 0 , 0.01 , 0.2 , 0.4 , 0.6 , 0.8 , 1 , 1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5 , 6]
        epsilon_values = [i * 20 for i in epsilon_valuesI]
    elif add_or_delete=='Add':
        epsilon_valuesI = [ 0 , 0.01 , 0.2 , 0.4 , 0.6 , 0.8 , 1 , 1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5 , 6]
        epsilon_values = [i * 0.3 for i in epsilon_valuesI]
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
    
    
    for j,eps in enumerate(epsilon_values):
    
        convert_path_P = '/home/mmitri/Documents/Stage/Data/iGraphFormatPerturbed/' + network_name + '_iGraphFormat_Perturbed.txt'
        
        G_P_networkX = convert_to_igraph_format(convert_path_P, algo_name='iGraph_algos' , net_name=network_name , perturb_model=noise_model , epsilon = eps , addORdel=add_or_delete , link_pred_file=None)
        
        G_P = Graph.Read_Edgelist(f=convert_path_P, directed=False)
        
        # Eigendecomposition of the graph Laplacian
        L_spectrum = normalized_laplacian_spectrum(G_P_networkX)
        # Spectral lower bound value (see Leskovec 2008)
        Fiedler_value_per_noise[i,j] = L_spectrum[1]/float(2)
