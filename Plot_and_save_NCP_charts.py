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
import matplotlib as mpl


def Plot_and_save_NCP_charts(network_name , noise_model , add_or_delete , nb_run , n_added_deleted_edges , epsilon_values , save__NCP_values_min_algo , save__NCP_k_values_algo , save__NCP_k_distribution , Fiedler_value_per_noise , algo_name):
        
    ''' 
    Do several plots (NMI, VI, NCPs)
    
    '''
    
    SUM_k_that_minimizes_conductance = []
    SUM_global_min_conductance = []
    SUM_median_min_conductance = []

    ''' Plot k where global min(NCP) is reached '''
    
    #mpl.rcParams['font.family'] = 'serif'
    
    fig1, ax1 = plt.subplots()
    ax2 = ax1.twinx()
        
    for omega in range(nb_run):
        
        # NCP_values_min_algo is save__NCP_values_min_algo with key omega
        NCP_values_min_algo = save__NCP_values_min_algo[omega]
        NCP_k_values_algo = save__NCP_k_values_algo[omega]
        
        k_that_minimizes_conductance = []
        global_min_conductance = []
        median_min_conductance = []
        
        for z in range(len(epsilon_values)): 
            # index of the min
            k_min = np.argmin(NCP_values_min_algo[z])
            range_k = NCP_k_values_algo[z]
            
            k_that_minimizes_conductance.append(range_k[k_min])
            global_min_conductance.append(np.min(NCP_values_min_algo[z]))
            median_min_conductance.append(np.median(NCP_values_min_algo[z]))
        
        SUM_k_that_minimizes_conductance.append(k_that_minimizes_conductance)
        SUM_global_min_conductance.append(global_min_conductance)
        SUM_median_min_conductance.append(median_min_conductance)
            
    if (noise_model != 'CONFIG'):
        ax2.plot(np.mean(n_added_deleted_edges, axis=0), np.mean(SUM_k_that_minimizes_conductance, axis=0) , linewidth=2 , linestyle=':', marker='x', markeredgewidth=2, color='r')
        ax2.set_xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        ax2.plot(epsilon_values, np.mean(SUM_k_that_minimizes_conductance, axis=0) , linewidth=2 , linestyle=':', marker='x', markeredgewidth=2, color='r')
        ax2.set_xlabel('Percentage of rewired edges (%)', fontsize=14)
    #plt.title('k where global min(NCP plot) is reached per noise level')
    ax2.set_ylabel('k (number of nodes in the cluster)' , color='r', fontsize=14)
    #ax2.set_ylim(np.min(np.mean(SUM_k_that_minimizes_conductance, axis=0))-10 , np.max(np.mean(SUM_k_that_minimizes_conductance, axis=0))+10)
    ax2.legend(loc = 'center left', bbox_to_anchor = (0.232, -0.28), fontsize=10)
    ax2.set_xlim(-2,30)
    
    ''' Global minimum value '''
    
    if (noise_model != 'CONFIG'):
        ax1.plot(np.mean(n_added_deleted_edges, axis=0), np.mean(SUM_global_min_conductance, axis=0) , linewidth=3 , linestyle='-', marker='o', markeredgewidth=2, label='Global min', color='b')
        ax1.plot(np.mean(n_added_deleted_edges, axis=0), np.mean(SUM_median_min_conductance, axis=0) , linewidth=3 , linestyle='-', marker='+', markeredgewidth=3, label='Median', color='g', markeredgecolor='k')    
        ax1.plot(np.mean(n_added_deleted_edges, axis=0), np.mean(Fiedler_value_per_noise, axis=0) , linewidth=3 , linestyle='-',  marker='D', markeredgewidth=2, label='Spectral lower bound', color='m')
        ax1.set_xlabel('Percentage of perturbed edges (%)', fontsize=14)
    else:
        ax1.plot(epsilon_values, np.mean(SUM_global_min_conductance, axis=0) , linewidth=3 , linestyle='-', marker='o', markeredgewidth=2, label='Global min', color='b')
        ax1.plot(epsilon_values, np.mean(SUM_median_min_conductance, axis=0) , linewidth=3 , linestyle='-', marker='+', markeredgewidth=3, label='Median', color='g', markeredgecolor='k')    
        ax1.plot(epsilon_values, np.mean(Fiedler_value_per_noise, axis=0) , linewidth=3 , linestyle='-',  marker='D', markeredgewidth=2, label='Spectral lower bound', color='m')
        ax1.set_xlabel('Percentage of rewired edges (%)', fontsize=14)
    if (noise_model != 'CONFIG'):
        plt.title(algo_name, fontsize=10)
    else:
        plt.title(algo_name, fontsize=10)
    ax1.set_yscale('log')
    ax1.set_ylabel('Quality score of communities', fontsize=14)
    ax1.set_xlim(-2,30)
    ax1.legend(loc = 'center left', bbox_to_anchor = (0.232, -0.28), fontsize=10)
    plt.draw()
    
    # SAVE
    fig1.savefig('/home/mmitri/Documents/Stage/Codes/Results/'+noise_model+'/'+network_name+'/NCPsPlots_'+network_name+'_'+noise_model+'_'+add_or_delete+'_'+algo_name+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
    
    
    
    
    
    
    
    
    ''' Plot the distribution of clusters' sizes as function of noise '''
    fig = plt.figure()
        
    for omega in range(nb_run):
        
        NCP_k_values_algo = save__NCP_k_values_algo[omega]
        NCP_k_distribution = save__NCP_k_distribution[omega]
        
        colors = np.linspace(1,0,len(NCP_k_distribution))
        for z in range(len(NCP_k_distribution)):
            if omega==0:
                if (noise_model != 'CONFIG'):
                    plt.plot(NCP_k_values_algo[z] , NCP_k_distribution[z] , label='Epsilon='+str(round(np.mean(n_added_deleted_edges, axis=0)[z],2))+'%' , c=str(colors[z]) , linewidth=1.5)
                else:
                    plt.plot(NCP_k_values_algo[z] , NCP_k_distribution[z] , label='Alpha='+str(round(epsilon_values[z],2))+'%' , c=str(colors[z]) , linewidth=1.5)
            else:
                if (noise_model != 'CONFIG'):
                    plt.plot(NCP_k_values_algo[z] , NCP_k_distribution[z] , c=str(colors[z]) , linewidth=1.5)
                else:
                    plt.plot(NCP_k_values_algo[z] , NCP_k_distribution[z] , c=str(colors[z]) , linewidth=1.5)
            
    plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize=10)
    plt.ylim(0,1)
    if (noise_model != 'CONFIG'):
        plt.title(algo_name, fontsize=10)
    else:
        plt.title(algo_name, fontsize=10)
    plt.ylabel('CDF', fontsize=14)
    plt.xlabel('k (number of nodes in the cluster)', fontsize=14)

    # SAVE
    fig.savefig('/home/mmitri/Documents/Stage/Codes/Results/'+noise_model+'/'+network_name+'/CDFofClustersSizes_'+network_name+'_'+noise_model+'_'+add_or_delete+'_'+algo_name+'.pdf', dpi=500, format='pdf', bbox_inches="tight")
