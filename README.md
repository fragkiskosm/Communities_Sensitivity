# Communities_Sensitivity

## Requirements ##
* Linux (and other Unix-like systems)
* Python 2.7.9 (Anaconda 2.2.0)
* python-igraph 0.7.1.post4 (containing all community detection algorithms except Spectral clustering and Metis)
* scipy 0.15.1
* networkx 1.9.1
* conda 3.10.0
* metis 0.1a0
* scikit-learn 0.15.2
* matplotlib 1.4.3
* numpy 1.9.2

## Description of scripts ##

* `NetworksFromSNAP.py`: script containing all functions needed to read the datasets (i.e., graphs), put them in iGraph/NetworkX or Metis format, extract the greatest connected component of each graph (GCC) and perturb the GCC using one of the perturbation strategies. It also contains the function METIS_Clustering that allows to use Metis algorithm to detect communities.

* `Add_Noise.py`: sciprt containing the functions used for graph perturbation: function configuration_model_perturbation (which is also the null model normally used in the definition  of the modularity), function uniform_perturbation (Noise based on uniform perturbation (Erdos-Renyi G(n,e/n) model)), function preferential_perturbation (Noise based on preferential perturbation to high degree nodes (Chung-Lu model)).

* `NCP_plot_for_paper.py`: script allowing to produce the figure "NCP plot of a real-world network (CA-HEP-TH) using Infomap algorithm."

* `Plot_and_save_NCP_charts.py`: function allowing to produce the structural sensitivity plots for the community detection algorithms. It also produces plots of the distribution (CDF) of clusters' sizes as function of noise.

* `normalized_laplacian_spectrum.py`: function that return the eigenvalues of the Laplacian of the graph.

* `NCP_plot.py`: function computing the NCP plot of a graph. The NCP plot measures the quality of the best possible community in a large network, as a function of the community size. See Lekovec et al. 2008 : "Community Structure in Large Networks Natural Cluster Sizes and the Absence of Large Well-Defined Clusters".

* `variation_of_information_score.py`: function used to compute the VI score for Spectral and Metis algorithm (their output format, i.e. the community structure, differs from the ones of algorithms using Igraph format graphs).

* `community_quality_NetworkX.py`: script containing functions modularity and conductance scores.

* `community_utils_NetworkX.py`: utils used by community_quality_NetworkX.py.
