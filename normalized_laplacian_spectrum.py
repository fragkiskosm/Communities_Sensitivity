# -*- coding: utf-8 -*-
"""

@author: mmitri
"""
import networkx as nx
from scipy import *

def normalized_laplacian_spectrum(G, weight='weight'):
    """Return eigenvalues of the Laplacian of G

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See to_numpy_matrix for other options.

    See Also
    --------
    laplacian_matrix
    """
    from scipy.linalg import eigvalsh
    #return eigvalsh(nx.normalized_laplacian_matrix(G,weight=weight).todense())
    import scipy.sparse as sparse

    w , v = sparse.linalg.eigsh(nx.normalized_laplacian_matrix(G,weight=weight), which='SM')
    return w

