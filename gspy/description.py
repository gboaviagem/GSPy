"""Networkx-related functions."""

import networkx as nx
import numpy as np


def describe(graph, return_dict=False):
    """Compute and print some graph-related metrics.

    Parameters
    ----------
    graph : nx.Graph() or np.ndarray
        If Numpy array, it is assumed to be the adjacency matrix.
    return_dict : bool, optional, default: False
        Whether to return a dictionary with info.
    """
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_matrix(graph)

    if isinstance(graph, nx.DiGraph):  # MultiDiGraph is also considered here
        is_dir = True
    else:
        is_dir = False

    d = dict()
    d["n_nodes"] = nx.number_of_nodes(graph)
    d["n_edges"] = nx.number_of_edges(graph)
    d["n_self_loops"] = nx.number_of_selfloops(graph)
    # d["n_triangles"] = nx.triangles(graph)
    # d["average_clustering"] = nx.average_clustering(graph)
    d["density"] = nx.density(graph)
    if is_dir:
        d["is_strongly_connected"] = nx.is_strongly_connected(graph)
        d["n_strongly_connected_components"] = nx.number_strongly_connected_components(graph)
        d["is_weakly_connected"] = nx.is_weakly_connected(graph)
        d["n_weakly_connected_components"] = nx.number_strongly_connected_components(graph)
    else:
        d["is_connected"] = nx.is_connected(graph)
        d["n_connected_components"] = nx.number_connected_components(graph)
    d["is_directed"] = nx.is_directed(graph)
    d["is_weighted"] = nx.is_weighted(graph)

    if return_dict:
        return d
    else:
        for key, value in d.items():
            print(key + ":", value)
