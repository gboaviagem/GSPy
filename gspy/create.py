"""Utilites for creating adjacency matrices or coordinate arrays."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def nearest_neighbors(X, n_neighbors=20, algorithm='ball_tree',
                      mode='distance', allow_self_loops=False):
    """Return the nearest neighbors' graph weighted adjacency matrix.

    This is a wrapper for the Scikit-learn NearestNeighbors.kneighbors_graph
    method.

    Parameters
    ----------
    X : np.ndarray()
    n_neighbors : int, optional, default: 20
    algorithm : str, optional, default: 'ball_tree'
    mode : str, optional, default: 'distance'

	Return
	------
	W : weighted adjacency matrix in CSR (Compressed Sparse Row) format
    """
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm).fit(X)
    W = nbrs.kneighbors_graph(X, mode=mode)

    return W

def adj_matrix_from_coords(coords, theta):
	"""Create a gaussian-weighted adjacency matrix using euclidean distance.

	Parameters
	----------
	coords : array
		(N, 2) array of coordinates.
	theta : float
		Variance of the weight distribution.

	References
	----------
	SHUMAN, D. I. et al. The emerging field of signal processing on graphs:
	Extending high-dimensional data analysis to networks and other irregular
	domains. IEEE Signal Process. Mag., IEEE, v. 30, n. 3, p. 83–98, 2013

	"""
	[N,M] = coords.shape
	A = np.zeros((N,N))
	for	i in tqdm(np.arange(1,N)):
		for j in np.arange(i):
			x1 = coords[i,0]
			y1 = coords[i,1]
			x2 = coords[j,0]
			y2 = coords[j,1]
			distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			if distance < 2*theta:
				A[i,j] = np.exp(-(distance**2)/(2*theta**2))
	print('adj_matrix_from_coords process is completed.')
	return A + A.transpose()

def adj_matrix_from_coords_limited(coords, limit):
	"""Create a nearest-neighbors graph with gaussian weights.

	Parameters
	----------
	coords : array
		(N, 2) array of coordinates.
	limit : int
		Minimum number of neighbors.

	References
	----------
	SHUMAN, D. I. et al. The emerging field of signal processing on graphs:
	Extending high-dimensional data analysis to networks and other irregular
	domains. IEEE Signal Process. Mag., IEEE, v. 30, n. 3, p. 83–98, 2013

	"""
	[N,M] = coords.shape
	A = np.zeros((N,N))
	for	i in tqdm(np.arange(1,N)):
		dist2i = np.sqrt((coords[:,0] - coords[i,0])**2 + (coords[:,1] - coords[i,1])**2)
		idx = np.argsort(dist2i)[1:limit+1]
		for j in idx:
			x1 = coords[i,0]
			y1 = coords[i,1]
			x2 = coords[j,0]
			y2 = coords[j,1]
			distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			if A[i,j] == 0:
				A[i,j] = np.exp(-(distance**2))
	return A + A.transpose()

def adj_matrix_from_coords2(coords, min_threshold):
	"""Create gaussian-weighted graphs, but with a weight threshold.

	Variation on the adj_matrix_from_coords function.

	Parameters
	----------
	coords : array
		(N, 2) array of coordinates.
	theta : float
		Variance of the weight distribution.

	"""
	[N,M] = coords.shape
	A = np.zeros((N,N))
	for	i in tqdm(np.arange(1,N)):
		for j in np.arange(i):
			x1 = coords[i,0]
			y1 = coords[i,1]
			x2 = coords[j,0]
			y2 = coords[j,1]
			distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			weight = 1.0/distance
			if weight > min_threshold:
				A[i,j] = weight
	print('adj_matrix_from_coords2 process is completed.')
	return A + A.transpose()

def adj_matrix_path(N, weights=None, directed=False):
	"""Returns the adjacency matrix of a path graph.

	Parameters
	----------
	N: int
		Number of graph nodes.
	weights: array, default=None
		Array with edge weights. If None, unit weights are used.
		If not None, then the given value of N is replaced by weights+1.
	directed: bool, default=False
		If True, a directed graph is created.

	"""
	if weights is None:
		A = np.tri(N, k=1) - np.tri(N, k=-2) - np.eye(N)
	else:
		N = len(weights) + 1
		A = np.zeros((N, N))
		A[:-1, 1:] = np.diag(weights)
		A = A + A.transpose()
	if directed:
		A = np.tril(A)
	return A

def coords_path(N):
	"""Coordinates of the vertices in the path graph.
	Parameters
	----------
	N : int
		Number of graph vertices.
	"""
	coords = np.array([[i, 0] for i in range(N)])
	return coords

def make_path(N, weights=None, directed=False):
	"""Create adjacency matrix and coordinates of a path graph.

	Parameters
	----------
	N: int
		Number of graph nodes.
	weights: array, default=None
		Array with edge weights. If None, unit weights are used.
		If not None, then the given value of N is replaced by weights+1.
	directed: bool, default=False
		If True, a directed graph is created.

	"""
	if weights is not None:
		assert N == len(weights) + 1, (
			"Length of weights array is {}, not compatible with "
			"{} vertices.".format(len(weights), N))
	A = adj_matrix_path(N, weights=weights, directed=directed)
	coords = coords_path(N)
	return A, coords

def make_grid(rows, columns, weights_r=None, weights_c=None):
	"""Create a grid graph.

	Parameters
	----------
	rows: int
		Number of rows in the grid.
	columns: int
		Number of columns in the grid.
	weights_r: array, default=None
		Weights in the rows.
	weights_c: array, default=None
		Weights in the columns.

	"""
	A1, coords1 = make_path(columns, weights=weights_c)
	A2, coords2 = make_path(rows, weights=weights_r)

	N1 = len(A1)
	N2 = len(A2)

	# Using the property that the grid graph is the cartesian product
	# of two path graphs.
	A = np.kron(A1, np.eye(N2)) + np.kron(np.eye(N1), A2)
	coords = list()
	for c1 in coords1[:, 0].ravel():
		for c2 in coords2[:, 0].ravel():
			coords.append([c1, c2])
	coords = np.array(coords)

	return A, coords

def adj_matrix_directed_ring(N, c=0):
	"""Returns the adjacency matrix of a ring graph.

	Parameters
	----------
	N: int
		Number of graph nodes.
	c: array
		First column of the adjacency matrix. It carries the edge weights.

	"""
	if c==0: # case in which the edge weights were not entered. Then, they are made equal to 1.
		c = np.zeros(N)
		c[1] = 1
	A = linalg.circulant(c)
	return A

def coords_ring_graph(N):
	"""Returns the vertices coordinates of a ring graph.

	Parameters
	----------
	N: int
		Number of graph nodes.

	"""
	coords = np.zeros((N,2))
	n = np.arange(N)
	coords[:,0] = np.cos(2.0*np.pi*n/N)
	coords[:,1] = -np.sin(2.0*np.pi*n/N)
	return coords

def coords_line_graph(A,coords,a):
	# Calculating the number of vertices of original graph
	N = len(coords)

	# Calculating the number of edges of original graph
	E = np.sum(A)

	coords_line_graph = np.zeros((E,2))

	row_idx = np.zeros(E,dtype=int)
	col_idx = np.zeros(E,dtype=int)

	e = 0
	for i in range(N):
		for j in range(N):
			if A[i,N-1-j]!=0:
				row_idx[e] = i
				col_idx[e] = N-1-j
				e = e + 1
		
	coords_line_graph[:,0] = coords[row_idx,0] + a*(coords[col_idx,0] - coords[row_idx,0])
	coords_line_graph[:,1] = coords[row_idx,1] + a*(coords[col_idx,1] - coords[row_idx,1])
	return coords_line_graph

def random_sensor_graph(N,theta=0.2):
	coords = np.random.rand(N,2)
	A = adj_matrix_from_coords(coords,theta)
	return A,coords

def line_graph(A,coords,a=0.5):
	# Warning: the graph is supposed to have only UNITARY weights.
	N = len(coords)
	A = 1*(A!=0) # FORCES UNITARY WEIGHTS.
	assert a > 0 and a < 1, \
		"Fractional parameter is out of bounds! (should be >0 and <1)"

	if np.array_equal(A.transpose(),A):
		# Undirected graph
		raise NotImplementedError("line_graph is not implemented for undirected graphs yet.")
	else:
		# Directed graph
		E = np.sum(A!=0) # number of edges
		LA = np.zeros((E,E))
		Lcoords = np.zeros((E,2))
		
		row_idx = np.zeros(E)
		col_idx = np.zeros(E)
		
		# In what follows, we pick the indexes of A corresponding to linked vertices. Each pair row_idx[i], col_idx[i] corresponds to an edge in the digraph, and therefore to a vertex in the line digraph. The order in which we pick them will define the ordering of vertices in the line digraph, and we do so column-by-column, from left to right, from top to bottom.
		col_idx, row_idx = np.where(np.transpose(A)!=0)

		Lcoords[:,0] = coords[row_idx,0] + a*(coords[col_idx,0] - coords[row_idx,0])
		Lcoords[:,1] = coords[row_idx,1] + a*(coords[col_idx,1] - coords[row_idx,1])

		for e in range(E):
			LA[np.where(col_idx==row_idx[e]),e] = 1
	return [LA,Lcoords]