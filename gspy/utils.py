"""General-purpose utilities."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def normalize_mtx_l1(A, axis=0):
	# Returns a version of the matrix with each column (axis=0) or row (axis=1)normalized to 1
	# with respect to the l1-norm (sum == 1).
	s = np.sum(A,axis=axis)
	sinv = 1.0/s
	return np.dot(A,np.diag(sinv))

def undir2dir(A_undirected):
	# RANDOM ORIENTATION an undirected graph
	N = len(A_undirected)
	A_directed = np.zeros((N,N))
	for row in (np.arange(N-1)+1):
		for col in range(row):
			if A_undirected[row,col] != 0:
				if np.random.randint(2)==0:
					A_directed[row,col] = A_undirected[row,col]
				else:
					A_directed[col,row] = A_undirected[row,col]
	return A_directed

def find_nearest(array,value):
	# Function to find the entry of the array "array" closest to the value "value"
	# CREDITS: written by "unutbu", as in https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def merge_line_digraph(A,coords,LA,Lcoords):
	# Takes a digraph and its line graph and merges into one graph. The line graph vertices are the last ones in the coords_merged array.
	N = len(coords)
	E = len(Lcoords)
	coords_merged = np.zeros((N+E,2))
	A_merged = np.zeros((N+E,N+E))
	
	coords_merged[0:N,:] = coords
	coords_merged[N:N+E,:] = Lcoords
	
	col_idx, row_idx = np.where(np.transpose(A)!=0) # indexes of connected vertices in the digraph
	Lcol_idx, Lrow_idx = np.where(np.transpose(LA)!=0) # indexes of connected vertices in the line graph
	
	for e in range(E):
		A_merged[N+e,col_idx[e]] = 1 # from source-vertex to edge
		A_merged[row_idx[e],N+e] = 1 # from edge to end-vertex
	return [A_merged,coords_merged]

def number_of_zero_crossings(A,x):
	# Returns the number of zero crossings in the signal x
	# defined over the graph with A as adjacency matrix.
	nzc = 0
	if np.sum(x >= 0)==0 or np.sum(x > 0)==0 or np.sum(x <= 0)==0 or np.sum(x < 0)==0:
		return nzc # no zero crossings
	if not np.array_equal(A.transpose(),A):
		# Directed graph
		for i in range(len(x)):
			row = A[i,:]
			adj_nodes = np.where(row!=0)[0]
			for j in adj_nodes:
				if x[i]*x[j] < 0:
					nzc += 1
		return nzc
	# For undirected graphs:
	# Which sign is predominant in the signal, plus or minus? Checking:
	if np.sum(x>=0) > np.sum(x < 0):
		index_minus = np.where(x<0)[0]
		for i in index_minus:
			row = A[i,:]
			adj_nodes = np.where(row!=0)[0]
			for j in adj_nodes:
				if x[i]*x[j] < 0:
					nzc += 1
	else:
		index_plus = np.where(x>=0)[0]
		for i in index_plus:
			row = A[i,:]
			adj_nodes = np.where(row!=0)[0]
			for j in adj_nodes:
				if x[i]*x[j] < 0:
					nzc += 1
	return nzc