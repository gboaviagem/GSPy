"""Graph operators."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def gft(M,x,showprogress=False):
	'''
	GFT of a signal as decomposition into the eigenbasis of matrix M.
	>> M: adjacency or Laplacian matrix.
	'''
	if showprogress:
		print('Starting the computation of the Fourier basis.')
	[eigvals,V] = np.linalg.eig(M)
	if showprogress:
		print('Computing the Fourier matrix.')
	Minv = np.linalg.inv(M)
	xhat = np.dot(Minv,x) # possibly a complex array!
	return xhat

def total_variation(A,x,eigval_max=0):
	if eigval_max == 0:
		# If eigval_max == 0 then we suppose the user didn't want to diagonalize A beforehand.
		[eigvals,V] = np.linalg.eig(A)
		eigval_max = np.max(np.abs(eigvals))
	Anorm = A/(1.0*eigval_max)
	TV = np.sum(np.abs(x - np.dot(Anorm,x)))
	return TV

def total_variation_2(A,x,eigval_max=0):
	if eigval_max == 0:
		# If eigval_max == 0 then we suppose the user didn't want to diagonalize A beforehand.
		[eigvals,V] = np.linalg.eig(A)
		eigval_max = np.max(np.abs(eigvals))
	Anorm = A/(1.0*eigval_max)
	TV = np.linalg.norm(x - np.dot(Anorm,x))
	return TV

def translation_girault(L):
	[eigvals,U] = np.linalg.eig(L)
	eigvals[np.where(eigvals < 1e-15)] = 0
	Uinv = np.linalg.inv(U)
	d = np.diag(L) # degrees
	A = np.diag(d) - L
	rho = np.max(np.dot(A,d)/(1.0*d)) # Obs.: element-wise division
	N = len(eigvals)
	TGhat = np.diag(np.exp(- 1j * np.pi * np.sqrt(eigvals/rho)))
	TG = np.dot(U,np.dot(TGhat,Uinv))
	return TG

def laplacian(A):
	# Returns the Laplacian of a graph, considering the in-degree matrix.
	[N,M] = A.shape
	assert N == M, "Error! Adjacency matrix is not square."

	Din = np.diag(np.sum(A,axis=1)) # in-degree matrix
	L = Din - A
	return L