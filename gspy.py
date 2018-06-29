import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def adj_matrix_from_coords(coords,theta,show_progress=False):
	[N,M] = coords.shape
	A = np.zeros((N,N))
	for	i in np.arange(1,N):
		if show_progress:
			print 100.0*i/N, '% of adj_matrix_from_coords process completed.'
		for j in np.arange(i):
			x1 = coords[i,0]
			y1 = coords[i,1]
			x2 = coords[j,0]
			y2 = coords[j,1]
			distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			if distance < 2*theta:
				A[i,j] = np.exp(-(distance**2)/(2*theta**2))
	print 'adj_matrix_from_coords process is completed.'
	return A + A.transpose()

def adj_matrix_from_coords_limited(coords,limit):
	print 'adj_matrix_from_coords_limited has initiated.'
	[N,M] = coords.shape
	A = np.zeros((N,N))
# 	coords_dist = np.sqrt((coords[:,0])**2 + (coords[:,1])**2)
	for	i in np.arange(1,N):
# 		print 'adj_matrix_from_coords_limited: ', 100.0*i/N, '% completed.'
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
	print 'adj_matrix_from_coords_limited process is completed.'
	return A + A.transpose()

def adj_matrix_from_coords2(coords,min_threshold):
	[N,M] = coords.shape
	A = np.zeros((N,N))
	for	i in np.arange(1,N):
		print 100.0*i/N, '% of adj_matrix_from_coords2 process completed.'
		for j in np.arange(i):
			x1 = coords[i,0]
			y1 = coords[i,1]
			x2 = coords[j,0]
			y2 = coords[j,1]
			distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
			weight = 1.0/distance
			if weight > min_threshold:
				A[i,j] = weight
	print 'adj_matrix_from_coords2 process is completed.'
	return A + A.transpose()

def adj_matrix_directed_ring(N,c=0):
	# Returns the adjacency matrix of a ring graph.
	# N: number of graph nodes.
	# c: first column of the adjacency matrix. It carries the edge weights.
	if c==0: # case in which the edge weights were not entered. Then, they are made equal to 1.
		c = np.zeros(N)
		c[1] = 1
	A = linalg.circulant(c)
	return A

def coords_ring_graph(N):
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

def plot_graph(A,coords,display_edges=1,display_axis=0,color='b',graph_node_size=80,show_progress=False):
	print 'plot_graph has initiated.'
	[rows,cols] = np.where(A!=0)
	plt.figure()
	if display_edges==1:
		if np.array_equal(A.transpose(),A):
			# Undirected graph
			for i in range(len(rows)):
				if show_progress:
					print 'plot_graph: ', 100.0*i/len(rows), '% of loop completed.'
				x1, y1 = coords[cols[i],0], coords[cols[i],1]
				x2, y2 = coords[rows[i],0], coords[rows[i],1]
				plt.plot([x1, x2], [y1, y2], c='0.5',zorder=1)
		else:
			# Directed graph
			# Arrow parameters (set proportionally to the plot dimensions)
			x_max = np.max(coords[:,0])
			x_min = np.min(coords[:,0])
			y_max = np.max(coords[:,1])
			y_min = np.min(coords[:,1])
			h_length = 0.04*np.max([x_max - x_min, y_max - y_min])
			# Drawing the edges (arrows)
			for j in range(len(cols)):
				if show_progress:
					print 100.0*j/len(cols), '% of loop completed.'
				x1, y1 = coords[cols[j],0], coords[cols[j],1]
				x2, y2 = coords[rows[j],0], coords[rows[j],1]
				plt.arrow(x1, y1, x2-x1, y2-y1, head_width=h_length/2.0, head_length=h_length, fc='0.5', ec='0.5',length_includes_head=True,overhang=0.3,zorder=1)
	plt.scatter(coords[:,0],coords[:,1],s=graph_node_size,c=color,edgecolor='face',zorder=2)
	if display_axis==0:
		plt.axis('off')
	plt.axis('tight')
	print 'plot_graph completed.'
	return True

def plot_graph_signal(A,coords,signal,display_edges=1,display_axis=0,cmin=0,cmax=0,graph_node_size=150,cfontsize=22,create_figure=True,edge_color_face=True,show_progress=False,arrow_scale=1.0):
	print 'plot_graph_signal has initiated.'
	if cmin==cmax:
		# case in which the user did not specify the colormap range.
		cmin = np.min(signal)
		cmax = np.max(signal)
	[rows,cols] = np.where(A!=0)
	if create_figure:
		plt.figure()
	if display_edges==1:
		if np.array_equal(A.transpose(),A):
			# Undirected graph
			for i in range(len(rows)):
				if show_progress:
					print 'plot_graph_signal: ', 100.0*i/len(rows), '% of loop completed.'
				x1, y1 = coords[cols[i],0], coords[cols[i],1]
				x2, y2 = coords[rows[i],0], coords[rows[i],1]
				plt.plot([x1, x2], [y1, y2], c='0.5',zorder=1)
		else:
			# Directed graph
			# Arrow parameters (set proportionally to the plot dimensions)
			x_max = np.max(coords[:,0])
			x_min = np.min(coords[:,0])
			y_max = np.max(coords[:,1])
			y_min = np.min(coords[:,1])
			h_length = 0.05*np.max([x_max - x_min, y_max - y_min])
			# Drawing the edges (arrows)
			for j in range(len(cols)):
				if show_progress:
					print 'plot_graph_signal: ', 100.0*j/len(cols), '% of loop completed.'
				x1, y1 = coords[cols[j],0], coords[cols[j],1]
				x2, y2 = coords[rows[j],0], coords[rows[j],1]
				plt.arrow(x1, y1, x2-x1, y2-y1, head_width=arrow_scale*h_length/2.0, head_length=arrow_scale*h_length, fc='0.5', ec='0.5',length_includes_head=True,overhang=0.3,zorder=1)
	if edge_color_face==True:
		plt.scatter(coords[:,0],coords[:,1],s=graph_node_size,c=signal,edgecolor='face',zorder=2)
	else:
		plt.scatter(coords[:,0],coords[:,1],s=graph_node_size,c=signal,zorder=2)
	cticks = np.linspace(cmin, cmax, 5, endpoint=True)
	if create_figure:
		cbar = plt.colorbar()
		plt.clim(cmin,cmax)
		cbar.ax.tick_params(labelsize=cfontsize)
	if display_axis==0:
		plt.axis('off')
	plt.axis('tight')
	print 'plot_graph_signal completed.'
	return True

def random_sensor_graph(N,theta=0.2):
	coords = np.random.rand(N,2)
	A = adj_matrix_from_coords(coords,theta)
	return A,coords

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

def stem(x,y,fsize=18,msize=10,color='b',linestyle='--',labelstr=0,alph=1):
	for i in range(np.array(x).size):
		plt.plot([x[i], x[i]], [0, y[i]],linestyle,c=color,zorder=1)
	if labelstr!=0:
		plt.scatter(x,y,s=10*msize,c=color,edgecolor='face',zorder=2,alpha=alph,label=labelstr)
	else:
		plt.scatter(x,y,s=10*msize,c=color,edgecolor='face',alpha=alph,zorder=2)
	plt.axis('tight')
	plt.tick_params(axis='both', which='major', labelsize=fsize-2)
	return True

def laplacian(A):
	# Returns the Laplacian of a graph, considering the in-degree matrix.
	[N,M] = A.shape
	if N!=M:
		print "Error! Adjacency matrix is not square."
		return 0
	Din = np.diag(np.sum(A,axis=1)) # in-degree matrix
	L = Din - A
	return L

def find_nearest(array,value):
	# Function to find the entry of the array "array" closest to the value "value"
	# CREDITS: written by "unutbu", as in https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    idx = (np.abs(array-value)).argmin()
    return array[idx]

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

def normalize_mtx_l1(A):
	# Returns a version of the matrix with each column normalized to 1
	# with respect to the l1-norm (sum == 1).
	s = np.sum(A,axis=0)
	sinv = 1.0/s
	return np.dot(A,np.diag(sinv))

def line_graph(A,coords,a=0.5):
	# Warning: the graph is supposed to have only UNITARY weights.
	N = len(coords)
	A = 1*(A!=0) # FORCES UNITARY WEIGHTS.
	if (a<=0) or (a>=1):
		print "Error! Fractional parameter is out of bounds! (should be >0 and <1)"
		return 0
	
	if np.array_equal(A.transpose(),A):
		# Undirected graph
		print "Error. line_graph is not implemented for undirected graphs yet."
		return 1
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

def gft(M,x,showprogress=False):
	'''
	GFT of a signal as decomposition into the eigenbasis of matrix M.
	>> M: adjacency or Laplacian matrix.
	'''
	if showprogress:
		print 'Starting the computation of the Fourier basis.'
	[eigvals,V] = np.linalg.eig(M)
	if showprogress:
		print 'Computing the Fourier matrix.'
	Minv = np.linalg.inv(M)
	xhat = np.dot(Minv,x) # possibly a complex array!
	return xhat
	
