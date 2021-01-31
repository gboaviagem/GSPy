"""Utilities for graph visualization based on matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import networkx as nx


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

def plt_graph(
		A, coords, display_edges=1, display_axis=0, color='b',
		graph_node_size=80, width=12, height=8):
	[rows,cols] = np.where(A!=0)
	plt.figure(figsize=(width, height))
	if display_edges==1:
		if np.array_equal(A.transpose(),A):
			# Undirected graph
			for i in tqdm(range(len(rows))):
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
			for j in tqdm(range(len(cols))):
				x1, y1 = coords[cols[j],0], coords[cols[j],1]
				x2, y2 = coords[rows[j],0], coords[rows[j],1]
				plt.arrow(
					x1, y1, x2-x1, y2-y1, head_width=h_length/2.0,
					head_length=h_length, fc='0.5', ec='0.5',
					length_includes_head=True, overhang=0.3, zorder=1)
	plt.scatter(
		coords[:,0], coords[:,1], s=graph_node_size, c=color,
		edgecolor='face', zorder=2)
	if display_axis == 0:
		plt.axis('off')
	plt.axis('tight')
	return True

def plt_graph_signal(
		A, coords, signal, display_edges=1, display_axis=0, cmin=0, cmax=0,
		graph_node_size=150, cfontsize=22, create_figure=True, verbose=True,
		edge_color_face=True, show_progress=False, arrow_scale=1.0):
	if verbose:
		print('plot_graph_signal has initiated.')
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
					print('plot_graph_signal: ', 100.0*i/len(rows), '% of loop completed.')
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
					print('plot_graph_signal: ', 100.0*j/len(cols), '% of loop completed.')
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
	if verbose:
		print('plot_graph_signal completed.')
	return True
