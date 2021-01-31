"""Plotly-based graphing functions."""

import numpy as np
import plotly.graph_objects as go
from gspy.utils import prepare_coords_for_plotly


def show_graph(
        coords, A=None, graph_signal=None, title=None,
        colorbar_title="Graph signal", verbose=True):
    """Plot a graph signal on a Plotly Graph Object.

    Parameters
    ----------
    coords : np.ndarray, shape=(N, 2)
        Nodes coordinates.
    A : np.ndarray, shape=(N,N), default=None
        Graph weighted adjacency matrix.
    graph_signal : np.ndarray, shape=(N,), default=None
        Graph signal.
    title : str, default=None
        Plot title.
    colorbar_title : str, default="Graph signal"
        Colorbar title.
    verbose : bool, default=True

    Return
    ------
    fig : go.Figure

    """
    # Adding the nodes
    markers = dict(color='blue', line_width=1) if graph_signal is None \
        else dict(
            color=graph_signal,
            colorscale='Viridis',
            colorbar=dict(
                title=colorbar_title
            ),
            line_width=1
        )

    fig = go.Figure(data=go.Scattergl(
        x = coords[:, 0].ravel(),
        y = coords[:, 1].ravel(),
        mode='markers',
        marker=markers
    ))

    # Plotting the edges
    if A is not None:
        x_array, y_array = prepare_coords_for_plotly(A, coords, verbose)

        fig.add_trace(go.Scatter(
            x=x_array,
            y=y_array,
            line=dict(
                color='gray',
                width=1,
                dash=None),
            name='Gaps',
        ))

    layout = dict(
        font_size=12,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title="",
            visible=False,
            zeroline=False),
        yaxis=dict(
            title="",
            gridwidth=1,
            visible=False,
            zeroline=True,
            zerolinewidth=1,
        ))

    fig.update_layout(**layout)
    return fig
