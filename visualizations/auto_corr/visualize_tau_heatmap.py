from visualizations.utils import matplotlib_to_plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def visualize_taus(taus, file_path):
    cm = plt.get_cmap("jet")
    cm = matplotlib_to_plotly(cm, 255, zero_color=None)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=np.fliplr(np.flipud(taus)), colorbar={"title": 'Tau'}, zmin=0, zmax=0.4,
                   colorscale=cm))
    fig.layout.height = 500
    fig.layout.width = 500
    fig.update_layout(xaxis={
        "tickmode": 'array',
        "tickvals": [i - 0.5 for i in range(0, 41, 5)],
        "ticktext": [i for i in range(-20, 21, 5)]
    }, yaxis={
        "tickmode": 'array',
        "tickvals": [i - 0.5 for i in range(0, 41, 5)],
        "ticktext": [i for i in range(-20, 21, 5)]
    })
    fig.update_layout(title="",
                      yaxis={"title": 'Distance (nm)'},
                      xaxis={"title": 'Distance (nm)',
                             "tickangle": 0},
                      font=dict(size=20))
    fig.add_shape(
        {'type': "circle", 'x0': 8.5 - 0.5, 'y0': 8.5 - 0.5, 'x1': 31.5 + 0.5, 'y1': 31.5 + 0.5,
         'xref': f'x',
         'yref': f'y', "line": dict(width=2, color="Black"), 'opacity': 0.7}, )
    fig.add_shape(
        {'type': "circle", 'x0': 1 - 0.5, 'y0': 1 - 0.5, 'x1': 39 + 0.5, 'y1': 39 + 0.5,
         'xref': 'x',
         'yref': 'y', "line": dict(width=2, color="Black"), 'opacity': 0.7}, )
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    fig.write_image(file_path, width=550, height=500)
