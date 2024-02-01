import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from itertools import product
import matplotlib.pyplot as plt
import visualizations.utils as utils


def visualize_trajectories(map, vector_field):
    """Given a raster and vector field, visualizes it"""
    x, y, v, u = [], [], [], []
    for x_, y_ in product(range(vector_field.shape[0]), range(vector_field.shape[1])):
        x.append(x_)
        y.append(y_)
        v.append(vector_field[x_, y_, 0])
        u.append(vector_field[x_, y_, 1])
    fig = ff.create_quiver(x, y, u, v,
                           scale=30,
                           arrow_scale=0.6,
                           name='quiver',
                           line=dict(color="#bf15c2"))

    cm = plt.get_cmap("jet")
    cm = utils.matplotlib_to_plotly(cm, 255)
    fig.add_trace(go.Heatmap(z=np.swapaxes(map, 0, 1), colorscale=cm))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(width=800, height=800, xaxis_range=[0, 40], yaxis_range=[0, 40])

    fig.show()
