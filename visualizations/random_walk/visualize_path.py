import numpy as np
from models.statistical.movment_tracker import MovementSeries
import plotly.express as px
import plotly.graph_objects as go
import typing


def visualize_path(movement_series_list: list[MovementSeries], save_path=None):
    """Generates a path graph"""
    # find the longest movement series:
    longest_series = movement_series_list[0]
    for movement_series in movement_series_list:
        if len(movement_series) > len(longest_series):
            longest_series = movement_series

    # get x and y arrays
    x, y, = [], []
    positions = longest_series.get_positions()
    for x_, y_ in positions:
        x.append(x_ - 20)
        y.append(y_ - 20)
    time = [i * 0.1 for i in range(len(x))]

    # generate the figure
    fig1 = px.line(x=x, y=y)
    fig1.update_traces(line=dict(color='rgba(50,50,50,0.2)'))
    fig = px.scatter(x=x,
                     y=y,
                     title="Tracked Signal Path Over Time",
                     labels={
                         "x": "Distance (nm)",
                         "y": "Distance (nm)",
                         "color": "Time (µs)"
                     },
                     template="simple_white",
                     color=time,
                     color_continuous_scale="pubu"
                     )
    fig.add_shape(type="circle",
                  xref="x",
                  yref="y",
                  x0=-11, y0=-11, x1=11, y1=11,
                  fillcolor='rgba(50,50,50,0)',
                  line=dict(color='rgba(50,50,50,0.8)', width=2))
    fig.add_shape(type="circle",
                  xref="x",
                  yref="y",
                  x0=-18.5, y0=-18.5, x1=18.5, y1=18.5,
                  fillcolor='rgba(50,50,50,0)',
                  line=dict(color='rgba(50,50,50,0.8)', width=2))
    fig.add_trace(fig1.data[0])
    fig.update_xaxes(range=[-20, 20],
                     dtick=5)
    fig.update_yaxes(range=[-20, 20],
                     dtick=5,
                     scaleanchor="x",
                     scaleratio=1)

    if save_path is None:
        fig.show(width=600, height=600)
    else:
        fig.write_image(save_path, width=600, height=600)
