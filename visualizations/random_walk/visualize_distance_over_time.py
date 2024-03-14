import numpy as np
from models.statistical.movment_tracker import MovementSeries
import plotly.express as px
from scipy.stats import sem
import typing


def visualize_distance_over_time(movement_series_list: list[MovementSeries], save_path=None):
    """Generates a mean square deviation graph"""
    totals, amounts, distances_by_length = [], [], []
    for movement_series in movement_series_list:
        positions = movement_series.get_positions()
        start_x, start_y = positions[0]
        for i, (x, y) in enumerate(positions):
            if len(totals) <= i:
                totals.append(0)
                amounts.append(0)
                distances_by_length.append([])
            dist = (start_x - x) ** 2 + (start_y - y) ** 2
            totals[i] += dist
            amounts[i] += 1
            distances_by_length[i].append(dist)
    avgs = np.array(totals) / np.array(amounts)
    x_axis = [i * 0.1 for i in range(len(avgs))]
    fig = px.scatter(x=x_axis,
                     y=avgs,
                     title="Average MSD Over Time of Tracked Signals",
                     labels={
                         "x": "Time (µs)",
                         "y": "MSD (nm^2)"
                     },
                     template="simple_white"
                     )
    # fig.add_hline(y=11 ** 2,
    #               line_dash="dot",
    #               annotation_text="Inner Tunnel Radius",
    #               annotation_position="top left")

    # add standard error of the mean
    upper = []
    lower = []
    for i, distance in enumerate(distances_by_length):
        error = sem(distance)
        upper.append(avgs[i] + error)
        lower.append(avgs[i] - error)
    fig.add_trace(px.line(x=x_axis, y=upper).data[0])
    fig.add_trace(px.line(x=x_axis, y=lower).data[0])
    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path)
