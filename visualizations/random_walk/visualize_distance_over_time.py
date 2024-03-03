import numpy as np
from models.statistical.movment_tracker import MovementSeries
import plotly.express as px
import typing


def visualize_distance_over_time(movement_series_list: list[MovementSeries], save_path=None):
    """Generates a mean square deviation graph"""
    totals, amounts = [], []
    for movement_series in movement_series_list:
        positions = movement_series.get_positions()
        start_x, start_y = positions[0]
        for i, (x, y) in enumerate(positions):
            if len(totals) <= i:
                totals.append(0)
                amounts.append(0)
            totals[i] += (start_x - x) ** 2 + (start_y - y) ** 2
            amounts[i] += 1
    avgs = np.array(totals) / np.array(amounts)
    fig = px.scatter(x=[i * 0.1 for i in range(len(avgs))],
                     y=avgs,
                     title="Average MSD over time of tracked signals",
                     labels={
                         "x": "Time (µs)",
                         "y": "MSD (nm^2)"
                     }
                     )
    # fig.add_hline(y=11 ** 2,
    #               line_dash="dot",
    #               annotation_text="Inner Tunnel Radius",
    #               annotation_position="top left")
    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path)
