import random
from itertools import product, combinations
from typing import List
from scipy import stats
import numpy as np
import plotly.express as px

from models.statistical.movment_tracker import MovementSeries


def calculate_joint_direction_amount(movement_series_array: List[MovementSeries], sim_len):
    # directions1 = []
    # directions2 = []
    # angles1 = []
    # angles2 = []
    # for i in range(sim_len):
    #     directions_at_time_i = []
    #     angles_at_time_i = []
    #     positions_at_time_i = []
    #     for movement_series in movement_series_array:
    #         if movement_series.is_i_in_range(i - 1) and movement_series.is_i_in_range(i):
    #             position = movement_series.get_ith_position(i - movement_series.get_start_i())
    #             direction = movement_series.get_ith_direction_4(i - movement_series.get_start_i())
    #             angle = movement_series.get_ith_angle(i - movement_series.get_start_i())
    #             if direction != -1:
    #                 directions_at_time_i.append(direction)
    #                 angles_at_time_i.append(angle)
    #                 positions_at_time_i.append(position)
    #
    #     if len(directions_at_time_i) >= 2:
    #         a, b = random.sample(directions_at_time_i, k=2)
    #         directions1.append(a)
    #         directions2.append(b)
    #     if len(angles_at_time_i) >= 2:
    #         for a, b in combinations(enumerate(angles_at_time_i), 2):
    #             if np.sqrt((positions_at_time_i[a[0]][0] - positions_at_time_i[b[0]][0]) ** 2 +
    #                        (positions_at_time_i[a[0]][1] - positions_at_time_i[b[0]][1]) ** 2) < 40:
    #                 angles1.append(a[1])
    #                 angles2.append(b[1])
    # arr3 = []
    # for i in range(len(directions1)):
    #     arr3.append((directions1[i] - directions2[i]) % 3)
    # print(f"mean of the difference in mutual direction (random walk would be 1): {np.mean(arr3)}")
    # angles1 = np.sin(np.deg2rad(angles1))
    # angles2 = np.sin(np.deg2rad(angles2))
    # res = stats.spearmanr(angles1, angles2)
    # print(f"spearman correlation: {res.statistic}, p-value: {res.pvalue}")
    # res = stats.pearsonr(angles1, angles2)
    # print(f"pearson correlation: {res.statistic}, p-value: {res.pvalue}")
    #
    # seen = {}
    # x = []
    # y = []
    # amounts = []
    # for i in range(len(angles1)):
    #     pair = (angles1[i], angles2[i])
    #     if pair in seen:
    #         amounts[seen[pair]] += 1
    #     else:
    #         seen[pair] = len(amounts)
    #         amounts.append(0)
    #         x.append(pair[0])
    #         y.append(pair[1])
    #
    # fig = px.scatter(x=x, y=y, size=amounts)
    # fig.show()

    total_together = 0
    total_opposite = 0
    total_side = 0
    for i in range(sim_len):
        angles_at_time_i = []
        positions_at_time_i = []
        for movement_series in movement_series_array:
            if movement_series.is_i_in_range(i - 1) and movement_series.is_i_in_range(i):
                position = movement_series.get_ith_position(i - movement_series.get_start_i())
                direction = movement_series.get_ith_direction_4(i - movement_series.get_start_i())
                angle = movement_series.get_ith_angle(i - movement_series.get_start_i())
                if direction != -1:
                    angles_at_time_i.append(angle)
                    positions_at_time_i.append(position)

        if len(angles_at_time_i) >= 2:
            for a, b in combinations(enumerate(angles_at_time_i), 2):
                angle1 = a[1]
                angle2 = b[1]
                dif = np.abs(angle1 - angle2) % 360
                if dif > 180:
                    dif = 360 - dif
                if dif < 45:
                    total_together += 1
                elif dif < 135:
                    total_side += 1
                else:
                    total_opposite += 1
    print(f"together: {total_together}, side: {total_side}, opposite:{total_opposite}")
    # together: 1182, side: 3160, opposite:1981
