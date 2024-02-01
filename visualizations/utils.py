import numpy as np


def matplotlib_to_plotly(cmap, pl_entries, zero_color=None):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        if k == 0 and zero_color != None:
            pl_colorscale.append([k * h, 'rgb' + zero_color])
        else:
            C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale
