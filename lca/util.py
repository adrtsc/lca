import numpy as np


def measure_border_cells(feature_values):
    # check which cells are border cells
    border_1 = feature_values['bbox-0'] == 0
    border_2 = feature_values['bbox-1'] == 0
    border_3 = feature_values['bbox-2'] == 2048
    border_4 = feature_values['bbox-3'] == 2048
    borders = border_1 | border_2 | border_3 | border_4
    borders = np.array(borders, dtype=int)

    return borders



