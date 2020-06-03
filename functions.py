import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os


def make_matrix(n_dim, angle, scale=1):
    """
    This method will return a square matrix with dimensionality given by n_dim.
    This matrix represents n_dim vectors that are placed at equal angles to one another.
    :param n_dim: Number of vectors to generate, as well as vector dimension.
    :param angle: Assuming an n_dim-dimensional basis for a vector space, this is the angle between
     an (n_dim - 1)-dimensional subspace and an n_dim-dimensional vector that lies outside of this subspace.
    :param scale: Length of output vectors.
    :return: n_dim * n_dim matrix representing equally angled vectors.
    """
    import numpy as np
    fill_value = scale * np.cos(angle) / np.sqrt(2)
    diag_value = scale * np.sin(angle)
    matrix = np.full((n_dim, n_dim), fill_value)
    for i in range(n_dim):
        matrix[i][i] = diag_value
    return round_to_zero(matrix)


def round_val(value):
    if abs(value) <= pow(10, -10):
        return 0
    return value


def round_to_zero(matrix):
    import numpy as np
    return np.vectorize(round_val)(matrix)


def make_rot_matrix_x(n_dim, angle):
    import numpy as np
    matrix = np.diag(np.ones(n_dim))
    matrix[n_dim - 1][n_dim - 1] = np.cos(angle)
    matrix[n_dim - 2][n_dim - 2] = np.cos(angle)
    matrix[n_dim - 2][n_dim - 1] = -np.sin(angle)
    matrix[n_dim - 1][n_dim - 2] = np.sin(angle)
    return matrix


def rotate_x(vector, angle):
    import numpy as np
    vector = np.array(vector)
    matrix = make_rot_matrix_x(vector.shape[0], angle)
    return np.matmul(matrix, vector)


def count_in_box(data, l):
    return (data <= l).prod(axis=1).sum(axis=0)


def parse_params_from_file(file_name):
    parts = file_name.split('_')
    theta_str = parts[3][:-5]
    dim_str = parts[4].split('.')[0][:-3]
    return float(theta_str), int(dim_str)


def make_norm(data_row, order):
    import numpy as np
    output = np.power(np.sum(np.power(np.abs(data_row), order)), 1/order)
    return output


def make_norms(data, norm_order):
    var_names = [x for x in data.columns.values if x.startswith('V')]
    norm_name = 'l' + str(norm_order)
    if norm_name in data.columns:
        data = data.drop([norm_name], axis=1)
    data[norm_name] = data[var_names].apply(lambda x: make_norm(x, norm_order), axis=1)
    return min(data[norm_name]), max(data[norm_name])
