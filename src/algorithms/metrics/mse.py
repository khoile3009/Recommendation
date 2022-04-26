
def mse_sparse(rating_matrix, prediction_matrix):
    y_non_zero_indices, x_non_zero_indices = rating_matrix.nonzero()
    n_non_zero = len(y_non_zero_indices)
    total_squared_error = 0
    for i in range(n_non_zero):
        total_squared_error += (rating_matrix[y_non_zero_indices[i]][x_non_zero_indices[i]] - prediction_matrix[y_non_zero_indices[i]][x_non_zero_indices[i]]) ** 2
    return total_squared_error /  n_non_zero