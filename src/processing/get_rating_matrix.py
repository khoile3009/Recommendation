import os
import random

import numpy as np
import pandas as pd
from src.processing.load_data import load_dataframe


def get_cache_path(num_people, num_item):
    return f"processed/sparse_{num_people}_{num_item}.npy"



def get_sparse_matrix(num_people = 1000, num_item = 1000):
    cache_path = get_cache_path(num_people, num_item)
    if os.path.exists(cache_path):
        target_matrix = np.load(cache_path)
    else:
        rating_df = load_dataframe("ratings")
        target_matrix = np.zeros([num_people, num_item])
        for index, row in pd.DataFrame.iterrows(rating_df):
            if row["userId"] < num_people:
                if row["movieId"] < num_item:
                    target_matrix[int(row["userId"]) - 1, int(row["movieId"]) - 1] = row["rating"]
            else:
                break
        np.save(cache_path, target_matrix)
    return target_matrix

def split_matrix(rating_matrix, validation_ratio = 0.1):
    y_indices, x_indices = rating_matrix.nonzero()
    num_non_zero = len(y_indices)
    validation_indices = random.sample(range(num_non_zero), int(num_non_zero * validation_ratio))
    validation_matrix = np.zeros(rating_matrix.shape)
    for validation_index in validation_indices:
        validation_matrix[y_indices[validation_index]][x_indices[validation_index]], rating_matrix[y_indices[validation_index]][x_indices[validation_index]] = rating_matrix[y_indices[validation_index]][x_indices[validation_index]], 0
    return rating_matrix, validation_matrix
if __name__ == "__main__": 
    rating_matrix = get_sparse_matrix(100, 100)
    rating_matrix, validation_matrix = split_matrix(rating_matrix)
    sparsity = len(rating_matrix.nonzero()[0])
    sparsity /= 10000
    print(sparsity)
    sparsity = len(validation_matrix.nonzero()[0])
    sparsity /= 10000
    print(sparsity)
