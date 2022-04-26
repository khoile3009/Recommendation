
from abc import ABC, abstractmethod

import numpy as np


class MatrixFactorizationModel(ABC):

    def __init__(
        self,
        num_users,
        num_items,
        num_features,
    ):
        self.num_features = num_features
        self.num_users = num_users
        self.num_items = num_items
        self.user_vectors = np.random.random((num_users, num_features))
        self.item_vectors = np.random.random((num_items, num_features))

    def check_input(self, rating_matrix = None, user_vectors = None, item_vectors = None):
        result = True
        if rating_matrix is not None:
            if rating_matrix.shape != (self.num_users, self.num_items):
                print(rating_matrix.shape, self.num_users, self.num_items)
                print('------------------- rating matrix shape is incorrect ---------------------')
                result = False
        if user_vectors is not None:
            if user_vectors.shape != (self.num_users, self.num_features):
                print('------------------- user vectors shape is incorrect ---------------------')
                result = False
        if item_vectors is not None:
            if item_vectors.shape != (self.num_items, self.num_features):
                print('------------------- item vectors shape is incorrect ---------------------')
                result = False
        return result

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def predict(self, user_id, item_id):
        return self.user_vectors[user_id - 1, :].dot(self.item_vectors[item_id - 1, :].T)

    def predict_all(self):
        return self.user_vectors.dot(self.item_vectors.T)
