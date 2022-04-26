# Alternating Least Square


from os import altsep

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from src.algorithms.matrix_factorization import MatrixFactorizationModel
from src.algorithms.metrics.mse import mse_sparse
from src.processing.get_rating_matrix import get_sparse_matrix, split_matrix


class AlternatingLeastSquare(MatrixFactorizationModel):
    """
    Alternating Least Square Algorithm to do explicit matrix factorization
    """

    def als_step_user(self, ratings, user_vectors, item_vectors, user_regularizer = 0):
        YTY = item_vectors.T.dot(item_vectors)
        lambdaI = np.eye(YTY.shape[0]) * user_regularizer
        for u in range(user_vectors.shape[0]):
            user_vectors[u, :] = solve((YTY + lambdaI), 
                                             ratings[u, :].dot(item_vectors))
        return user_vectors

    def als_step_item(self, ratings, user_vectors, item_vectors, item_regularizer = 0):
        XTX = user_vectors.T.dot(user_vectors)
        lambdaI = np.eye(XTX.shape[0]) * item_regularizer
        for i in range(item_vectors.shape[0]):
            item_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].dot(user_vectors))
        return item_vectors

    def train(self, rating_matrix, user_vectors = None, item_vectors = None, user_reg = 0, item_reg = 0, num_iterations = 10, validation_matrix=None):
        if not self.check_input(rating_matrix, user_vectors, item_vectors):
            return self.user_vectors, self.item_vectors
        if validation_matrix is None or not self.check_input(validation_matrix):
            return self.user_vectors, self.item_vectors
        if isinstance(user_vectors, np.ndarray):
            self.user_vectors = user_vectors
        if isinstance(item_vectors, np.ndarray):
            self.item_vectors = item_vectors

        mse_trains = []
        mse_validations = []
        for i in range(num_iterations):
            self.user_vectors = self.als_step_user(rating_matrix, self.user_vectors, self.item_vectors, user_regularizer=user_reg)
            self.item_vectors = self.als_step_item(rating_matrix, self.user_vectors, self.item_vectors, item_regularizer=item_reg)

            prediction = self.predict_all()
            mse_trains.append(mse_sparse(rating_matrix, prediction))
            mse_validations.append(mse_sparse(validation_matrix, prediction))
            print(f'MSE train: {mse_trains[i]}, validation: {mse_validations[i]}')
        plt.plot(range(num_iterations), mse_trains)
        plt.plot(range(num_iterations), mse_validations)
        plt.show()
if __name__ == '__main__':
    model = AlternatingLeastSquare(10000, 10000, 5)
    rating_matrix = get_sparse_matrix(10000, 10000)
    rating_matrix, validation_matrix = split_matrix(rating_matrix, 0.1)
    
    model.train(rating_matrix, num_iterations=100, validation_matrix=validation_matrix, user_reg=20, item_reg=20)
    print(model.predict_all())

        

        

        