import math
import random
from os import altsep

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.linalg import solve
from src.algorithms.matrix_factorization import MatrixFactorizationModel
from src.algorithms.metrics.mse import mse_sparse
from src.algorithms.nn.models import MLP_model, neuMF_model
from src.processing.get_rating_matrix import get_sparse_matrix, split_matrix
from tensorflow.keras import Input, Model, layers, models


def get_train_validation_indices(rating_matrix, validation_ratio=0.1, test_ratio=0.1):
    y_indices, x_indices = rating_matrix.nonzero()
    num_non_zero = len(y_indices)
    indices = [i for i in range(num_non_zero)]
    random.shuffle(indices)
    train_indices = indices[:int(num_non_zero * (1 - test_ratio - validation_ratio))]
    train_indices = [(y_indices[i], x_indices[i]) for i in train_indices]
    validation_indices = indices[int(num_non_zero * (1 - test_ratio - validation_ratio)): int(num_non_zero * (1 - test_ratio))]
    validation_indices = [(y_indices[i], x_indices[i]) for i in validation_indices]
    test_indices = indices[int(num_non_zero * (1 - test_ratio)):]
    test_indices = [(y_indices[i], x_indices[i]) for i in test_indices]
    return train_indices, validation_indices, test_indices

def get_data_from_indices(rating_matrix, indices):
    user_inputs = np.zeros((len(indices), rating_matrix.shape[0] ))
    item_inputs = np.zeros((len(indices), rating_matrix.shape[1] ))
    targets = np.zeros((len(indices), 1))
    for index, (y_index, x_index) in enumerate(indices):
        user_inputs[index, :] = rating_matrix[y_index, :]
        user_inputs[index, x_index] = 0
        item_inputs[index, :] = rating_matrix[:, x_index]
        item_inputs[index, y_index] = 0
        targets[index, : ] = rating_matrix[y_index, x_index]
    return user_inputs, item_inputs, targets



def data_generator(rating_matrix, indices, batch_size=8):
    current_start_index = 0
    while 1:
        idx = np.random.randint(len(indices), size=batch_size)
        batch_indices = [indices[id] for id in idx]

        user_inputs, item_inputs, targets = get_data_from_indices(rating_matrix, batch_indices)

        yield [user_inputs, item_inputs], targets

def train(model, rating_matrix, validation_ratio=0.1):
    train_indices, validation_indices, _ = get_train_validation_indices(rating_matrix, validation_ratio=0.1, test_ratio=0)
    batch_size = 16
    train_generator = data_generator(rating_matrix, train_indices, batch_size)
    validation_generator = data_generator(rating_matrix, validation_indices, batch_size)
    history = model.fit(
        train_generator,
        epochs=100,
        steps_per_epoch=math.ceil(len(train_indices) / batch_size),
        validation_data=validation_generator,
        validation_steps=math.ceil(len(validation_indices) / batch_size)
    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model

if __name__ == '__main__':

    rating_matrix = get_sparse_matrix(1000, 1000)
    # model = MLP_model(1000)
    model = neuMF_model(1000)
    model = train(model, rating_matrix, 0.1)


