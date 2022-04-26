import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models


def MLP_model(user_size=1000, item_size=1000):
    input1 = Input(shape=(user_size,), dtype=tf.float32, name='user_vector')
    input2 = Input(shape=(item_size,), dtype=tf.float32, name='item_vector')
    concat = layers.concatenate(axis=1, inputs=[input1, input2])
    concat = layers.Dropout(0.25)(concat)
    d1 = layers.Dense(64, activation="relu")(concat)
    d2 = layers.Dense(32, activation="relu")(d1)
    d3 = layers.Dense(16, activation="relu")(d2)
    do = layers.Dropout(0.25)(d3)
    y = layers.Dense(1)(do)

    model = Model(inputs=[input1, input2], outputs=[y])
    model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])
    return model

def neuMF_model(user_size=1000, item_size=1000):
    input1 = Input(shape=(user_size,), dtype=tf.float32, name='user_vector')
    input2 = Input(shape=(item_size,), dtype=tf.float32, name='item_vector')
    concat = layers.concatenate(axis=1, inputs=[input1, input2])
    concat = layers.Dropout(0.25)(concat)

    d1 = layers.Dense(64, activation="relu")(concat)
    d1 = layers.Dropout(0.25)(d1)
    d2 = layers.Dense(32, activation="relu")(d1)
    d3 = layers.Dense(16, activation="relu")(d2)
    do = layers.Dropout(0.25)(d3)

    gmf = layers.Multiply()([input1, input2])
    gmf = layers.Dropout(0.25)(gmf)
    do = layers.concatenate(axis=1, inputs=[do, gmf])

    y = layers.Dense(1)(do)

    model = Model(inputs=[input1, input2], outputs=[y])
    model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == '__main__':
    model = MLP_model(10000, 10000)
    model.summary()
