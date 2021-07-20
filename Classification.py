import preprocess
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy


dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"


class Classification:
    def __init__(self, method):
        self.data = preprocess.readCsv(dataPath)
        self.method = method
        self.labels = preprocess.column_name_list

    def 

    def nnTrain(self):
        print(self.data)
        X = self.data.drop(['odor'], axis=1)
        y = pd.DataFrame(self.data['odor'])
        X_encoded = preprocess.preprocessData(X)
        y_encoded = preprocess.preprocessData(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, train_size=0.66, test_size=0.33, random_state=42)
        print("all set")
        nnModel = tf.keras.models.Sequential()
        nnModel.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        nnModel.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        nnModel.add(tf.keras.layers.Dense(9, activation=tf.nn.log_softmax))
        nnModel.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=[tf.keras.metrics.categorical_crossentropy])
        relu_history = nnModel.fit(X_train, y_train, epochs=50)
        print(relu_history)
        val_loss, val_mse = nnModel.evaluate(X_test, y_test)
        print(val_loss, val_mse)
        predicted_y_test = nnModel.predict(X_test)
        print(" test: ", CategoricalCrossentropy(y_test, predicted_y_test))
        return relu_history


if __name__ == '__main__':
    nn = Classification("nn")
    relu_history = nn.nnTrain()
    print(relu_history.history.keys())
    plt.figure(figsize=(10, 5))
    plt.plot(relu_history.history['categorical_crossentropy'], label='relu')
    plt.title('categorical_crossentropy')
    plt.ylabel('categorical_crossentropy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
