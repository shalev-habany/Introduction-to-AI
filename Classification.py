import statistics

import sklearn.metrics
from sklearn.svm import SVC
import preprocess
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"


class Classification:
    def __init__(self, method, kernal, num_of_trees):
        self.data = preprocess.readCsv(dataPath)
        self.encodedData = preprocess.preprocessData(self.data)
        self.X_train = self.encodedData[0]
        self.X_test = self.encodedData[1]
        self.y_train = self.encodedData[2]
        self.y_test = self.encodedData[3]
        self.method = method
        self.labels = preprocess.column_name_list
        self.kernal = kernal
        self.num_of_trees = num_of_trees

    def randomForest(self):
        accuracyList = []
        for feature in self.y_train.columns.values:
            rf_model = RandomForestClassifier(n_estimators=self.num_of_trees)
            rf_model.fit(self.X_train, self.y_train[feature])
            print(feature + " accuracy score: ", accuracy_score(self.y_test[feature], rf_model.predict(self.X_test)))
            accuracyList.append(accuracy_score(self.y_test[feature], rf_model.predict(self.X_test)))
            print("prediction", rf_model.predict(self.X_test))
        print("average accuracy: ", statistics.mean(accuracyList))
        return statistics.mean(accuracyList)
        #     plotList.append([i, accuracy_score(self.y_test, rf_model.predict(self.X_test))])
        # plt.plot([item[0] for item in plotList], [item[1] for item in plotList])
        # plt.title('accuracy score')
        # plt.ylabel('accuracy')
        # plt.xlabel('number of trees')
        # plt.legend("accuracy_score")
        # plt.show()

    def kernalSVM(self):
        accuracyList = []
        for feature in self.y_train.columns.values:
            SVM_model = SVC(C=1, kernel=self.kernal)
            SVM_model.fit(self.X_train, self.y_train[feature])
            print(feature + " accuracy score: ", accuracy_score(self.y_test[feature], SVM_model.predict(self.X_test)))
            accuracyList.append(accuracy_score(self.y_test[feature], SVM_model.predict(self.X_test)))
        print("average accuracy: ", statistics.mean(accuracyList))

    def nnTrain(self):
        accuracyList = []
        print("all set")
        nnModel = tf.keras.models.Sequential()
        nnModel.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        nnModel.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        nnModel.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        nnModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        dct = {}
        for feature in self.y_train.columns.values:
            relu_history = nnModel.fit(self.X_train, self.y_train[feature], epochs=50)
            print(relu_history)
            accuracyList.append(nnModel.evaluate(self.X_test, self.y_test[feature])[1])
            # dct[feature + ' y_test'] = self.y_test[feature]
            # dct[feature + 'predicted'] = nnModel.predict(self.X_test)
            print(feature + " accuracy score: ", nnModel.evaluate(self.X_test, self.y_test[feature])[1])
        print("average accuracy: ", statistics.mean(accuracyList))
        # Keras = pd.DataFrame(dct)
        # Keras.to_csv("results.csv")


if __name__ == '__main__':
    nn = Classification("s", 'rbf', 10)
    # nn.nnTrain()
    # plotList = []
    # for i in range(1, 20):
    #     nn = Classification("s", 'linear', 20)
    #     plotList.append([i, nn.randomForest()])
    # plt.plot([item[0] for item in plotList], [item[1] for item in plotList])
    # plt.title('accuracy score')
    # plt.ylabel('accuracy')
    # plt.xlabel('number of trees')
    # plt.legend("accuracy_score")
    # plt.show()
    nn.nnTrain()
    # nn.randomForest()
    # relu_history = nn.nnTrain()
    # print(relu_history.history.keys())
    # plt.figure(figsize=(10, 5))
    # plt.plot(relu_history.history['categorical_crossentropy'], label='relu')
    # plt.title('categorical_crossentropy')
    # plt.ylabel('categorical_crossentropy')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.show()
