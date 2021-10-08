import statistics
from sklearn.svm import SVC
import preprocess
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataPath = r"C:\Users\shale\Desktop\Introduction-to-AI\Data\mushrooms_data.csv"
reducedDataPath = r"C:\Users\shale\Desktop\Introduction-to-AI\Data\reduced_data.csv"
deletedDataPath = r"C:\Users\shale\Desktop\Introduction-to-AI\Data\mushrooms_data_missing.csv"


class Classification:
    def __init__(self, kernal="linear", num_of_trees=10, dimensionreduction="no", deletedDataRead="no"):
        self.deletedDataRead = deletedDataRead
        self.dimensionreduction = dimensionreduction
        self.deletedData = preprocess.readCsv(deletedDataPath)
        self.reducedData = pd.read_csv(reducedDataPath)
        self.data = preprocess.readCsv(dataPath)
        self.encodedData = preprocess.preprocessData(self.data)
        self.reducedEncodedData = preprocess.preprocessReducedData(self.reducedData, self.data)
        self.deletedEncodedData = preprocess.preprocessDeletedData(self.deletedData)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = preprocess.column_name_list
        self.kernal = kernal
        self.num_of_trees = num_of_trees

    def set_data(self):
        if self.dimensionreduction == "yes":
            self.X_train = self.reducedEncodedData[0]
            self.X_test = self.reducedEncodedData[1]
            self.y_train = self.reducedEncodedData[2]
            self.y_test = self.reducedEncodedData[3]
        elif self.deletedDataRead == "yes":
            self.X_train = self.deletedEncodedData[0]
            self.X_test = self.deletedEncodedData[1]
            self.y_train = self.deletedEncodedData[2]
            self.y_test = self.deletedEncodedData[3]
        else:
            self.X_train = self.encodedData[0]
            self.X_test = self.encodedData[1]
            self.y_train = self.encodedData[2]
            self.y_test = self.encodedData[3]

    def randomForest(self):
        accuracyList = []
        for feature in self.y_train.columns.values:
            rf_model = RandomForestClassifier(n_estimators=self.num_of_trees)
            rf_model.fit(self.X_train, self.y_train[feature])
            # print(feature + " accuracy score: ", accuracy_score(self.y_test[feature], rf_model.predict(self.X_test)))
            accuracyList.append(accuracy_score(self.y_test[feature], rf_model.predict(self.X_test)))
            # print("prediction", rf_model.predict(self.X_test))
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
        if self.deletedDataRead == "yes":
            nnModel.add(tf.keras.layers.Dense(32, input_shape=(self.X_train.shape[1],), activation=tf.nn.relu))
            nnModel.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        if self.deletedDataRead == "no":
            nnModel.add(tf.keras.layers.Dense(128, input_shape=(self.X_train.shape[1],), activation=tf.nn.relu))
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


# if __name__ == '__main__':
#     for i in range(1, 20, 1):
#         print("num of trees: ", i)
#         nn = Classification('linear', i, deletedDataRead="yes")
#         nn.set_data()
#         nn.randomForest()
