import preprocess
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"


class Clustering:
    def __init__(self, method):
        self.data = preprocess.readCsv(dataPath)
        self.encodedData = preprocess.preprocessData(self.data)
        self.X_train = self.encodedData[0]
        self.X_test = self.encodedData[1]
        self.y_train = self.encodedData[2]
        self.y_test = self.encodedData[3]
        self.method = method

    def convert_cluster_to_letters(self, nparray):
        npList = list(nparray)
        for i in range(len(npList)):
            if npList[i] == 0:
                npList[i] = "a"
            if npList[i] == 1:
                npList[i] = "b"
            if npList[i] == 2:
                npList[i] = "c"
            if npList[i] == 3:
                npList[i] = "d"
            if npList[i] == 4:
                npList[i] = "e"
            if npList[i] == 5:
                npList[i] = "f"
            if npList[i] == 6:
                npList[i] = "g"
            if npList[i] == 7:
                npList[i] = "h"
            if npList[i] == 8:
                npList[i] = "i"
        return npList

    def K_means(self):
        KMeans_model = KMeans(n_clusters=9)
        KMeans_model.fit(self.X_train)
        predicted_y = KMeans_model.predict(self.X_test)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        print(df)
        dummyData = pd.get_dummies(df)
        print(dummyData)
        dummyData.hist()
        plt.show()

        # print(predicted_y)
        # counterList = []
        # yList = list(predicted_y)
        # for i in range(8):
        #     counterList.append([i, yList.count(i)])
        # clusterDict = {}
        # for element in counterList:
        #     numList = []
        #     for i in range(element[1]):
        #         numList.append(element[0])
        #     clusterDict['cluster ' + str(element[0])] = numList
        # df = pd.DataFrame(clusterDict)
        # print(df)
        # df.hist()
        # plt.show()
        # plt.pie(np.array(counterList))
        # plt.show()


if __name__ == '__main__':
    km = Clustering('kmeans')
    km.K_means()
