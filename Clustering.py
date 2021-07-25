import preprocess
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import DimationReduction
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"


class Clustering:

    def __init__(self, method, dataPath):
        '''

        :param method:
        :param dataPath:
        '''
        self.dataPath = dataPath
        self.data = preprocess.readCsv(self.dataPath)
        self.encodedData = preprocess.preprocessData(self.data)
        self.X = self.encodedData[4]
        self.y = self.encodedData[5]
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

    def gmm(self, data):
        gmm_model = GaussianMixture(n_components=9)
        predicted_y = gmm_model.fit_predict(data)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        return df, predicted_y

    def K_means(self, data):
        KMeans_model = KMeans(n_clusters=9)
        predicted_y = KMeans_model.fit_predict(data)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        return df, predicted_y
        # print(df)
        # dummyData = pd.get_dummies(df)
        # print(dummyData)
        # dummyData.hist()
        # plt.show()

    def Hierarchical_clustering(self, data):
        hierarchical_model = AgglomerativeClustering(n_clusters=9)
        predicted_y = hierarchical_model.fit_predict(data)
        # self.histogramPlotter(predicted_y)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        return df, predicted_y

    def histogramPlotter(self, predicted_y):
        print(predicted_y)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        print(df)
        dummyData = pd.get_dummies(df)
        print(dummyData)
        dummyData.hist()
        plt.show()

    def plotClustering(self):
        dr = DimationReduction.DimantionReduction()
        dr.reduceDimentionForPlot()
        if self.method == "K_means":
            clusters = self.K_means(dr.reduced_X_for_plot)[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('k.html')
            fig.show()
        if self.method == "Hierarchical":
            clusters = self.Hierarchical_clustering(dr.reduced_X_for_plot)[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('h.html')
            fig.show()
        if self.method == 'gmm':
            clusters = self.gmm(dr.reduced_X_for_plot)[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('g.html')
            fig.show()

    def calc_silhouette_score(self):
        if self.method == 'gmm':
            silhouette_avg = silhouette_score(self.X, self.gmm(self.X)[1])
            sample_silhouette_values = silhouette_samples(self.X, self.gmm(self.X)[1])
            print(self.method + ' silhouette samples: ', sample_silhouette_values)
            print(self.method + "silhouette score: ", silhouette_avg)
            return silhouette_avg
        if self.method == 'K_means':
            silhouette_avg = silhouette_score(self.X, self.K_means(self.X)[1])
            print(self.method + "silhouette score: ", silhouette_avg)
            return silhouette_avg
        if self.method == 'Hierarchical':
            silhouette_avg = silhouette_score(self.X, self.Hierarchical_clustering(self.X)[1])
            print(self.method + "silhouette score: ", silhouette_avg)
            return silhouette_avg

if __name__ == '__main__':
    dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"
    km = Clustering('gmm', dataPath)
    km.calc_silhouette_score()
