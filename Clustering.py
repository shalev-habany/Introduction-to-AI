import preprocess
import DimationReduction
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, fowlkes_mallows_score

dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"
reducedDataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\reduced_data.csv"


class Clustering:

    def __init__(self, method, dimensionReduction="no"):
        '''

        :param method:
        :param dataPath:
        '''
        self.dimensionReduction = dimensionReduction
        self.reducedData = pd.read_csv(reducedDataPath)
        self.data = preprocess.readCsv(dataPath)
        self.reducedEncodedData = preprocess.preprocessReducedData(self.reducedData, self.data)
        self.encodedData = preprocess.preprocessData(self.data)
        self.X = None
        self.y = None
        self.method = method

    def set_data(self):
        if self.dimensionReduction == "yes":
            self.X = self.reducedEncodedData[4]
            self.y = self.reducedEncodedData[5]
        if self.dimensionReduction == "no":
            self.X = self.encodedData[4]
            self.y = self.encodedData[5]

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

    def gmm(self, n_clusters=9):
        gmm_model = GaussianMixture(n_components=n_clusters)
        predicted_y = gmm_model.fit_predict(self.X)
        yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(yList, columns=['cluster'])
        return df, predicted_y

    def K_means(self, n_clusters=9):
        KMeans_model = KMeans(n_clusters=n_clusters)
        predicted_y = KMeans_model.fit_predict(self.X)
        # yList = self.convert_cluster_to_letters(predicted_y)
        df = pd.DataFrame(predicted_y, columns=['cluster'])
        # for i in range(1, 9, 1):
        #     cluster_a = df.loc[df['cluster'] == i]
        #     print(cluster_a)
        #     cluster_a_with_labels = pd.concat([cluster_a, self.y], axis=1, join="inner")
        #     print(cluster_a_with_labels)
        #     df_for_histo = cluster_a_with_labels.drop(["cluster"], axis=1)
        #     print(df_for_histo)
        #     df_for_histo.hist()
        #     plt.show()
        # dummyData = pd.get_dummies(df)
        # print(dummyData)
        return df, predicted_y
        # dummyData.hist()
        # plt.show()

    def Hierarchical_clustering(self, n_clusters=9):
        hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters)
        predicted_y = hierarchical_model.fit_predict(self.X)
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
        dr.reduceDimensionForPlot()
        if self.method == "K_means":
            self.X = dr.reduced_X_for_plot
            clusters = self.K_means()[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('k.html')
            fig.show()
        if self.method == "Hierarchical":
            self.X = dr.reduced_X_for_plot
            clusters = self.Hierarchical_clustering()[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('h.html')
            fig.show()
        if self.method == 'gmm':
            self.X = dr.reduced_X_for_plot
            clusters = self.gmm()[0]
            print(clusters)
            new_df = pd.concat([dr.reduced_X_for_plot, clusters[['cluster']]], axis=1)
            print(new_df)
            fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="cluster")
            fig.write_html('g.html')
            fig.show()

    def calc_silhouette_score(self):
        if self.method == 'gmm':
            silhouette_avg = silhouette_score(self.X, self.gmm()[1])
            # sample_silhouette_values = silhouette_samples(self.X, self.gmm(self.X)[1])
            # print(self.method + ' silhouette samples: ', sample_silhouette_values)
            print(self.method + " silhouette score: ", silhouette_avg)
            a = np.array(self.y)
            labelEncoded_y = np.where(a == 1)[1]
            fowlkes_mallows_avg = fowlkes_mallows_score(labelEncoded_y, self.gmm()[1])
            print(self.method + " fowlkes mallows score: ", fowlkes_mallows_avg)
            return silhouette_avg
        if self.method == 'K_means':
            silhouette_avg = silhouette_score(self.X, self.K_means()[1])
            print(self.method + "silhouette score: ", silhouette_avg)
            a = np.array(self.y)
            labelEncoded_y = np.where(a == 1)[1]
            fowlkes_mallows_avg = fowlkes_mallows_score(labelEncoded_y, self.K_means()[1])
            print(self.method + " fowlkes mallows score: ", fowlkes_mallows_avg)
            return silhouette_avg, fowlkes_mallows_avg
        if self.method == 'Hierarchical':
            silhouette_avg = silhouette_score(self.X, self.Hierarchical_clustering()[1])
            print(self.method + "silhouette score: ", silhouette_avg)
            a = np.array(self.y)
            labelEncoded_y = np.where(a == 1)[1]
            fowlkes_mallows_avg = fowlkes_mallows_score(labelEncoded_y, self.Hierarchical_clustering()[1])
            print(self.method + " fowlkes mallows score: ", fowlkes_mallows_avg)
            return silhouette_avg


# if __name__ == '__main__':
#     km = Clustering('gmm')
#     km.set_data()
#     km.plotClustering()
