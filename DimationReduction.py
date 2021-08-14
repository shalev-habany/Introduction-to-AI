import preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import numpy as np

dataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv"
reducedDataPath = r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\reduced_data.csv"


class DimantionReduction:
    def __init__(self, n_components=28):
        self.data = preprocess.readCsv(dataPath)
        self.encodedData = preprocess.preprocessData(self.data)
        self.X_train = self.encodedData[0]
        self.X_test = self.encodedData[1]
        self.y_train = self.encodedData[2]
        self.y_test = self.encodedData[3]
        self.X = self.encodedData[4]
        self.y = self.encodedData[5]
        self.n_components = n_components
        self.reduced_X = None
        self.reduced_X_for_plot = None

    def reduceDimension(self):
        # normalized_X = StandardScaler().fit_transform(self.X)
        pca = PCA(n_components=self.n_components)
        principalComponents = pca.fit_transform(self.X)
        column_names = []
        for i in range(self.n_components):
            column_names.append('principal component' + str(i))
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=column_names)
        self.reduced_X = principalDf
        print(self.reduced_X)
        # self.reduced_X.to_csv(reducedDataPath)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of features')
        plt.ylabel('explained variance')
        plt.show()

    def reduceDimensionForPlot(self):
        # normalized_X = StandardScaler().fit_transform(self.X)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(self.X)
        # print(pca.explained_variance_ratio_)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        self.reduced_X_for_plot = principalDf

    def ICA_reduceDimentionForPlot(self):
        # normalized_X = StandardScaler().fit_transform(self.X)
        ica = FastICA(n_components=2)
        principalComponents = ica.fit_transform(self.X)
        # print(pca.explained_variance_ratio_)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        self.reduced_X_for_plot = principalDf

    def plotReducedData(self):
        new_df = pd.concat([self.reduced_X_for_plot, self.data[['odor']]], axis=1)
        print(new_df)
        fig = px.scatter(new_df, x='principal component 1', y='principal component 2', color="odor")
        fig.write_html('labels_figure.html')
        fig.show()
        # df = self.reduced_X
        # df.plot.scatter(x='principal component 1', y='principal component 2')
        # # OR (with pandas 0.13 and up)
        # plt.show(block=True)


# if __name__ == '__main__':
#     # 28 saves 93% explained variance of the data
#     dr = DimantionReduction(101)
#     # dr.ICA_reduceDimentionForPlot()
#     # dr.plotReducedData()
#     dr.reduceDimension()