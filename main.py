import Classification
import Clustering


class Main:
    def __init__(self, data, approach, method):
        self.data = data
        self.approach = approach
        self.method = method

    def createClassifier(self):
        if self.data == "reduced":
            return Classification.Classification(dimensionreduction="yes")
        if self.data == "deleted":
            return Classification.Classification(deletedDataRead="yes")
        return Classification.Classification()

    def createClustering(self):
        if self.data == "reduced":
            return Clustering.Clustering(dimensionReduction="yes", method=self.method)
        return Clustering.Clustering(method=self.method)


if __name__ == '__main__':
    approach = input("choose approach (supervised/unsupervised): ")
    data = input("choose data(regular/reduced/deleted): ")

    if approach == "supervised":
        method = input("choose method (neural networks/SVM/random forest): ")
        m = Main(data, approach, method)
        classifier = m.createClassifier()
        classifier.set_data()
        if method == "neural networks":
            classifier.nnTrain()
        elif method == "random forest":
            classifier.randomForest()
        else:
            classifier.kernalSVM()
    else:
        method = input("choose method (K_means/Hierarchical/gmm): ")
        m = Main(data, approach, method)
        clustering = m.createClustering()
        clustering.set_data()
        clustering.calc_silhouette_score()
        with_figure = input("want figure?(yes/no): ")
        if with_figure == "yes":
            clustering.plotClustering()

