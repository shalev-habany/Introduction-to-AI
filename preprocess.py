import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

column_name_list = ["class", 'cap-shape',
                    'cap-surface',
                    'cap-color',
                    'bruises',
                    'odor',
                    'gill-attachment',
                    'gill-spacing',
                    'gill-size',
                    'gill-color',
                    'stalk-shape',
                    'stalk-surface-above-ring',
                    'stalk-surface-below-ring',
                    'stalk-color-above-ring',
                    'stalk-color-below-ring',
                    'veil-type',
                    'veil-color',
                    'ring-number',
                    'ring-type',
                    'spore-print-color',
                    'population',
                    'habitat']


def readCsv(csvFile):
    return pd.read_csv(csvFile, names=column_name_list)


def preprocessData(df):
    X = df.drop(['odor'], axis=1)
    y = pd.DataFrame(df['odor'])
    converted_X = pd.get_dummies(X)
    converted_y = pd.get_dummies(y)
    # converted_y.hist()
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(
        converted_X, converted_y, train_size=0.66, test_size=0.33, random_state=42)
    # y_train.hist()
    # plt.show()
    return [X_train, X_test, y_train, y_test, converted_X, converted_y]


def preprocessReducedData(reduced_X, originalData):
    # print(reduced_X)
    # print(list(reduced_X.columns))
    y = pd.DataFrame(originalData['odor'])
    converted_y = pd.get_dummies(y)
    # print(converted_y)
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_X, converted_y, train_size=0.66, test_size=0.33, random_state=42)
    return [X_train, X_test, y_train, y_test, reduced_X, converted_y]


def preprocessWithAD(converted_X):
    ad_model = IsolationForest(random_state=0)
    converted_X['outliers'] = pd.Series(ad_model.fit_predict(converted_X))
    converted_X['outliers'] = converted_X['outliers'].map({1: 0, -1: 1})
    print(converted_X['outliers'].value_counts())
    # for i in range(len(converted_X['outliers'])):
    #     if converted_X.loc[i].at['outliers']:
    #         converted_X = converted_X.drop(i)
    # # print(converted_X)
    # converted_X = converted_X.drop(columns=['outliers'], axis=1)
    # # print(converted_X)
    # return converted_X


def preprocessDeletedData(df):
    X = df.drop(['odor'], axis=1)
    y = pd.DataFrame(df['odor'])
    converted_X = pd.get_dummies(X)
    converted_y = pd.get_dummies(y)
    for column in converted_X.columns:
        if "-" in column:
            converted_X = converted_X.drop([column], axis=1)
    converted_y = converted_y.drop(['odor_-'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        converted_X, converted_y, train_size=0.66, test_size=0.33, random_state=42)
    return [X_train, X_test, y_train, y_test, converted_X, converted_y]



# if __name__ == '__main__':
#     df = readCsv(r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data_missing.csv")
#     preprocessDeletedData(df)
#     # data = preprocessData(df)[5]
#     # print(data.shape)
#
#     # print(data.shape[1])
#     # preprocessWithAD(data)
