import pandas as pd
import numpy as np

column_name_list = ["class", 'cap-shape',
                    'cap-surface',
                    'cap-color',
                    'bruises?',
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
    converted_df = pd.get_dummies(df)
    return converted_df


if __name__ == '__main__':
    df = readCsv(r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv")
    print(preprocessData(df))
