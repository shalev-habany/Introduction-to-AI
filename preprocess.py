import pandas as pd
import numpy as np


def preprocessData(csvFilePath):
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
    df = pd.read_csv(csvFilePath, names=column_name_list, dtype=str)
    return df

    

if __name__ == '__main__':
    df = preprocessData(r"C:\Users\shalev\Desktop\Introduction_to_AI\Introduction-to-AI\Data\mushrooms_data.csv")
    classlist = df['class'].values
    print(classlist)