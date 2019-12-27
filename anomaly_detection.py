'''
Written by Ben McCoy, Dec 2019

This script will take a learning profile and determine the mean and standard
deviation for each time period of the data and store it in a pandas DF.

Then, using an anomaly detection algorithm, take new data and determine if it
anomalous.
'''

import argparse
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_data', type=str,
                        help='example data filepath')

    args = parser.parse_args()

    learning_data = get_data(args.learning_data)
    feature_df_labels = (list(learning_data.columns))

    list_dict = {}

    for column in feature_df_labels:
        list_dict[column] = []

    for index, row in learning_data.iterrows():
        for column in feature_df_labels:
            list_dict[column].append(row[column])

    mean_dict = {}
    for item in list_dict:
        try:
            mean_dict[item] = sum(list_dict[item])/len(list_dict[item])
        except:
            print(list_dict[item])

    print(mean_dict)

def get_data(filename):
    return pd.read_csv(filename)

if __name__ == "__main__":
    main()
