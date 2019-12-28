'''
Written by Ben McCoy, Dec 2019

This script will take a learning profile and determine the mean and standard
deviation for each time period of the data and store it in a pandas DF.

Then, using an anomaly detection algorithm, take new data and determine if it
anomalous.
'''

import argparse
import pandas as pd
import statistics

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_data', type=str,
                        help='example data filepath')

    args = parser.parse_args()

    learning_data = get_data(args.learning_data)
    feature_df_labels = (list(learning_data.columns))
    feature_df_labels.pop()

    feature_df = pd.DataFrame(index=feature_df_labels, columns=['mean', 'SD'])

    for column in feature_df_labels:
        feature_df.loc[column, 'mean'] = learning_data[column].mean()
        feature_df.loc[column, 'SD'] = statistics.stdev(learning_data[column])

    print(feature_df)

def get_data(filename):
    return pd.read_csv(filename)

if __name__ == "__main__":
    main()
