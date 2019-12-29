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
    parser.add_argument('cv_data', type=str,
                        help='example data filepath')
    args = parser.parse_args()

    learning_data = get_data(args.learning_data)
    print('learning data collected')
    # print(learning_data)

    feature_df = fill_feature_df(learning_data)
    print('feature df created and filled')
    # print(feature_df)

    cv_data = get_data(args.cv_data)
    print('collected cross validation data')
    # print(cv_data)

    process_data(cv_data)

def get_data(filename):
    return pd.read_csv(filename)

def fill_feature_df(learning_data):
    feature_df_labels = (list(learning_data.columns))
    feature_df_labels.pop()

    feature_df = pd.DataFrame(index=feature_df_labels, columns=['mean', 'SD'])

    for column in feature_df_labels:
        feature_df.loc[column, 'mean'] = learning_data[column].mean()
        feature_df.loc[column, 'SD'] = statistics.stdev(learning_data[column])
    return feature_df

def process_data(df):
    '''
    takes the data created by gen_data.py and replaces the classes with
    anomalous or non-anomalous
    '''

    for i in range(len(df)):
        if df.loc[i, 'class'] == 'clean':
            df.loc[i, 'class'] = '0'
        else:
            df.loc[i, 'class'] = '1'

    print(df)

if __name__ == "__main__":
    main()
