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
import math

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_data', type=str,
                        help='example data filepath')
    parser.add_argument('test_data', type=str,
                        help='example data filepath')
    args = parser.parse_args()

    training_data = get_data(args.learning_data)
    print('learning data collected')
    # print(training_data)

    scaled_training_data = scale_features(training_data)
    print('training data scaled')
    # print(scaled_training_data)

    learning_df, cv_df = process_training_data(scaled_training_data)
    print('learning and CV data sets organised')
    # print(learning_df)
    # print(cv_df)

    feature_df = fill_feature_df(learning_df)
    print('feature df created and filled')
    # print(feature_df)

    learn_epsillon(feature_df, cv_df)

def get_data(filename):
    return pd.read_csv(filename)

def learn_epsillon(feature_df, cv_df):
    feature_df_labels = (list(feature_df.index))
    print(feature_df_labels)
    eps = 0

    for index, row in cv_df.iterrows():
        print(index)
        eps_curr = 1
        for item in feature_df_labels:

            # print(row[item])
            # print(feature_df.loc[item, 'mean'])
            # print(feature_df.loc[item, 'SD'])

            # print((-(((row[item]-feature_df.loc[item, 'mean'])**2)/(2*feature_df.loc[item, 'SD']))))

            eps_curr = eps_curr * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[item,'SD'])))*(math.exp(-((row[item]-feature_df.loc[item,'mean'])**2)/(2*(feature_df.loc[item,'SD'])))))
            # print((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[item,'SD'])))*(math.exp(-((row[item]-feature_df.loc[item,'mean'])**2)/(2*(feature_df.loc[item,'SD'])))))

        print(eps_curr)

def scale_features(training_data):
    '''scales features using the min-max method, drops the class column for
    convenience'''

    scaled_training_data = training_data.drop('class', 1)
    scaled_training_data = scaled_training_data - scaled_training_data.min()
    scaled_training_data = scaled_training_data / scaled_training_data.max()
    scaled_training_data['class'] = training_data['class'].values
    return scaled_training_data

def fill_feature_df(learning_df):
    feature_df_labels = (list(learning_df.columns))
    feature_df_labels.pop()

    feature_df = pd.DataFrame(index=feature_df_labels, columns=['mean', 'SD'])

    for column in feature_df_labels:
        feature_df.loc[column, 'mean'] = learning_df[column].mean()
        feature_df.loc[column, 'SD'] = statistics.stdev(learning_df[column])
    return feature_df

def process_training_data(scaled_training_data):
    '''
    takes the data created by gen_data.py and replaces the classes with
    anomalous or non-anomalous
    '''
    feature_df_labels = (list(scaled_training_data.columns))
    learning_df = pd.DataFrame(columns=feature_df_labels)
    cv_df = pd.DataFrame(columns=feature_df_labels)

    for i in range(len(scaled_training_data)):
        if scaled_training_data.loc[i, 'class'] == 'clean':
            scaled_training_data.loc[i, 'class'] = '0'
        else:
            scaled_training_data.loc[i, 'class'] = '1'
        if i % 5 == 0:
            cv_df = cv_df.append(scaled_training_data.loc[i], ignore_index=True)
        else:
            learning_df = learning_df.append(scaled_training_data.loc[i], ignore_index=True)

    return learning_df, cv_df

if __name__ == "__main__":
    main()
