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
from sklearn.model_selection import train_test_split

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

    learning_df, cv_df, test_df = process_training_data(scaled_training_data)
    print('learning and CV data sets organised')
    # print(learning_df)
    # print(cv_df)
    # print(test_df)

    feature_df = fill_feature_df(learning_df)
    print('feature df created and filled')
    # print(feature_df)

    eps_min = learn_epsillon(feature_df, learning_df)
    print('minimum episllon found in training set')
    # print(eps_min)

    validate_epsillon(eps_min, feature_df, test_df)

def get_data(filename):
    return pd.read_csv(filename)

def validate_epsillon(eps_min, feature_df, cv_df):

    feature_df_labels = (list(feature_df.index))
    incorrect_df = pd.DataFrame(columns=feature_df_labels)

    for index, row in cv_df.iterrows():
        eps_curr = 1
        expected_result = row['class']
        for item in feature_df_labels:
            eps_curr = eps_curr * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[item,'SD'])))*(math.exp(-((row[item]-feature_df.loc[item,'mean'])**2)/(2*(feature_df.loc[item,'SD'])))))
        # print(eps_curr)
        if eps_curr < eps_min and expected_result == '1':
            print('no issues')
        elif eps_curr >= eps_min and expected_result == '0':
            print('no issues')
        else:
            print('issues')

def learn_epsillon(feature_df, df):
    feature_df_labels = (list(feature_df.index))
    eps_min = math.inf

    for index, row in df.iterrows():
        eps_curr = 1
        for item in feature_df_labels:
            eps_curr = eps_curr * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[item,'SD'])))*(math.exp(-((row[item]-feature_df.loc[item,'mean'])**2)/(2*(feature_df.loc[item,'SD'])))))
        if eps_curr < eps_min:
            eps_min = eps_curr

    return eps_min

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
    anomalous_df = pd.DataFrame(columns=feature_df_labels)
    non_anomalous_df = pd.DataFrame(columns=feature_df_labels)

    for i in range(len(scaled_training_data)):
        if scaled_training_data.loc[i, 'class'] == 'clean':
            scaled_training_data.loc[i, 'class'] = '0'
            non_anomalous_df = non_anomalous_df.append(scaled_training_data.loc[i], ignore_index=True)
        else:
            scaled_training_data.loc[i, 'class'] = '1'
            anomalous_df = anomalous_df.append(scaled_training_data.loc[i], ignore_index=True)

    cv_df, test_df = train_test_split(anomalous_df, test_size=0.5)
    learning_df, temp_df = train_test_split(non_anomalous_df, test_size=0.2)
    temp_cv_df, temp_test_df = train_test_split(temp_df, test_size=0.5)
    cv_df = cv_df.append(temp_cv_df, ignore_index=True)
    test_df = test_df.append(temp_test_df, ignore_index=True)

    return learning_df, cv_df, test_df

if __name__ == "__main__":
    main()
