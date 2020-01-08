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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('learning_data', type=str,
                        help='example data filepath')
    parser.add_argument('test_data', type=str,
                        help='example data filepath')
    args = parser.parse_args()

    training_data = get_data(args.learning_data)
    training_data.replace(0, 0.01, inplace=True)
    print('learning data collected')
    # print(training_data)
    # training_data.mean().plot()

    scaled_training_data = scale_features(training_data)
    print('training data scaled')
    # print(scaled_training_data)

    learning_df, cv_df, test_df = process_training_data(scaled_training_data)
    # learning_df, cv_df, test_df = process_training_data(training_data)
    print('learning and CV data sets organised')
    # print(learning_df)
    # print(cv_df)
    # print(test_df)

    feature_df = fill_feature_df(learning_df)
    print('feature df created and filled')
    feature_df['mean+sd'] = feature_df['mean'] + feature_df['SD']
    feature_df['mean-sd'] = feature_df['mean'] - feature_df['SD']
    # feature_df.plot()
    # plt.show()
    # print(feature_df)

    eps_min = learn_epsillon(feature_df, learning_df)
    print('minimum episllon found in training set of ', eps_min)
    # print(eps_min)

    eps_min, incorrect_eps, incorrect_anom = validate_epsillon(eps_min, feature_df, cv_df)
    print('epsillon cross validated')
    if len(incorrect_eps) > 0:
        print('A new lower value of eps_min was found ', eps_min)
        print('The data that updated eps_min was:')
        print(incorrect_eps)
        incorrect_eps = incorrect_eps.drop(columns=['class', 'old_eps', 'new_eps']).T
        incorrect_eps.plot()
    plt.show()

    if len(incorrect_anom) > 0:
        print('###### WARNING ######')
        print('The following anomalies were not detected by epsillon')
        print(incorrect_anom)
        incorrect_anom.drop(columns=['class', 'old_eps', 'new_eps'], inplace=True).T
        incorrect_anom.plot()

def get_data(filename):
    return pd.read_csv(filename)

def validate_epsillon(eps_min, feature_df, cv_df):

    # init the anom and non_anon dfs, as well as the incorrect_df and feature list
    feature_df_labels = (list(feature_df.index))
    incorrect_eps = pd.DataFrame(columns=feature_df_labels)
    incorrect_anom = pd.DataFrame(columns=feature_df_labels)
    non_anom_df = cv_df[cv_df['class'] == 'clean']
    anom_df = cv_df[cv_df['class'] != 'clean']

    # iterate through non anomalous CV data
    for index, row in non_anom_df.iterrows():
        eps = 1
        for label in feature_df_labels:
            eps = eps * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[label,'SD'])))*(math.exp(-((row[label]-feature_df.loc[label,'mean'])**2)/(2*(feature_df.loc[label,'SD'])))))
        # If a lower eps value is found, it is updated and the row and old value stored
        if eps < eps_min:
            incorrect_eps = incorrect_eps.append(non_anom_df.loc[index, :], ignore_index=True)
            # incorrect_eps.iloc[-1, incorrect_eps.columns.get_loc('old_eps')] = eps_min
            # incorrect_eps['old_eps'] = eps_min
            # incorrect_eps['new_eps'] = eps
            eps_min = eps

    # iterate through anomalous CV data
    for index, row in anom_df.iterrows():
        eps = 1
        for label in feature_df_labels:
            eps = eps * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[label,'SD'])))*(math.exp(-((row[label]-feature_df.loc[label,'mean'])**2)/(2*(feature_df.loc[label,'SD'])))))
        # If an anopmaly slips through eps, the row is stored and a message is printed
        if eps > eps_min:
            incorrect_anom = incorrect_anom.append(anom_df.loc[index, :], ignore_index=True)
            print('an eps value of ', eps, ' was found for anomalous data, the current eps_min value is ', eps_min)

    return eps_min, incorrect_eps, incorrect_anom

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
    convenience then picks it up again'''

    scaled_training_data = training_data.drop('class', 1)
    scaler = StandardScaler()
    scaled_training_data = pd.DataFrame(scaler.fit_transform(scaled_training_data), columns=scaled_training_data.columns)
    print(scaler)
    scaled_training_data['class'] = training_data['class'].values
    return scaled_training_data

def fill_feature_df(learning_df):
    # learning_df.mean().plot()
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

    non_anom_df = scaled_training_data[scaled_training_data['class'] == 'clean']
    anom_df = scaled_training_data[scaled_training_data['class'] != 'clean']

    cv_df, test_df = train_test_split(anom_df, test_size=0.5)
    learning_df, temp_df = train_test_split(non_anom_df, test_size=0.2)
    temp_cv_df, temp_test_df = train_test_split(temp_df, test_size=0.5)
    cv_df = cv_df.append(temp_cv_df, ignore_index=True)
    test_df = test_df.append(temp_test_df, ignore_index=True)

    return learning_df, cv_df, test_df

if __name__ == "__main__":
    main()
