'''
Written by Ben McCoy, Dec 2019

This script takes a csv of data where the columns include all of the x values,
as well as if the data is anomalous. Each row is a unique example that is used
to teach the model.

The data is then scaled using the sklearn StandardScaler to end up with
epsillon values that do not break the computer's memory. (for example, when
using 48 time periods with 25,000MW data points my epsillon value was on the
order of 1e-200 or so.)

Upon scaling the data, it is split into a learning, cross-validation and
testing dataset with the following breakdown of data:

learning: 80% of non-anomalous data
cross validation: 10% of non-anomalous data, 50% of anomalous data
testing: 10% of non-anomalous data, 50% of anomalous data

The data is split randomly using the sklearn train_test_split function.

The learning dataset is used to determine the minimum episllon value of non-
anomalous data which is then stored to be cross validated.

The CV data set is used to ensure that anomalies are caught, and if there is a
lower epsillon value, then the epsillon value is updated.

The test dataset is then used to determine how accurate the validated epsillon
value is.

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
    print('\n1. learning data collected')
    # print(training_data)

    scaled_training_data = scale_features(training_data)
    print('\n2. training data scaled')
    # print(scaled_training_data)

    learning_df, cv_df, test_df = process_training_data(scaled_training_data)
    # learning_df, cv_df, test_df = process_training_data(training_data)
    print('\n3. learning and CV data sets organised')
    # print(learning_df)
    # print(cv_df)
    # print(test_df)

    feature_df = fill_feature_df(learning_df)
    print('\n4. feature df created and filled')
    # print(feature_df)

    eps_min = learn_epsillon(feature_df, learning_df)
    print('\n5. minimum episllon found in training set of ', eps_min)
    # print(eps_min)

    eps_min, incorrect_eps, incorrect_anom = validate_epsillon(eps_min, feature_df, cv_df)
    print('\n6. epsillon cross validated')
    if len(incorrect_eps) > 0:
        print('A new lower value of eps_min was found ', eps_min)
        print('The data that updated eps_min was:')
        print(incorrect_eps)
    if len(incorrect_anom) > 0:
        print('###### WARNING ######')
        print('The following anomalies were not detected by epsillon')
        print(incorrect_anom)

    eps_min, incorrect_eps, incorrect_anom = test_epsillon(eps_min, feature_df, test_df)
    print('\n7. epsillon tested and incorrect detection is below:')
    if len(incorrect_eps) > 0:
        print('A new lower value of eps_min was found ', eps_min)
        print('The data that updated eps_min was:')
        print(incorrect_eps)
    if len(incorrect_anom) > 0:
        print('###### WARNING ######')
        print('The following anomalies were not detected by epsillon')
        print(incorrect_anom)

def get_data(filename):
    '''Opens a csv with name filename and returns a pandas df.'''
    return pd.read_csv(filename)

def scale_features(training_data):
    '''Scales features using the StandardScaler method, drops the class column
    before scaling and then picks it up again once data is scaled.'''

    scaled_training_data = training_data.drop('class', 1)
    scaler = StandardScaler()
    scaled_training_data = pd.DataFrame(scaler.fit_transform(scaled_training_data), columns=scaled_training_data.columns)
    scaled_training_data['class'] = training_data['class'].values
    return scaled_training_data

def process_training_data(scaled_training_data):
    '''Takes the scaled data and eperates it into the learning, CV and testing
    datasets. First the data is split into anomalous and non-anomalous data
    sets, then the learning, CV and testing datasets are created usinf the
    sklearn train_test_split function.'''

    non_anom_df = scaled_training_data[scaled_training_data['class'] == 'clean']
    anom_df = scaled_training_data[scaled_training_data['class'] != 'clean']

    cv_df, test_df = train_test_split(anom_df, test_size=0.5)
    learning_df, temp_df = train_test_split(non_anom_df, test_size=0.2)
    temp_cv_df, temp_test_df = train_test_split(temp_df, test_size=0.5)
    cv_df = cv_df.append(temp_cv_df, ignore_index=True)
    test_df = test_df.append(temp_test_df, ignore_index=True)

    return learning_df, cv_df, test_df

def fill_feature_df(learning_df):
    '''takes the learning dataset and uses it to determine the mean and standard
    deciation of each column of data. These features are stored in the feature
    dataframe which is returned.'''

    feature_df_labels = (list(learning_df.columns))
    feature_df_labels.pop()
    feature_df = pd.DataFrame(index=feature_df_labels, columns=['mean', 'SD'])

    for column in feature_df_labels:
        feature_df.loc[column, 'mean'] = learning_df[column].mean()
        feature_df.loc[column, 'SD'] = statistics.stdev(learning_df[column])

    return feature_df

def learn_epsillon(feature_df, df):
    '''This function takes a learning data set and iterates through to determine
    the minimum value of epsillon that is not considered anomalous. The minimum
    epsillon value is returned.'''

    feature_df_labels = (list(feature_df.index))
    eps_min = math.inf

    for index, row in df.iterrows():
        eps_curr = 1
        for item in feature_df_labels:
            eps_curr = eps_curr * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[item,'SD'])))*(math.exp(-((row[item]-feature_df.loc[item,'mean'])**2)/(2*(feature_df.loc[item,'SD'])))))
        if eps_curr < eps_min:
            eps_min = eps_curr

    return eps_min

def validate_epsillon(eps_min, feature_df, cv_df):
    '''The current learned minimum epsillon value and the feature dataset are
    used to determine if there is a non-anomalous example that would be
    incorrectly deteccted by the current epsillon value. If some non-anomalous
    data is detected as anomalous, then epsillon is updated. Additionally, the
    anomalous data is checked to ensure that is has an epsillon value smaller
    than the current minimum epsillon value. An updated eps_min is returned
    as well as any incorrectly detected examples.'''

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

def test_epsillon(eps_min, feature_df, test_df):
    '''Epsillon is tested using the final set of data and any incorrect
    detections are recorded and returned in a dataframe.'''

    non_anom_df = test_df[test_df['class'] == 'clean']
    anom_df = test_df[test_df['class'] != 'clean']
    feature_df_labels = (list(feature_df.index))
    incorrect_eps = pd.DataFrame(columns=feature_df_labels)
    incorrect_anom = pd.DataFrame(columns=feature_df_labels)

    for index, row in non_anom_df.iterrows():
        eps = 1
        for label in feature_df_labels:
            eps = eps * ((1/((math.sqrt(2*math.pi))*math.sqrt(feature_df.loc[label,'SD'])))*(math.exp(-((row[label]-feature_df.loc[label,'mean'])**2)/(2*(feature_df.loc[label,'SD'])))))
        # If a lower eps value is found, it is updated and the row and old value stored
        if eps < eps_min:
            incorrect_eps = incorrect_eps.append(non_anom_df.loc[index, :], ignore_index=True)
            print('NOTE: a lower value of eps was found: ', eps)


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

if __name__ == "__main__":
    main()
