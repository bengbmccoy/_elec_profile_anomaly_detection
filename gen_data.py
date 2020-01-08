'''
Written by Ben McCoy, Dec 2019

This code takes sample data and from it generates additional data that is
either clean (matches the example data with added noise), outages (has at least
one 0kW reading per day of data) or demand spike (has at least one major demand
spike per day that is at least double regular consumptioin).

Example command line uasge:

to generate 7000 days of clean data:
python gen_data.py example_data.csv -data_type clean -new_filename clean.csv -pr -days_gen 7000
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('example_data', type=str,
                        help='example data filepath')
    parser.add_argument('-data_type', default='clean',
                        help='generate "clean" data or "outage" data')
    parser.add_argument('-days_gen', nargs='?', type=int, default=500,
                        help='the number of days of data to generate, default is 500')
    parser.add_argument('-new_filename', type=str,
                        help='the generated data filepath')
    parser.add_argument('-pr', '--print',
                        help='prints each new day of data',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plots the cost function over time',
                        action='store_true')
    args = parser.parse_args()

    test_data = open_csv(args.example_data)
    test_data['ID'] = test_data['ProfileName'] + test_data['ProfileArea']
    id_list = set(test_data['ID'].tolist())
    columns = test_data.columns
    good_cols = []

    for col in columns:
        if str(col[0:4]) == 'Peri' or str(col) == 'ID' or str(col) == 'SettlementDate':
            good_cols.append(col)

    drop_cols = list(set(columns) - set(good_cols))
    for col in drop_cols:
        test_data.drop(col, axis=1, inplace=True)

    df_dict = {}
    for id in id_list:
        df_dict[id] = test_data[test_data['ID'] == id]
    # print(df_dict)

    test_data = df_dict['CLOADNSWCECOUNTRYENERGY']

    avg_dict, sd_dict = get_val_dict(test_data)

    # parse number of days to generate
    days_to_gen = args.days_gen
    data_type = args.data_type

    gen_data = gen_aemo_data(avg_dict, sd_dict, days_to_gen)


    # # Sum the data into a total column
    # test_data['Total - MW'] = test_data.sum(axis=1)
    #
    # val_dict = get_empty_val_dict(test_data)
    # val_dict = fill_val_dict(test_data, val_dict)
    # avg_val_dict = get_avg_val_dict(val_dict)
    # sd_val_dict = get_sd_val_dict(val_dict)
    #
    # # generate all of the new data
    # gen_data = gen_new_data(data_type, val_dict, avg_val_dict, sd_val_dict, days_to_gen)
    # gen_data.mean().plot()
    # plt.show()

    if args.print:
        print(gen_data)

    if args.plot:
        # gen_data.T.plot()
        gen_data.mean().plot()
        plt.show()

    # Save CSV
    save_csv(gen_data, args.new_filename)

def gen_aemo_data(avg_dict, sd_dict, days_to_gen):

    new_data_df = pd.DataFrame(columns=list(avg_dict.keys()))
    for column in list(avg_dict.keys()):
        new_data_df[column] = np.random.normal(avg_dict[column], sd_dict[column], days_to_gen)
    new_data_df['class'] = 'clean'
    return new_data_df

def gen_new_data(data_type, val_dict, avg_val_dict, sd_val_dict, days_to_gen):
    # creates new data based on the avg and std dev of a time period and then
    # appends new data to a pandas dataframe
    # Random 0s are added by generating a randint between 0 and 10 and if the
    # int is below 2, then a 0 is added to that value.

    # create the new pandas whcih will comtain generated data
    gen_data = pd.DataFrame(columns=list(val_dict.keys()))

    # Generate noisy data with no outages
    if data_type == 'clean':
        for i in range(days_to_gen):
            new_data = {}
            for key, value in val_dict.items():
                new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
            gen_data = gen_data.append(new_data, ignore_index=True)

    # Randomly add 0kW values to the data, or generate noisy data using the
    # average and standard deviation.
    if data_type == 'outage':
        for i in range(days_to_gen):
            new_data = {}
            for key, value in val_dict.items():
                if random.randint(0,100) < 5:
                    new_data[key] = 0
                else:
                    new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
            gen_data = gen_data.append(new_data, ignore_index=True)
        # Ensure every day has at least one 0kW value
        for i in range(len(gen_data)):
            if 0 not in (gen_data.loc[i,:].values.tolist()):
                gen_data.iloc[i,random.randint(0,len(gen_data.loc[i,:].values.tolist())-1)] = 0

    if data_type == 'demand_spike':
        for i in range(days_to_gen):
            new_data = {}
            for key, value in val_dict.items():
                new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
            gen_data = gen_data.append(new_data, ignore_index=True)
        for i in range(len(gen_data)):
            for j in range(random.randint(1,4)):
                gen_data.iloc[i,random.randint(0,len(gen_data.loc[i,:].values.tolist())-1)] = (float(avg_val_dict[key]) * random.uniform(2,5)) + (float(sd_val_dict[key]) * random.uniform(-1,1))

    if data_type == 'clean':
        gen_data['class'] = 'clean'
    elif data_type == 'outage':
        gen_data['class'] = 'outage'
    elif data_type == 'demand_spike':
        gen_data['class'] = 'demand_spike'
    return gen_data

def get_sd_val_dict(sd_dict):
    # create a dict with key as times and val as Std Dev of total - MW
    sd_val_dict = {}
    for key, value in sd_dict.items():
        sd_val_dict[key] = statistics.stdev(value)
    return sd_val_dict

def get_avg_val_dict(val_dict):
    # create a dict with key as times and val as mean of total - MW
    avg_val_dict = {}
    for key, value in val_dict.items():
        avg_val_dict[key] = statistics.mean(value)
    return avg_val_dict

def fill_val_dict(test_data, val_dict):
    # appends total - MW values to the empty lists
    for index, row in test_data.iterrows():
        val_dict[str(index)[11:16]].append(row['Total - MW'])
    return val_dict

def get_val_dict(test_data):
    # Get dict of mean values of time periods and stdev of time periods, data
    # to be extraceted is from aemo profile data

    sd_dict = {}
    avg_dict = {}
    for period in list(test_data.drop(columns=['SettlementDate', 'ID']).columns):
        avg_dict[str(period)[-2:]] = test_data[period].mean()
        sd_dict[str(period)[-2:]] = statistics.stdev(test_data[period])

    return avg_dict, sd_dict

def get_empty_val_dict(test_data):
    # get a dictionary of empty lists, the key of each list is a time period
    sd_dict = {}
    times = []
    timestamps = test_data.index.tolist()
    for i in timestamps:
        if str(i)[11:16] not in sd_dict:
            times.append(str(i)[11:16])
            sd_dict[str(i)[11:16]] = []
    return sd_dict

def save_csv(df, file_name):
    df.to_csv(file_name, index=False)

def open_csv(file_name):
    # opens the dataframe with the index columns as the first column and the
    # values parsed as dates
    return pd.read_csv(file_name, parse_dates=True)

if __name__ == "__main__":
    main()
