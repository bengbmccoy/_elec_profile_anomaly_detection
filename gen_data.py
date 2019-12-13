# hello_world

import pandas as pd
import matplotlib.pyplot as plt
import random
import statistics
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('example_data', type=str,
                        help='example data filepath')
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
    # print(test_data)

    test_data['Total - MW'] = test_data.sum(axis=1)
    test_day = test_data.head(48)
    totals_list = test_day['Total - MW'].tolist()
    days_to_gen = args.days_gen

    # get a dictionary of empty lists, the key of each list is a time period
    sd_dict = {}
    times = []
    timestamps = test_data.index.tolist()
    for i in timestamps:
        if str(i)[11:16] not in sd_dict:
            times.append(str(i)[11:16])
            sd_dict[str(i)[11:16]] = []
    # print(sd_dict)

    # appends total - MW values to the empty lists
    for index, row in test_data.iterrows():
        sd_dict[str(index)[11:16]].append(row['Total - MW'])
    # print(sd_dict)

    # create a dict with key as times and val as mean of total - MW
    avg_val_dict = {}
    for key, value in sd_dict.items():
        avg_val_dict[key] = statistics.mean(value)
    # print(avg_val_dict)

    # create a dict with key as times and val as Std Dev of total - MW
    sd_val_dict = {}
    for key, value in sd_dict.items():
        sd_val_dict[key] = statistics.stdev(value)
    # print(sd_val_dict)

    # create the new pandas whcih will comtain generated data
    gen_data = pd.DataFrame(columns=list(sd_dict.keys()))
    # print(gen_data)

    # creates new data based on the avg and std dev of a time period.
    # appends new data to a CSV and saves it
    for i in range(days_to_gen):
        new_data = {}
        for key, value in sd_dict.items():
            new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
        # print(new_data)
        gen_data = gen_data.append(new_data, ignore_index=True)


    # Add class columns of 0s
    gen_data['class'] = 0
    if args.print:
        print(gen_data)

    # Save CSV
    save_csv(gen_data, args.new_filename)

    if args.plot:
        gen_data.T.plot()
        plt.show()

def save_csv(df, file_name):
    df.to_csv(file_name)

def open_csv(file_name):
    ''' opens the dataframe with the index columns as the first column and the
    values parsed as dates
    '''

    return pd.read_csv(file_name, index_col=0, parse_dates=True)

main()
