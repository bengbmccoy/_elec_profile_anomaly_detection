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
    # print(test_data)

    # parse number of days to generate
    days_to_gen = args.days_gen
    data_type = args.data_type

    # Sum the data into a total column
    test_data['Total - MW'] = test_data.sum(axis=1)

    val_dict = get_empty_val_dict(test_data)
    val_dict = fill_val_dict(test_data, val_dict)
    avg_val_dict = get_avg_val_dict(val_dict)
    sd_val_dict = get_sd_val_dict(val_dict)

    # create the new pandas whcih will comtain generated data
    gen_data = pd.DataFrame(columns=list(val_dict.keys()))

    # generate all of the new data
    gen_data = gen_new_data(data_type, gen_data, val_dict, avg_val_dict, sd_val_dict, days_to_gen)


    if args.print:
        print(gen_data)

    if args.plot:
        gen_data.T.plot()
        plt.show()

    # Save CSV
    save_csv(gen_data, args.new_filename)

def gen_new_data(data_type, gen_data, val_dict, avg_val_dict, sd_val_dict, days_to_gen):
    # creates new data based on the avg and std dev of a time period.
    # appends new data to a CSV and saves it
    # Random 0s are added by generating a randint between 0 and 10 and if the
    # int is below 2, then a 0 is added to that value.
    if data_type == 'outage':
        for i in range(days_to_gen):
            new_data = {}
            for key, value in val_dict.items():
                if random.randint(0,10) < 1:
                    new_data[key] = 0
                else:
                    new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
            gen_data = gen_data.append(new_data, ignore_index=True)
        for i in range(len(gen_data)):
            if 0 not in (gen_data.loc[i,:].values.tolist()):
                gen_data.iloc[i,random.randint(0,len(gen_data.loc[i,:].values.tolist()))] = 0

    if data_type == 'clean':
        for i in range(days_to_gen):
            new_data = {}
            for key, value in val_dict.items():
                new_data[key] = float(avg_val_dict[key]) + (float(sd_val_dict[key]) * random.uniform(-1,1))
            gen_data = gen_data.append(new_data, ignore_index=True)

    if data_type == 'clean':
        gen_data['class'] = 0
    else:
        gen_data['class'] = 1
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
    df.to_csv(file_name)

def open_csv(file_name):
    # opens the dataframe with the index columns as the first column and the
    # values parsed as dates
    return pd.read_csv(file_name, index_col=0, parse_dates=True)

if __name__ == "__main__":
    main()
