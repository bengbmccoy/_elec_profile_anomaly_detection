# _elect_profile_anomaly_detection
A repo to build and train an anomaly detection model that can detect unusual operation for electricity profiles.


# Anomaly Detection
I will be using generated electricity profile data to learn what the mean and variance of each data point throughout a day to build a model of what is expected operation of an electricity profile.

I will then use the anomaly detection algorithm to detect when anomalous operation occurs.
Algorithm:
- P(x;mew,sigma^2) = 1/(((2*pi)^0.5)*sigma)*exp(-(((x-mew)^2)/(2*sigma^2)))

Finally I will use the cross validation dataset to determine a value for epsillon that is the cutoff between regular and anomalous operation.
- if P(x) > epsilon, then register an anomaly, epsilon will be determiend using the cross validation dataset.

The feature vector table will look like the below, before it is normalised:

| index   | x1  | x2   | ... | x24  | y   |
|---------|-----|------|-----|------|-----|
| Date1   | 2   | 2    | ... | 2    | 0   |
| Date2   | 2.5 | 2.5  | ... | 2    | 0   |
| ...     | ... | ...  | ... | ...  | ... |
| Date999 | 2.1 | 2.05 | ... | 2.25 | 0   |

The table of mew & sigma^2 will look like the below, and be stored as a csv within this repo:

| feature | mew | sigma^2 |
|---------|-----|---------|
| x1      | 2   | 0.5     |
| x2      | 2   | 0.4     |
| ...     | ... | ...     |
| x24     | 2.1 | 0.4     |

# Generating Data
As I do not have a wealth of data for this project, I will be generating it, essentially from scratch. The generation of data will be done in two stages: 1. generating "regular" data, 2. generating anomalous data.

To generate the "regular" regular I will use a vanilla base profile and then use randomisation to increase the variance, eventually creating approximatley 1000 regular profiles with no anomalies. This dataset will be split into 60:20:20, with 60% being used for training/learning the mew and sigma parameters, 20% being used in cross validation and the last 20% in testing.

To generate the anomalous profiles I will do each of these by hand and will create several dozen profiles. These will be combined into the cross validation and testing data sets, but not into the training dataset.

Example command line uasge:

to generate 7000 days of clean data:
python .\gen_data.py .\example_data.csv -data_type clean -new_filename clean.csv -pr -days_gen 7000

to generate 7000 days of outage data:
python .\gen_data.py .\20191201_OpenNEM.csv -data_type outage -new_filename outage.csv -pr -days_gen 7000

to run the unittest script (verbose):
python -m unittest test_gen_data -v


# Update 13/12/2019

I created a csv with 500 entries of data generated from a sample of 1 week of NEM Watch data. Then I manually adjusted the final three entries of the 500 data sets and added 0kW consumption for 2 data periods randomly. I then used the logistic regression classifier on the generated data to try and train a model that could determine power outages. Upon testing the data against a subsection of manually edited data this rough test correctly identified the power outages on 2 out of 3 power outages and was correct on 2 out of 2 non-power outage data. Where the model was wrong in detecting a power outage, the 0kW incident occurred on a time that had not previously been trained with a 0kW incident.

TO improve this model I would create a model trained on a training set that had at least one training example with a 0kW on each time stamp.  

The data was created by sampling a full week of data and calculating the average and std dev of each time period in a day. The formula new_val = avg + (std dev * random.uniform(0,1)) to create a random data set with noise.

IMPROVEMENTS:
- I need to clean up the code, add more functions and more comments.
- I need to add command line variable for ease of use
- I need to add some functionality to automate the generation of anomalous operation

# Update 15/12/2019

I have done a number of things since the last update:

1. The 'clean' data generator works and has been tested
2. the 'outage' data generator is complete and randomly assigns 0kW values to the data. Then the function tests each day to ensure there is at least one 0kW value per day, if there is no a 0kW value is randomly added to the day.
3. The data was used to train a logistic regression model that turned out good results, however I did not train the model with enough outage data and the data didn't account for outages at every time period of the day. I fixed this issue by creating much more outage data to train the model, ensuring that each time period would be represented.
4. The command line arguments are working, as well as the plot and print command line arguments.
5. I added some basic unittests, but will expand on these in the near future.

# Update 18/12/2019

I ran 500 days of clean data and 500 days of outage data through the logistic regression classifier and then 5 days of outage data to see if the model could learn to classify outages. It could not. However, I determined that it was becuase the data I was generating was different to the data that I was test and expecting to detect an outage.

The training data had an outage rate of approximately 20% (of all data points, not on 20% of days) which was far higher than the 4% outage rate that I was using to test the data. I reworked the script to include only one outage per day for the training data. When I tested the new data on the logistic regression classifier it worked with 5 out of 5 testing data sets.

I understand that this is just a simple test and more testing data would be required. However I was pleased to have seen where there was an issue and have ammened the code to include a more reasonable outage rate of roughly 8% (2 outages per day).

I ran a second test of the model, but this time using 15000 training examples (50/50 outage/clean) and managed to get an even better result than the test above!
