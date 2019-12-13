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


# Update 13/12/2019

I created a csv with 500 entries of data generated from a sample of 1 week of NEM Watch data. Then I manually adjusted the final three entries of the 500 data sets and added 0kW consumption for 2 data periods randomly. I then used the logistic regression classifier on the generated data to try and train a model that could determine power outages. Upon testing the data against a subsection of manually edited data this rough test correctly identified the power outages on 2 out of 3 power outages and was correct on 2 out of 2 non-power outage data. Where the model was wrong in detecting a power outage, the 0kW incident occurred on a time that had not previously been trained with a 0kW incident.

TO improve this model I would create a model trained on a training set that had at least one training example with a 0kW on each time stamp.  

The data was created by sampling a full week of data and calculating the average and std dev of each time period in a day. The formula new_val = avg + (std dev * random.uniform(0,1)) to create a random data set with noise.

IMPROVEMENTS:
- I need to clean up the code, add more functions and more comments.
- I need to add command line variable for ease of use
- I need to add some functionality to automate the generation of anomalous operation
