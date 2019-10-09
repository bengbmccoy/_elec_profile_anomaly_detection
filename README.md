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

