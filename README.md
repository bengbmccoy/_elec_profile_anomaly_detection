# _elect_profile_anomaly_detection
A repo to build and train an anomaly detection model that can detect unusual operation for electricity profiles.


# Anomaly Detection
I will be using generated electricity profile data to learn what the mean and variance of each data point throughout a day to build a model of what is expected operation of an electricity profile.

I will then use the anomaly detection algorithm to detect when anomalous operation occurs.
Algorithm: 
- P(x;mew,sigma^2) = 1/(((2*pi)^0.5)*sigma)*exp(-(((x-mew)^2)/(2*sigma^2)))

Finally I will use the cross validation dataset to determine a value for epsillon that is the cutoff between regular and anomalous operation.
- if P(x) > epsilon, then register an anomaly, epsilon will be determiend using the cross validation dataset.

The feature vector table will look like the below:

| index   | x1  | x2   | ... | x24  |
|---------|-----|------|-----|------|
| Date1   | 2   | 2    | ... | 2    |
| Date2   | 2.5 | 2.5  | ... | 2    |
| ...     | ... | ...  | ... | ...  |
| Date999 | 2.1 | 2.05 | ... | 2.25 |

