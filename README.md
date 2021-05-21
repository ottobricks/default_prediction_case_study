## Klarna Case Study | Otto von Sperling | 18th to 25th of May, 2021

# [WorkInProgress]

# Problem definition
By and large, we are asked to predict the probability that a given user will default their next payment.
Unfortunately, we are not provided with a time series that leads to each transaction, but instead are
provided with pre-computed variables that describe such series.

The first of Klarna's leadership principles is "customer obsession", which leads us
to the conclusion that a smooth experience for the customer is king. Thus, we must keep incorrect warnings to a minimum so that the customer
is not bothered by waves of notifications or even calls from our agents. Ikes, that would be a terrible experience.

## Metrics for our business goal
**TALK ABOUT thresholds for the predictions, how to set and tune them.**
To bridge this experiment with a clear business goal, we can take advantage of the previous ideas and formulate KPIs that reflect our business goals.

1. Flag as "in risk of default" no more than 5% of customers incorrectly.
2. Maximize the number of clients correctly flagged as "in risk of default"

# Experiment Overview

1. Sanity Check
2. Exploration
3. Feature Engineering
4. Baseline Model
5. Hyperparameter Tunning
6. Results and Evaluation

Each part of the experiment will have it's own dedicated folder for the sake of clarity.
