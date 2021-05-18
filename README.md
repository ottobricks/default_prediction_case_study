## Klarna Case Study | Otto von Sperling | 18th to 25th of May, 2021

# [WorkInProgress]

# Problem definition
By and large, we are asked to predict the probability that a given user will default their next payment.
Unfortunately, we are not provided with a time series that leads to each transaction, but instead are
provided with pre-computed features that describe such series.

It goes without saying that not all defaults are equal. The larger the amount that goes unpaid, the larger the risk exposure we face.
Thus, it seems reasonable to take into account the amount of money that is still owed by a customer when predicting the risk score.
Despite this not being the actual problem statement, which asks us to focus on the probability of defaults, we will compare results at the end,
and hopefully conclude that a more nuanced approach can lead to greater success.

Not only of risk exposure management are made good solutions. The first of Klarna's leadership principles is "customer obsession", which leads us
to the conclusion that a smooth experience for the customer is king. Thus, we must keep incorrect warnings to a minimum so that the customer
is not bothered by waves of notifications or even calls from our agents. Ikes, that would be a terrible experience.

## Metrics for our business goal
To bridge this experiment with a clear business goal, we can take advantage of the previous ideas and formulate KPIs that reflect our business goals.

1. Amount of money correctly flagged as in risk of default
2. Number of times we incorrectly flag a user as in risk of default

# Experiment Overview

1. Sanity Check
2. Exploration
3. Feature Selection
4. Baseline Model
5. Hyperparameter Tunning
6. Results and Evaluation

Each part of the experiment will have it's own dedicated folder for the sake of clarity.
