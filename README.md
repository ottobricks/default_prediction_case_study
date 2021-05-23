
[![Klarna logo](assets/logo.png)](https://https://www.klarna.com/us/)

# Klarna Case Study | Data Science
## **Otto von Sperling** | 18th to 25th of May, 2021

## Problem definition
By and large, we are asked to predict the probability that a given user will default their next payment.
Altough it would have been nice to work with customers' time series data, we are provided with pre-computed variables that describe such series.

The first of Klarna's leadership principles is "customer obsession", which leads us
to the conclusion that a smooth experience for the customer is king. Thus, we must keep incorrect warnings to a minimum so that the customer
is not bothered by waves of notifications or even calls from our agents.

## Metrics for our business goal
**TALK ABOUT thresholds for the predictions, how to set and tune them.**
In order to guarantee a smooth experience for our customers, we score and evaluate our models via threshold analysis. That is, the model outputs predictions for the likelyhood of customers defaulting and we follow up with an ad-hoc selection of thresholds to decide whether to flag observations. Such thresholds will be selected based on two key performance indicators:

1. Flag as "in risk of default" no more than 10% of customers incorrectly.
2. Maximize the number of clients correctly flagged as "in risk of default"

## Experiment Overview

1. Sanity Check - check consistency, missing values and macro behaviour.
2. Exploration - test hypothesis and select variables
3. Feature Engineering - create preprocessor module
4. Baseline Model - score a simple baseline model
5. Tuned Model - search hyperparameters and models
6. Results and Evaluation - score best model and present results

Each part of the experiment will have it's own dedicated folder for the sake of clarity.

## Deployment
In order to make our best model available, we will use two strategies of deployment, the second one being a fall-back option.
The first way we make our model available is through Amazon SageMaker endpoints. Requests should be sent to the following IP/port/route:

> 192.168.0.1:80/api/v1/default-risk/predict

The expected payload is a json containing at least the following items:
