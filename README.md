
[![Klarna logo](assets/logo.png)](https://https://www.klarna.com/us/)

# Klarna Case Study | Data Science
## **Otto von Sperling** | 18th to 25th of May, 2021

## 1. Problem definition
By and large, we are asked to predict the probability that a given user will default their next payment.
Altough it would have been nice to work with customers' time series data, we are provided with pre-computed variables that describe such series.

The first of Klarna's leadership principles is "customer obsession", which leads us
to the conclusion that a smooth experience for the customer is king. Thus, we must keep incorrect warnings to a minimum so that the customer
is not bothered by waves of notifications or even calls from our agents.

## 2. Metrics for our business goal
**TALK ABOUT thresholds for the predictions, how to set and tune them.**
In order to guarantee a smooth experience for our customers, we score and evaluate our models via threshold analysis. That is, the model outputs predictions for the likelyhood of customers defaulting and we follow up with an ad-hoc selection of thresholds to decide whether to flag observations. Such thresholds will be selected based on two key performance indicators:

1. Flag as "in risk of default" no more than 5% of customers incorrectly.
2. Maximize the number of clients correctly flagged as "in risk of default"

## 3. Experiment Overview

1. Sanity Check - check consistency, missing values and macro behaviour.
2. Exploration - test hypothesis and select variables
3. Feature Engineering - create preprocessor module
4. Baseline Model - score a simple baseline model
5. Tuned Model - search hyperparameters and models
6. Results and Evaluation - score best model and present results

Each part of the experiment will have it's own dedicated folder for the sake of clarity.

## 4. Results
Throughtout our experiment, we show that testing hypothsesis and thinking about your data are fundamental steps so as to have any chance at success in a prediction task. This was most sucessfully highlighted during our custom undersample strategy to handle class imbalance in our target label that took our baseline performance to a much better level.

In the end, we fit a model that respects our 1st KPI and only blocks 3% of users incorrectly but is able to identify nearly 30% of defaults correctly (at predicted_risk > 0.85). Whilst this result are far from great, we believe it achieves the goal of having a reasonable enough model. Of course, much more effort can go towards hyperparameter tunning to squeeze any potential gains and choosing different models, but we believe exploration to be more relevant in the context of a case study, thus more time was spent on it.

All in all, we believe this is a fun project with some schallenges with regards to data quality that enables one to come up with creative solutions. Not all steps are perfect but major experiment decisions were given a good amount of thought, considering the 1 week span of this exercise. We (I, myself and my coffee mug) will be happy to discuss both technical and philosophical details of our methods and implementation.

## 5. Deployment
In order to make our best model available, we will use two strategies of deployment, the second one being a fall-back option.
The first way we make our model available is through Amazon SageMaker endpoints. Requests should be sent to the following IP/port/route:

> 192.168.0.1:80/api/v1/default-risk/predict

The expected payload is a json containing at least the following items:
```python
{
    "age": int,
    "status_last_archived_0_24m": float or int,
    "account_worst_status_0_3m": float or int,
    "account_worst_status_3_6m": float or int,
    "account_worst_status_6_12m": float or int,
    "merchant_category": str,
    "num_arch_dc_0_12m": int,
    "num_active_div_by_paid_inv_0_12m": int
}
```
The fall-back strategy of deployment will be a Flask API runnig on an EC2 instance on AWS. The expected payload is the same as above and requests should be sent to:

> 192.168.0.1:80/api/v1/default-risk/predict