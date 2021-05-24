
[![Klarna logo](assets/logo.png)](https://https://www.klarna.com/us/)

# Klarna Case Study | Data Science
## **Otto von Sperling** | 18th to 25th of May, 2021

## 1. Problem definition
By and large, the case study asks that we predict the probability that a given user will default their next payment.
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
We will make our model available through AWS Lambda. Amazon SageMaker endpoints was the first option but I have to admit I am a bit rusty with it. It started to take too much time to make the container work with all the sagemaker requirements, so I decided to go trhough a route that I am more used to at the moment. However, I will be digging back into SageMaker endpoints over the weekend.

Nevertheless, the endpoint is available at:

> https://cumnecdgaa.execute-api.eu-west-2.amazonaws.com/dev/api/v1/default_risk/predict

The expected payload is a json containing at least the following items:
```python
{
    "headers": {
        "Authorization": "klarna-case-study"
    },
    "data": """'[{"uuid":"1229c83c-6338-4c4b-a20f-065ecca45b4a",
                  "account_amount_added_12_24m":28472,
                  "account_days_in_dc_12_24m":0.0,
                  "account_days_in_rem_12_24m":0.0,
                  "account_days_in_term_12_24m":0.0,
                  "account_incoming_debt_vs_paid_0_24m":0.0,
                  "account_status":1.0,
                  "account_worst_status_0_3m":1.0,
                  "account_worst_status_12_24m":1.0,
                  "account_worst_status_3_6m":1.0,
                  "account_worst_status_6_12m":1.0,
                  "age":29,
                  "avg_payment_span_0_12m":8.24,
                  "avg_payment_span_0_3m":7.8333333333,
                  "merchant_category":"Diversified electronics",
                  "merchant_group":"Electronics",
                  "has_paid":true,
                  "max_paid_inv_0_12m":37770.0,
                  "max_paid_inv_0_24m":37770.0,
                  "name_in_email":"F1+L",
                  "num_active_div_by_paid_inv_0_12m":0.037037037,
                  "num_active_inv":1,
                  "num_arch_dc_0_12m":0,
                  "num_arch_dc_12_24m":0,
                  "num_arch_ok_0_12m":25,
                  "num_arch_ok_12_24m":16,
                  "num_arch_rem_0_12m":0,
                  "num_arch_written_off_0_12m":0.0,
                  "num_arch_written_off_12_24m":0.0,
                  "num_unpaid_bills":1,
                  "status_last_archived_0_24m":1,
                  "status_2nd_last_archived_0_24m":1,
                  "status_3rd_last_archived_0_24m":1,
                  "status_max_archived_0_6_months":1,
                  "status_max_archived_0_12_months":1,
                  "status_max_archived_0_24_months":1,
                  "recovery_debt":0,
                  "sum_capital_paid_account_0_12m":116,
                  "sum_capital_paid_account_12_24m":27874,
                  "sum_paid_inv_0_12m":265347,
                  "time_hours":14.1708333333,
                  "worst_status_active_inv":1.0}]'"""
}
```
It's quite easy to generate the expected data format with Pandas. All it takes is:
```python
import pandas as pd

pd.read_csv("dataset.csv").to_json(orient="records")
```

The route is capable of handling multiple requests at once or one at a time.
The output is a string in the same json format as the input, and can be easily transformed into back into a DataFrame:
```python
pd.read_json(response.content, orient="records")
```

## 6. Running the Experiment

> Before anything else, you must create a `data/` directory on the top-level dir and add the `dataset.csv` file for it all to work.

It's quite simple to run the project. First you must navigate to the top-level directory and build the Docker image:
```bash
docker build -t klarna-case-study -f ./Dockerfile .
```

Then, run it interactively with bash:
```bash
docker run klarna-case-study:latest
```

After that, all you need to do is run the project:
```bash
python build_and_run_project_tree.py
```

Otherwise, if you want to run the notebooks without fussing around with docker, you can take the following steps.
First you must navigate to the top-level of the project and install the package manager Poetry:
```bash
python -m pip install poetry
```

Then, let poetry do the heavy lifting (it may take a little while):
```bash
python -m poetry install
```

And that's it. You may now spin jupyter and explore the notebooks for yourslef.
