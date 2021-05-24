import os
import subprocess

print("Building project tree...")
subprocess.run("mkdir -p ./data/train/preprocessed".split())
subprocess.run("mkdir -p ./data/test".split())
subprocess.run("mkdir -p ./data/predict".split())
subprocess.run("mkdir -p ./ml_artifacts/gridsearch_results".split())
print("\n\tDone building!\n")

print("Preparing to run experiment steps...")
subprocess.run("poetry run jupytext --to py:percent src/0_split_data/split_train_validation_test.ipynb".split())
subprocess.run("poetry run jupytext --to py:percent src/3_feature_engineering/1_define_preprocessors.ipynb".split())
subprocess.run("poetry run jupytext --to py:percent src/3_feature_engineering/2_transform_data.ipynb".split())
subprocess.run("poetry run jupytext --to py:percent src/4_baseline_model/1_train_and_score.ipynb".split())
subprocess.run("poetry run jupytext --to py:percent src/5_tuned_model/*.ipynb".split())
subprocess.run("poetry run jupytext --to py:percent src/6_results/*.ipynb".split())
print("\n\tDone preparing!\n")

print("Running script to hold out test set...")
subprocess.run("poetry run python ./split_train_validation_test.py".split(), cwd=f"{os.getcwd()}/src/0_split_data")
print("\n\tDone with hold out script!\n")

print("Running script to define preprocessors...")
subprocess.run("poetry run python ./1_define_preprocessors.py".split(), cwd=f"{os.getcwd()}/src/3_feature_engineering/")
print("\n\tDone with preprocessors script!\n")

print("Running script to transform data...")
subprocess.run("poetry run python ./2_transform_data.py".split(), cwd=f"{os.getcwd()}/src/3_feature_engineering/")
print("\n\tDone with data transformation script!\n")

print("Running script to fit baseline and score validation...")
subprocess.run("poetry run python ./1_train_and_score.py".split(), cwd=f"{os.getcwd()}/src/4_baseline_model/")
print("\n\tDone with baseline script!\n")

print("Running script to handle imbalanced_data...")
subprocess.run("poetry run python ./1_handle_imbalance.py".split(), cwd=f"{os.getcwd()}/src/5_tuned_model/")
print("\n\tDone with imbalance script!\n")

print("Running script to select models...")
subprocess.run("poetry run python ./2_select_hyperparams_and_models.py".split(), cwd=f"{os.getcwd()}/src/5_tuned_model/")
print("\n\tDone with model selection script!\n")

print("Running script to fit best model and score data...")
subprocess.run("poetry run python ./1_train_and_score.py".split(), cwd=f"{os.getcwd()}/src/6_results/")
print("\n\tDone with scoring script!\n")