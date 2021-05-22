import subprocess

subprocess.run("mkdir -p ./data/train/preprocessed".split())
subprocess.run("mkdir -p ./data/test".split())
subprocess.run("mkdir -p ./data/predict".split())
subprocess.run("mkdir -p ./ml_artifacts".split())