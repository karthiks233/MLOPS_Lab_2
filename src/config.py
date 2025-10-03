"""
Shared configuration for MLOps pipeline
"""
import datetime

# Common settings
TIMESTAMP = datetime.datetime.now().strftime("%y%m%d")
MLFLOW_TRACKING_URI = "./mlruns"

# Dataset configurations
SYNTHETIC_DATA_CONFIG = {
    "n_samples": (1, 2000),
    "n_features": 6,
    "n_informative": 3,
    "n_redundant": 0,
    "n_repeated": 0,
    "n_classes": 2,
    "random_state": 42,
    "shuffle": True
}

# Model configurations
MODELS = {
    "RandomForest": {
        "class": "RandomForestClassifier",
        "params": {"random_state": 42}
    },
    "DecisionTree": {
        "class": "DecisionTreeClassifier", 
        "params": {"random_state": 42}
    }
}

# File paths
PATHS = {
    "data": "data/",
    "models": "models/",
    "metrics": "metrics/",
    "comparison_models": "models/comparison/"
}

# Experiment names
EXPERIMENTS = {
    "main": "MLOPS_LAB_2",
    "comparison": "Reuters_Corpus_Volume"
}
