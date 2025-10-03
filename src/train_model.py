# from sklearn.datasets import fetch_rcv1
import mlflow, datetime, os, pickle, random
import sklearn
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import argparse
import datetime

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    # Add timestamp argument - From GitHub Actions

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    # args = parser.parse_args()
    
    # Access the timestamp
    # timestamp = args.timestamp

    timestamp = datetime.datetime.now().strftime("%y%m%d")
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Check if the file exists within the folder
    X, y = make_classification(
                            n_samples=random.randint(1, 2000),
                            n_features=6,
                            n_informative=3,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            random_state=0,
                            shuffle=True,
                        )
    if not os.path.exists('data/'): 
        os.makedirs('data/')
    
    with open('data/data.pickle', 'wb') as data:
        pickle.dump(X, data)
        
    with open('data/target.pickle', 'wb') as data:
        pickle.dump(y, data)  
            
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "MLOPS_LAB_2"
    current_time = datetime.datetime.now().strftime("%y%m%d")
    experiment_name = f"{dataset_name}_{current_time}"    
    
    try:
        experiment_id = mlflow.create_experiment(f"{experiment_name}")
    except mlflow.exceptions.MlflowException:
        # Experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):
        
        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1]}
        
        mlflow.log_params(params)
            
        
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X, y)


        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, y)

        gbc =  GradientBoostingClassifier(random_state=0)
        gbc.fit(X, y)


        y_predict_forest = forest.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict_forest),
                            'F1 Score': f1_score(y, y_predict_forest)})


        y_predict_dt = dt.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict_dt),
                            'F1 Score': f1_score(y, y_predict_dt)})

        y_predict_gbc = gbc.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict_gbc),
                            'F1 Score': f1_score(y, y_predict_gbc)})
        
        if not os.path.exists('models/'): 
            # then create it.
            os.makedirs("models/")
            
        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename_rf = f'models/{model_version}_rf_model.joblib'
        dump(dt, model_filename_rf)
        model_filename = f'models/{model_version}_dt_model.joblib'
        dump(gbc, model_filename)
        model_filename = f'models/{model_version}_gbc_model.joblib'
        dump(forest, model_filename)
                    