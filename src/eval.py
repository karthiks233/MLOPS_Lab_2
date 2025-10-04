import pickle, os, json, random
from sklearn.metrics import f1_score, accuracy_score
import joblib, glob, sys
import argparse
from sklearn.datasets import make_classification
import datetime

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    # Add timestamp argument - From GitHub Actions

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp

    # timestamp = datetime.datetime.now().strftime("%y%m%d")
    try:
        # Try to load model with current timestamp first
        model_version = f'models/model_{timestamp}'
        model_rf = joblib.load(f'{model_version}_rf_model.joblib')
        model_dt = joblib.load(f'{model_version}_dt_model.joblib')
        model_gbc = joblib.load(f'{model_version}_gbc_model.joblib')
        print(f"Loaded model: {model_version}.joblib")
    except:
        try:
            # If current timestamp fails, find the most recent model files
            model_files_rf = glob.glob('models/model_*_rf_model.joblib')
            model_files_dt = glob.glob('models/model_*_dt_model.joblib')
            model_files_gbc = glob.glob('models/model_*_gbc_model.joblib')
            
            if not model_files_rf or not model_files_dt or not model_files_gbc:
                raise FileNotFoundError('Required model files not found')
            
            # Get the most recent model files
            latest_rf = max(model_files_rf, key=os.path.getctime)
            latest_dt = max(model_files_dt, key=os.path.getctime)
            latest_gbc = max(model_files_gbc, key=os.path.getctime)
            
            model_rf = joblib.load(latest_rf)
            model_dt = joblib.load(latest_dt)
            model_gbc = joblib.load(latest_gbc)
            
            print(f"Loaded latest models: RF={latest_rf}, DT={latest_dt}, GBC={latest_gbc}")
        except Exception as e:
            raise ValueError(f'Failed to load models: {str(e)}')
        
    try:
        # Check if the file exists within the folder
        X, y = make_classification(
                            n_samples=random.randint(100, 2000),
                            n_features=6,
                            n_informative=3,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            random_state=0,
                            shuffle=True,
                        )
    except Exception as e:
        raise ValueError(f'Failed to create the data: {str(e)}')
    
    y_predict_rf = model_rf.predict(X)
    y_predict_dt = model_dt.predict(X)
    y_predict_gbc = model_gbc.predict(X)
    metrics = {"F1_Score_RandomForest":f1_score(y, y_predict_rf),
               "F1_Score_DecisionTree":f1_score(y, y_predict_dt),
               "F1_Score_GradientBoosting":f1_score(y, y_predict_gbc),
               "Accuracy_RandomForest":accuracy_score(y, y_predict_rf),
               "Accuracy_DecisionTree":accuracy_score(y, y_predict_dt),
               "Accuracy_GradientBoosting":accuracy_score(y, y_predict_gbc)}
    
    # Save metrics to a JSON file

    if not os.path.exists('metrics/'): 
        # then create it.
        os.makedirs("metrics/")
        
    with open(f'metrics/{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
               
    