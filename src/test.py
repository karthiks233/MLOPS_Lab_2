
from sklearn.datasets import fetch_rcv1
import mlflow, datetime, os, pickle, random
import sklearn
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

# rcv1 = fetch_rcv1()
# pickle.dump(rcv1.data, open('../data/data.pickle', 'wb'))
# pickle.dump(rcv1.target, open('../data/target.pickle', 'wb'))
# Check if the file exists within the folder
if os.path.exists('../data/'): 
    X = pickle.load(open('../data/data.pickle', 'rb'))
    y = pickle.load(open('../data/target.pickle', 'rb'))
    y = y.toarray()[:, random.randint(0, 3)]

else:
    rcv1 = fetch_rcv1()
    pickle.dump(rcv1.data, open('../data/data.pickle', 'wb'))
    pickle.dump(rcv1.target, open('../data/target.pickle', 'wb'))  
    X = rcv1.data  
    y = rcv1.target.toarray()[:, random.randint(0, 3)]

mlflow.set_tracking_uri("./mlruns")
dataset_name = "Test3-MLOPS-LAB-2"
current_time = datetime.datetime.now().strftime("%y%m%d")
experiment_name = f"{dataset_name}_{current_time}"    
experiment_id = mlflow.create_experiment(f"{experiment_name}")

with mlflow.start_run(experiment_id=experiment_id,
                      run_name= f"{dataset_name}"):
    
    params = {
                "dataset_name": dataset_name,
                "number of datapoint": X.shape[0],
                "number of dimensions": X.shape[1]}
    
    mlflow.log_params(params)
    
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, 
                                                                                test_size = 0.9,
                                                                                shuffle = True)
    
    
    dt = joblib.load('models/model_251002_dt_model.joblib')
    dt.fit(train_X, train_y)
    
    y_predict = dt.predict(test_X)
    mlflow.log_metrics({'Accuracy': accuracy_score(test_y, y_predict),
                        'F1 Score': f1_score(test_y, y_predict)})

    metrics = {'Accuracy': accuracy_score(test_y, y_predict),
                        'F1 Score': f1_score(test_y, y_predict)}
    
    if os.path.exists('models/'): 
        print(f"Saving model to {experiment_id}.joblib")
        dump(dt, f'models/{current_time}_{experiment_id}.joblib')

    if os.path.exists('metrics/'):
        with open(f'metrics/{current_time}_{experiment_id}_metrics.json', 'w') as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
                       
 